"""Collect a small AI digest from public primary sources. Python 3.11+, no packages."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import gzip
from html import escape, unescape
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

UTC = timezone.utc
ROOT = Path(__file__).resolve().parent
SECTIONS = {"announcements": "Company announcements", "models": "Models with published weights", "research": "Research & engineering"}
START, END = "<!-- radar:start -->", "<!-- radar:end -->"
LOG = logging.getLogger("radar")
MAX_BYTES = 12_000_000


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "AI-Radar/1.0 (public-source digest)", "Accept-Encoding": "gzip"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                data = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError("Response exceeds size limit")
            # Some endpoints send gzip even without a Content-Encoding header.
            if data.startswith(b"\x1f\x8b"):
                data = gzip.decompress(data)
            if len(data) > MAX_BYTES:
                raise ValueError("Decompressed response exceeds size limit")
            return data
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(1 + attempt)
    raise RuntimeError("Request failed")


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            result = parsedate_to_datetime(value)
        except (ValueError, TypeError):
            result = None
        if result is None:
            for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
                try:
                    result = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    pass
    return result.replace(tzinfo=UTC) if result and result.tzinfo is None else result.astimezone(UTC) if result else None


def canonical_url(url: str) -> str:
    parts = urlsplit(unescape(url.strip()))
    if parts.scheme not in {"https", "http"} or not parts.netloc:
        raise ValueError("Entry URL must be HTTP(S)")
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid", "_fb_noscript"}]
    path = quote(parts.path.rstrip("/") or "/", safe="/%:@!$&'()*+,;=-._~")
    if parts.netloc.lower() == "arxiv.org":
        path = re.sub(r"v\d+$", "", path)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(sorted(query)), ""))


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.links: list[tuple[str, str]] = []
        self.text: list[str] = []
        self.times: list[str] = []
        self.jsonld: list[str] = []
        self._anchor: list[str] | None = None
        self._href = ""
        self._script: list[str] | None = None
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "style"}:
            self._skip += 1
            if tag == "script" and attrs.get("type") == "application/ld+json":
                self._script = []
        if tag == "meta":
            key = attrs.get("property") or attrs.get("name")
            if key and attrs.get("content"):
                self.meta[key.lower()] = attrs["content"]
        if tag == "time" and attrs.get("datetime"):
            self.times.append(attrs["datetime"])
        if tag == "a":
            self._href = attrs.get("href", "")
            self._anchor = []

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_data(self, data):
        if self._script is not None:
            self._script.append(data)
        if not self._skip:
            self.text.append(data)
            if self._anchor is not None:
                self._anchor.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._script is not None:
            self.jsonld.append("".join(self._script))
            self._script = None
        if tag in {"script", "style"}:
            self._skip = max(0, self._skip - 1)
        if tag == "a" and self._anchor is not None:
            self.links.append((self._href, " ".join(self._anchor)))
            self._anchor = None


def plain_text(value: str) -> str:
    parser = PageParser()
    parser.feed(value)
    return " ".join(unescape(" ".join(parser.text)).split())


def excerpt(value: str, words: int = 24) -> str:
    text = plain_text(value)
    if text.lower() in {'tbd', 'todo', 'coming soon', 'description', 'n/a'}:
        return ''
    parts = text.split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def entry(source: dict, title: str, url: str, date: datetime, summary: str = "", **details) -> dict:
    url = canonical_url(url)
    return {"id": url, "title": plain_text(title), "url": url, "source": source["name"],
            "category": source["category"], "published": date.isoformat(),
            "summary": excerpt(summary), **details}


def in_window(date: datetime | None, now: datetime, days: int) -> bool:
    return date is not None and now - timedelta(days=days) <= date <= now


def matches(text: str, keywords: list[str]) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.I) for word in keywords)


def child_text(element, names):
    for name in names:
        for child in element:
            if child.tag.rsplit("}", 1)[-1] == name:
                if name == "link" and child.get("href"):
                    if child.get("rel", "alternate") == "alternate":
                        return child.get("href")
                elif child.text:
                    return child.text
    return ""


def collect_feed(source: dict, config: dict, now: datetime, seen: set[str]) -> list[dict]:
    root = ET.fromstring(fetch(source["url"]))
    if root.tag.rsplit("}", 1)[-1] not in {"rss", "RDF", "feed"}:
        raise ValueError("Source returned something other than RSS/Atom")
    items = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] in {"item", "entry"}]
    if not items:
        raise ValueError("Feed contains no entries")
    result = []
    for item in items:
        title = child_text(item, ["title"])
        date = parse_date(child_text(item, ["pubDate", "published", "date"]))
        # Atom updates are only a fallback when the publisher provides no publication date.
        if not date:
            date = parse_date(child_text(item, ["updated"]))
        if not in_window(date, now, config["lookback_days"]):
            continue
        url = child_text(item, ["link"])
        if not title or not url:
            continue
        url = canonical_url(url)
        summary = child_text(item, ["description", "summary", "encoded", "content"])
        if source.get("keywords") and not matches(title + " " + plain_text(summary), source["keywords"]):
            continue
        if source.get("preprint") and not matches(title, config["research_keywords"]):
            continue
        if source["name"] == "Hugging Face":
            # HF's feed includes community posts; only follow the configured publishers.
            path = urlsplit(url).path.strip("/").split("/")
            if len(path) > 2 and path[1].lower() not in {a.lower() for a in config["huggingface_authors"]}:
                continue
        result.append(entry(source, title, url, date, summary, preprint=source.get("preprint", False)))
    result.sort(key=lambda item: (item["published"], item["id"]), reverse=True)
    selected = result[:config["max_research_per_run"] if source.get("preprint") else config["max_per_source"]]
    return [item for item in selected if item["id"] not in seen]


def find_publication_date(parser: PageParser) -> datetime | None:
    for key in ("article:published_time", "datepublished", "date", "pubdate"):
        date = parse_date(parser.meta.get(key))
        if date:
            return date

    def walk(value):
        if isinstance(value, dict):
            if value.get("@type") in {"Article", "NewsArticle", "BlogPosting", "ScholarlyArticle"} and value.get("datePublished"):
                yield value["datePublished"]
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for raw in parser.jsonld:
        try:
            for value in walk(json.loads(raw)):
                date = parse_date(value)
                if date:
                    return date
        except (ValueError, TypeError):
            continue
    for value in parser.times:
        date = parse_date(value)
        if date:
            return date
    # Anthropic and Meta currently print the publication date as ordinary body text.
    text = " ".join(parser.text)
    match = re.search(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+20\d{2}\b", text)
    return parse_date(match.group()) if match else None


def collect_page(source: dict, config: dict, now: datetime, seen: set[str]) -> list[dict]:
    listing = PageParser()
    listing.feed(fetch(source["url"]).decode("utf-8"))
    candidates: dict[str, str] = {}
    for href, title in listing.links:
        if not href:
            continue
        try:
            url = canonical_url(urljoin(source["url"], href))
        except ValueError:
            continue
        parts = urlsplit(url)
        if parts.netloc != urlsplit(source["url"]).netloc or not parts.path.startswith(source["path_prefix"]):
            continue
        if parts.path == source["path_prefix"].rstrip("/") or "/team/" in parts.path:
            continue
        candidates.setdefault(url, title)
    if not listing.links:
        raise ValueError("No links found in source HTML")
    result, dated, failures = [], 0, 0
    for url, label in list(candidates.items())[:24]:
        try:
            parser = PageParser()
            parser.feed(fetch(url).decode("utf-8"))
            date = find_publication_date(parser)
            dated += date is not None
            if not in_window(date, now, config["lookback_days"]):
                continue
            title = parser.meta.get("og:title") or parser.meta.get("title") or label
            if not title:
                continue
            summary = parser.meta.get("og:description") or parser.meta.get("description", "")
            listing_summary = listing.meta.get("og:description") or listing.meta.get("description", "")
            if summary == listing_summary:
                summary = ""  # Generic site copy is not an article summary.
            result.append(entry(source, title, url, date, summary, preprint=False))
        except (OSError, ValueError) as error:
            failures += 1
            LOG.warning("Article unavailable: %s (%s)", url, error)
    if candidates and not dated:
        raise ValueError("Unable to extract publication dates; source markup may have changed")
    if failures:
        LOG.warning("%s: %d article requests failed", source["name"], failures)
    result.sort(key=lambda item: (item["published"], item["id"]), reverse=True)
    return [item for item in result[:config["max_per_source"]] if item["id"] not in seen]


def model_entry(model: dict, author: str, config: dict, now: datetime) -> dict | None:
    date = parse_date(model.get("createdAt"))
    if not in_window(date, now, config["lookback_days"]) or model.get("private") or model.get("disabled"):
        return None
    model_id = model.get("id", "")
    if not model_id.startswith(author + "/"):
        return None
    if re.search(r"(?:^|[-_])checkpoint(?:[-_]|$)", model_id, re.I):
        return None
    card = model.get("cardData") or {}
    if config.get("require_model_license", True) and not card.get("license"):
        return None
    tags = {str(tag).lower() for tag in model.get("tags", [])}
    if tags & {"peft", "lora", "merge", "merged", "gguf", "gptq", "awq", "mlx"}:
        return None
    weights = [item.get("rfilename", "") for item in model.get("siblings", [])
               if item.get("rfilename", "").lower().endswith((".safetensors", ".bin", ".pt", ".pth", ".gguf"))
               and "adapter" not in item.get("rfilename", "").lower()]
    if not weights:
        return None
    license_id = card.get("license", "not specified")
    if isinstance(license_id, list):
        license_id = ", ".join(map(str, license_id))
    if license_id == "other":
        license_id = card.get("license_name", "custom (see model card)")
    return entry({"name": author, "category": "models"}, model_id,
                 "https://huggingface.co/" + quote(model_id, safe="/"), date,
                 license=str(license_id), task=model.get("pipeline_tag") or "not specified",
                 gated=model.get("gated", False), weight_files=len(weights))


def collect_models(author: str, config: dict, now: datetime, seen: set[str]) -> list[dict]:
    params = urlencode({"author": author, "sort": "createdAt", "direction": -1, "limit": 40, "full": "true", "cardData": "true"})
    models = json.loads(fetch("https://huggingface.co/api/models?" + params))
    if not isinstance(models, list):
        raise ValueError("Unexpected Hugging Face API response")
    result = []
    for model in models:
        item = model_entry(model, author, config, now)
        if item:
            result.append(item)
    result.sort(key=lambda item: (item["published"], item["id"]), reverse=True)
    return [item for item in result[:config["max_models_per_author"]] if item["id"] not in seen]


def collect(config: dict, now: datetime, seen: set[str]) -> tuple[list[dict], list[str], int]:
    jobs = [(source["name"], collect_feed, source) for source in config["feeds"]]
    jobs += [(source["name"], collect_page, source) for source in config["pages"]]
    jobs += [("HF / " + author, collect_models, author) for author in config["huggingface_authors"]]
    entries, errors, successful = {}, [], 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fn, source, config, now, seen): name for name, fn, source in jobs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                successful += 1
                LOG.info("%s: %d new entries", name, len(result))
                for item in result:
                    # Stable choice when two feeds link to the same article.
                    previous = entries.get(item["id"])
                    if previous is None or item["source"] < previous["source"]:
                        entries[item["id"]] = item
            except Exception as error:
                message = f"{name}: {error}"
                errors.append(message)
                LOG.warning("Source failed: %s", message)
    return sorted(entries.values(), key=lambda item: (item["published"], item["id"]), reverse=True), sorted(errors), successful


def md(value) -> str:
    text = escape(str(value), quote=False)
    return re.sub(r"([\\`*\[\]_|])", r"\\\1", " ".join(text.split()))


def event_key(item: dict) -> str:
    if item['category'] == 'models':
        return item['id']
    title = re.sub(r'\W+', ' ', item['title'].lower()).strip()
    return item['category'] + ':' + item['published'][:10] + ':' + title


def render_digest(day: str, items: list[dict], limit: int | None = None) -> str:
    lines = [f"# AI Radar — {day}", "", "Automatically collected from linked sources. Dates are UTC publication dates; model dates are repository creation dates.",
             "Publisher excerpts reflect their claims. Research entries are not independently validated.", ""]
    for category, title in SECTIONS.items():
        selected = sorted((item for item in items if item["category"] == category), key=lambda item: (item["published"], item["id"]), reverse=True)
        if limit:
            selected = selected[:limit]
        if not selected:
            continue
        lines.extend([f"## {title}", ""])
        for item in selected:
            # Angle brackets prevent spaces/parentheses in source URLs breaking Markdown.
            lines.extend([f"### [{md(item['title'])}](<{item['url']}>)", "",
                          f"{md(item['source'])} · {item['published'][:10]}" + (" · **Preprint**" if item.get("preprint") else ""), ""])
            if item["summary"]:
                lines.extend([f"> {md(item['summary'])}", ""])
            if category == "models":
                lines.extend([f"Task: {md(item['task'])} · License: {md(item['license'])} · Access: {'gated (approval or terms may apply)' if item['gated'] else 'ungated'}", "",
                              "Weight files found in the repository. Check its model card and license before use.", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8", newline="\n")
    temp.replace(path)
    return True


def publish(root: Path, new: list[dict], now: datetime) -> int:
    if not new:
        return 0
    readme = (root / "README.md").read_text(encoding="utf-8")
    if readme.count(START) != 1 or readme.count(END) != 1 or readme.index(START) >= readme.index(END):
        raise ValueError("README must contain one ordered pair of radar markers")
    state_path = root / "data" / "entries.json"
    existing = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else []
    by_id = {item["id"]: item for item in existing}
    event_keys = {event_key(item) for item in existing}
    added = []
    for item in new:
        if item['id'] not in by_id and event_key(item) not in event_keys:
            added.append(item)
            event_keys.add(event_key(item))
    if not added:
        return 0
    day = now.date().isoformat()
    for item in added:
        by_id[item["id"]] = {**item, "discovered": day}
    all_items = sorted(by_id.values(), key=lambda item: (item["published"], item["id"]), reverse=True)
    daily = [item for item in all_items if item["discovered"] == day]
    latest = render_digest(day, all_items, limit=8).replace(f"# AI Radar — {day}", "## Latest from AI Radar", 1)
    # README headings need to be nested under the digest section.
    latest = re.sub(r"^## (?!Latest)(.*)$", r"### \1", latest, flags=re.M)
    latest = re.sub(r"^### \[", "#### [", latest, flags=re.M)
    archives = sorted({item["discovered"] for item in all_items}, reverse=True)
    index = "# Digest archive\n\nEach file groups entries by the UTC date they were first collected.\n\n" + "\n".join(f"- [{date}]({date}.md)" for date in archives) + "\n"
    before, after = readme.split(START, 1)[0], readme.split(END, 1)[1]
    block = f"{START}\n{latest}\n[Browse all daily digests](digests/README.md).\n{END}"
    # Persist state last: interrupted output rendering can be retried safely.
    write_if_changed(root / "digests" / f"{day}.md", render_digest(day, daily))
    write_if_changed(root / "digests" / "README.md", index)
    write_if_changed(root / "README.md", before + block + after)
    write_if_changed(state_path, json.dumps(all_items, ensure_ascii=False, indent=2) + "\n")
    return len(added)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--dry-run", action="store_true", help="Fetch sources and print a preview without changing files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    state = ROOT / "data" / "entries.json"
    existing = json.loads(state.read_text(encoding="utf-8")) if state.exists() else []
    now = datetime.now(UTC)
    new, errors, successful = collect(config, now, {item["id"] for item in existing})
    if not successful:
        LOG.error("All sources failed; existing files were preserved")
        return 1
    if args.dry_run:
        print(render_digest(now.date().isoformat(), new))
    else:
        LOG.info("Published %d new entries; %d sources failed", publish(ROOT, new, now), len(errors))
    if errors:
        LOG.warning("Coverage is partial: %d sources succeeded, %d failed", successful, len(errors))
    # The workflow can publish valid results and then flag partial coverage for attention.
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
