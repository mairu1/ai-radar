"""Inspect public source formats before choosing collectors."""
import concurrent.futures
import json
import gzip
import re
import urllib.request
import xml.etree.ElementTree as ET

URLS = [
    "https://openai.com/news/rss.xml",
    "https://deepmind.google/blog/rss.xml",
    "https://deepmind.google/blog/",
    "https://blog.google/technology/ai/rss/",
    "https://huggingface.co/blog/feed.xml",
    "https://blogs.nvidia.com/feed/",
    "https://www.microsoft.com/en-us/research/feed/",
    "https://ai.meta.com/blog/rss/",
    "https://ai.meta.com/blog/",
    "https://www.anthropic.com/news",
    "https://www.anthropic.com/research",
    "https://mistral.ai/news/",
    "https://www.anthropic.com/sitemap.xml",
    "https://mistral.ai/sitemap.xml",
    "https://export.arxiv.org/rss/cs.AI",
    "https://rss.arxiv.org/rss/cs.AI",
    "https://www.anthropic.com/news/claude-haiku-5-5",
    "https://huggingface.co/api/models?author=Qwen&sort=createdAt&direction=-1&limit=2&full=true&cardData=true",
]


def probe(url):
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AI-Radar/0.1"})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read(8_000_000)
        if data.startswith(b"\x1f\x8b"):
            data = gzip.decompress(data)
        if b"<html" in data[:500].lower():
            text = data.decode("utf-8", errors="replace")
            links = re.findall(r'<a\b[^>]*href=[\"\']([^\"\']+)', text)
            return {"url": url, "meta": re.findall(r'<meta\b[^>]*>', text)[:25],
                    "jsonld": re.findall(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', text)[:1],
                    "dates": re.findall(r'.{0,50}(?:datePublished|published_time|datetime=|Oct 7, 2026|October 6, 2026).{0,100}',text)[:5],
                    "links": list(dict.fromkeys(x for x in links if any(p in x for p in ['/news/', '/research/', '/blog/'])))[:10]}
        if "api/models" in url:
            models = json.loads(data)
            return {"url": url, "models": [
                {k: m.get(k) for k in ["id", "createdAt", "lastModified", "pipeline_tag", "cardData", "siblings"]}
                for m in models
            ]}
        root = ET.fromstring(data)
        items = [e for e in root.iter() if e.tag.split("}")[-1] == "item"]
        return {"url": url, "root": root.tag, "items": len(items),
                "sample": [{c.tag: c.text for c in item if c.tag in ["title", "link", "pubDate"]}
                           for item in items[:1]],
                "urls": [e.text for e in root.iter() if e.tag.endswith("loc")][:3]}
    except ET.ParseError:
        text = data.decode("utf-8", errors="replace")
        return {"url": url, "html_bytes": len(data),
                "head": text[:100], "meta": re.findall(r'<meta\b[^>]*>', text)[:25],
                "jsonld": re.findall(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', text)[:1],
                "times": re.findall(r'<time[^>]*>.*?</time>', text)[:2],
                "links": re.findall(r'<a\b[^>]*href=[\"\']([^\"\']+)', text)[:45]}
    except Exception as error:
        return {"url": url, "error": str(error)}


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(probe, URLS):
            print(json.dumps(result))
