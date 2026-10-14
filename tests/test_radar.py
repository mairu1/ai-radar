import copy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import radar

NOW = datetime(2026, 10, 14, 12, tzinfo=timezone.utc)
CONFIG = json.loads((radar.ROOT / 'config.json').read_text(encoding='utf-8'))
SOURCE = {'name': 'Example', 'category': 'announcements', 'url': 'https://example.com/feed'}
RSS = b'''<rss version="2.0"><channel>
<item><title>New model</title><link>https://example.com/new?utm_source=rss</link><pubDate>Tue, 13 Oct 2026 10:00:00 GMT</pubDate><description>&lt;p&gt;A model &amp;amp; a tool.&lt;/p&gt;</description></item>
<item><title>Old model</title><link>https://example.com/old</link><pubDate>Tue, 01 Sep 2026 10:00:00 GMT</pubDate></item>
<item><title>Future model</title><link>https://example.com/future</link><pubDate>Tue, 20 Oct 2026 10:00:00 GMT</pubDate></item>
<item><title>No date</title><link>https://example.com/no-date</link></item>
</channel></rss>'''


class DatesAndURLs(unittest.TestCase):
    def test_date_formats_and_missing_dates(self):
        for value in ('2026-10-14T12:00:00Z', 'Wed, 14 Oct 2026 12:00:00 GMT', 'October 14, 2026', 'Oct 14, 2026'):
            self.assertEqual(radar.parse_date(value).date(), NOW.date())
        self.assertIsNone(radar.parse_date('not a date'))
        self.assertIsNone(radar.parse_date(None))

    def test_url_dedup_and_markdown_safe_url(self):
        self.assertEqual(radar.canonical_url('https://example.com/model/?utm_source=x&a=1#section'), 'https://example.com/model?a=1')
        self.assertEqual(radar.canonical_url('https://arxiv.org/abs/2610.12345v2'), 'https://arxiv.org/abs/2610.12345')
        self.assertNotIn('>', radar.canonical_url('https://example.com/a>bad'))
        with self.assertRaises(ValueError):
            radar.canonical_url('javascript:alert(1)')


class Feeds(unittest.TestCase):
    @patch('radar.fetch', return_value=RSS)
    def test_window_and_seen_entries(self, _fetch):
        items = radar.collect_feed(SOURCE, CONFIG, NOW, set())
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['url'], 'https://example.com/new')
        self.assertEqual(items[0]['summary'], 'A model & a tool.')
        self.assertEqual(radar.collect_feed(SOURCE, CONFIG, NOW, {'https://example.com/new'}), [])

    @patch('radar.fetch', return_value=b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Release</title><link rel="self" href="https://example.com/api"/><link rel="alternate" href="https://example.com/release"/><published>2026-10-13T00:00:00Z</published></entry></feed>''')
    def test_atom(self, _fetch):
        self.assertEqual(radar.collect_feed(SOURCE, CONFIG, NOW, set())[0]['url'], 'https://example.com/release')

    @patch('radar.fetch', return_value=b'<html><body>Access denied</body></html>')
    def test_failed_response_is_reported(self, _fetch):
        with self.assertRaises(ValueError):
            radar.collect_feed(SOURCE, CONFIG, NOW, set())

    def test_keyword_boundaries(self):
        self.assertFalse(radar.matches('Annual chair exhibition', ['AI']))
        self.assertTrue(radar.matches('New AI inference engine', ['AI']))

    @patch('radar.fetch')
    def test_seen_top_entry_does_not_backfill_older_items(self, mock_fetch):
        mock_fetch.return_value = RSS.replace(b'</channel>', b'<item><title>Other recent item</title><link>https://example.com/other</link><pubDate>Mon, 12 Oct 2026 10:00:00 GMT</pubDate></item></channel>')
        config = {**CONFIG, 'max_per_source': 1}
        self.assertEqual(radar.collect_feed(SOURCE, config, NOW, {'https://example.com/new'}), [])


class ArticlePages(unittest.TestCase):
    def test_dates_and_scripts(self):
        parser = radar.PageParser()
        parser.feed('''<script type="application/ld+json">{"@type":"BlogPosting","datePublished":"2026-10-06","dateModified":"2026-10-14"}</script><p>October 14, 2026</p>''')
        self.assertEqual(radar.find_publication_date(parser).date().isoformat(), '2026-10-06')
        self.assertNotIn('datePublished', ' '.join(parser.text))
        parser = radar.PageParser()
        parser.feed('<p>Research</p><p>October 7, 2026</p>')
        self.assertEqual(radar.find_publication_date(parser).date().isoformat(), '2026-10-07')

    @patch('radar.fetch')
    def test_page_same_host_filter_and_generic_description(self, mock_fetch):
        source = {'name': 'Example', 'category': 'announcements', 'url': 'https://example.com/news', 'path_prefix': '/news/'}
        listing = b'''<meta name="description" content="Generic site copy"><a href="mailto:press@example.com">Contact</a><a href="/news/new">A release</a><a href="https://other.com/news/new">Wrong host</a><a href="/news/new">Duplicate</a>'''
        article = b'''<meta property="og:title" content="New model"><meta name="description" content="Generic site copy"><time datetime="2026-10-13"></time>'''
        mock_fetch.side_effect = [listing, article]
        items = radar.collect_page(source, CONFIG, NOW, set())
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['summary'], '')
        self.assertEqual(mock_fetch.call_count, 2)


class Models(unittest.TestCase):
    def setUp(self):
        self.model = {'id': 'Qwen/New', 'createdAt': '2026-10-13T00:00:00Z', 'lastModified': '2026-10-14T00:00:00Z',
                      'siblings': [{'rfilename': 'model.safetensors'}], 'tags': [], 'cardData': {'license': 'other', 'license_name': 'custom-research'}, 'gated': 'auto'}

    def test_weights_license_and_creation_date(self):
        item = radar.model_entry(self.model, 'Qwen', CONFIG, NOW)
        self.assertEqual(item['license'], 'custom-research')
        self.assertEqual(item['published'][:10], '2026-10-13')
        self.assertTrue(item['gated'])

    def test_old_updated_placeholder_and_adapters_excluded(self):
        model = copy.deepcopy(self.model)
        model['createdAt'] = '2026-01-01T00:00:00Z'
        self.assertIsNone(radar.model_entry(model, 'Qwen', CONFIG, NOW))
        model = copy.deepcopy(self.model)
        model['siblings'] = [{'rfilename': 'README.md'}]
        self.assertIsNone(radar.model_entry(model, 'Qwen', CONFIG, NOW))
        model['siblings'] = [{'rfilename': 'adapter_model.safetensors'}]
        self.assertIsNone(radar.model_entry(model, 'Qwen', CONFIG, NOW))
        model = copy.deepcopy(self.model)
        model['tags'] = ['peft']
        self.assertIsNone(radar.model_entry(model, 'Qwen', CONFIG, NOW))
        model = copy.deepcopy(self.model)
        model['cardData'] = {}
        self.assertIsNone(radar.model_entry(model, 'Qwen', CONFIG, NOW))
        model = copy.deepcopy(self.model)
        model['id'] = 'Qwen/New-checkpoint-15000'
        self.assertIsNone(radar.model_entry(model, 'Qwen', CONFIG, NOW))


class Publishing(unittest.TestCase):
    def test_no_churn_next_day_and_same_day_append(self):
        with tempfile.TemporaryDirectory(dir=radar.ROOT, prefix='.radar-test-') as temp:
            root = Path(temp)
            self.assertTrue(root.resolve().is_relative_to(radar.ROOT.resolve()))
            readme = root / 'README.md'
            readme.write_text('# Test\n\n<!-- radar:start -->\n<!-- radar:end -->\n\nSetup stays here.\n', encoding='utf-8')
            first = radar.entry(SOURCE, 'New model', 'https://example.com/a', NOW, 'Publisher summary.')
            second = radar.entry(SOURCE, 'New tool', 'https://example.com/b', NOW)
            self.assertEqual(radar.publish(root, [first], NOW), 1)
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(radar.publish(root, [first], NOW), 0)
            crosspost = {**first, 'id': 'https://other.com/a', 'url': 'https://other.com/a', 'source': 'Other'}
            self.assertEqual(radar.publish(root, [crosspost], NOW), 0)
            next_day = NOW.replace(day=15)
            self.assertEqual(radar.publish(root, [], next_day), 0)
            self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()})
            self.assertEqual(radar.publish(root, [second], NOW), 1)
            daily = (root / 'digests' / '2026-10-14.md').read_text(encoding='utf-8')
            self.assertIn('New model', daily)
            self.assertIn('New tool', daily)
            self.assertIn('Setup stays here.', readme.read_text(encoding='utf-8'))
            self.assertEqual(len(json.loads((root / 'data' / 'entries.json').read_text(encoding='utf-8'))), 2)

    def test_bad_markers_dont_write_state(self):
        with tempfile.TemporaryDirectory(dir=radar.ROOT, prefix='.radar-test-') as temp:
            root = Path(temp)
            self.assertTrue(root.resolve().is_relative_to(radar.ROOT.resolve()))
            (root / 'README.md').write_text('No markers', encoding='utf-8')
            with self.assertRaises(ValueError):
                radar.publish(root, [radar.entry(SOURCE, 'Title', 'https://example.com/a', NOW)], NOW)
            self.assertFalse((root / 'data').exists())


class Network(unittest.TestCase):
    @patch('radar.urlopen')
    def test_gzip_response(self, mock_open):
        response = MagicMock()
        response.read.return_value = gzip.compress(RSS)
        mock_open.return_value.__enter__.return_value = response
        self.assertEqual(radar.fetch('https://example.com/feed'), RSS)

    @patch('radar.collect_feed', side_effect=ValueError('Source down'))
    def test_all_sources_fail(self, _collect_feed):
        config = {**CONFIG, 'feeds': [SOURCE], 'pages': [], 'huggingface_authors': []}
        entries, errors, successful = radar.collect(config, NOW, set())
        self.assertEqual(entries, [])
        self.assertEqual(successful, 0)
        self.assertEqual(len(errors), 1)


if __name__ == '__main__':
    unittest.main()
