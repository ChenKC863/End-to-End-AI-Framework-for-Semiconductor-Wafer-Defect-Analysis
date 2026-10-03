"""Dataset routing tests; no running API or Ollama required."""
import contextlib
import io
import json
import itertools
from pathlib import Path
import runpy
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_knowledge_live.py'


class LiveDatasetTests(unittest.TestCase):
    def run_script(self, *args):
        with patch('sys.argv', [str(SCRIPT), *args]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(str(SCRIPT), run_name='__main__')

    def test_live_requests_and_report_use_selected_dataset(self):
        for dataset, model in itertools.product(('S', 'M'), ('llama3.2:3b', 'qwen2.5-coder:7b')):
            with self.subTest(dataset=dataset, model=model), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / 'report.json'
                client = MagicMock()
                client.post.return_value.status_code = 200
                client.post.return_value.json.return_value = {'trace': []}
                with patch('httpx.Client') as factory:
                    factory.return_value.__enter__.return_value = client
                    self.run_script('--database', dataset, '--model', model, '--output', str(output))
                self.assertEqual(client.post.call_count, 4)
                self.assertTrue(all(call.kwargs['json']['database'] == dataset for call in client.post.call_args_list))
                self.assertEqual(json.loads(output.read_text())['database'], dataset)
                self.assertTrue(all(call.kwargs['json']['model'] == model for call in client.post.call_args_list))
                self.assertEqual(json.loads(output.read_text())['model'], model)

    def test_check_only_selects_saved_database_and_supports_legacy_s(self):
        for dataset in ('S', 'M'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as folder:
                db = Path(folder) / 'fixture.db'
                with contextlib.closing(sqlite3.connect(db)) as conn:
                    conn.execute('CREATE TABLE wafers(image_path TEXT, anomaly_score REAL, true_label TEXT, pred_label TEXT)')
                    conn.execute("INSERT INTO wafers VALUES('image.jpg', -0.1, 'Donut', 'Donut')")
                    conn.commit()
                response = {'citations': ['K1'], 'generation_status': 'extractive_fallback'}
                report = {'cases': [{'status': 200, 'response': dict(response)} for _ in range(4)]}
                if dataset == 'M':
                    report['database'] = dataset
                    report['model'] = 'qwen2.5-coder:7b'
                    for case in report['cases']:
                        case['response']['model'] = report['model']
                report['cases'][2]['response'].update(answer='1 record', sql_result={'database': dataset, 'rows': [[1]]})
                report['cases'][3]['response'].update(sql_result={'database': dataset, 'rows': [['image.jpg', -0.1]]}, trace=[{'tool': tool, 'status': 'ok'} for tool in ('search_knowledge', 'query_wafer_sql')])
                output = Path(folder) / 'report.json'
                output.write_text(json.dumps(report), encoding='utf-8')
                connect = sqlite3.connect
                with patch('sqlite3.connect', side_effect=lambda *a, **k: connect(db)) as mocked:
                    with self.assertRaises(SystemExit) as stopped:
                        self.run_script('--check-only', '--output', str(output))
                    self.assertEqual(stopped.exception.code, 0)
                    self.assertIn(f'wafer_features_classwise_{dataset}.db?mode=ro', mocked.call_args.args[0])
                checks = json.loads(output.read_text())['automated_evidence_checks']
                self.assertEqual(len(checks), 12)
                self.assertTrue(all(checks.values()))

    def test_dataset_mismatch_is_rejected_without_overwriting_report(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'report.json'
            original = json.dumps({'database': 'M', 'cases': []})
            output.write_text(original, encoding='utf-8')
            with self.assertRaises(SystemExit) as stopped:
                self.run_script('--check-only', '--database', 'S', '--output', str(output))
            self.assertEqual(stopped.exception.code, 2)
            self.assertEqual(output.read_text(), original)

    def test_model_mismatch_is_rejected_without_overwriting_report(self):
        for response_mismatch in (False, True):
            with self.subTest(response_mismatch=response_mismatch), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / 'report.json'
                report = {'model': 'qwen2.5-coder:7b', 'cases': [{'response': {'model': 'llama3.2:3b'}} for _ in range(4)]}
                original = json.dumps(report)
                output.write_text(original, encoding='utf-8')
                options = [] if response_mismatch else ['--model', 'llama3.2:3b']
                with self.assertRaises(SystemExit) as stopped:
                    self.run_script('--check-only', '--output', str(output), *options)
                self.assertEqual(stopped.exception.code, 2)
                self.assertEqual(output.read_text(), original)

    def test_default_output_names_separate_models_and_datasets(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.chdir(folder):
            with patch('httpx.Client') as factory:
                client = factory.return_value.__enter__.return_value
                client.post.return_value.status_code = 200
                client.post.return_value.json.return_value = {'trace': []}
                self.run_script()
                for dataset in ('S', 'M'):
                    self.run_script('--database', dataset, '--model', 'qwen2.5-coder:7b')
            self.assertTrue(Path('docs/knowledge-live-results-S-llama3.2-3b.json').exists())
            for dataset in ('S', 'M'):
                self.assertTrue(Path(f'docs/knowledge-live-results-{dataset}-qwen2.5-coder-7b.json').exists())
