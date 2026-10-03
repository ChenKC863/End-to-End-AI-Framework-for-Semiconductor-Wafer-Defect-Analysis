"""Run: python -m unittest discover -s tests -p test_react_api.py -v"""
import sqlite3
import json
from contextlib import closing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from wafer_llm_query import react_api as api


class QueryAPITests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        path = Path(self.folder.name) / "test.db"
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE wafers(id INTEGER, image_path TEXT, split TEXT, true_label TEXT, pred_label TEXT, pred_prob REAL, anomaly_score REAL, is_anomaly INTEGER)")
            connection.executemany("INSERT INTO wafers VALUES(?, ?, 'test', 'Donut', 'Donut', 0.9, ?, ?)", [(i, f"image_{i}.jpg", i / 1000, int(i < 3)) for i in range(205)])
            connection.commit()
        self.mapping = patch.dict(api.DATABASES, {"S": path, "M": path})
        self.mapping.start()
        self.client = TestClient(api.app)

    def tearDown(self):
        self.client.close()
        self.mapping.stop()
        self.folder.cleanup()

    def test_overview(self):
        response = self.client.get('/api/overview?database=S')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['total'], 205)
        self.assertEqual(response.json()['anomalies'], 3)

    def test_query_roundtrip_and_anomaly_order(self):
        with patch.object(api, 'chat', AsyncMock(side_effect=["```sql\nSELECT image_path, anomaly_score FROM wafers ORDER BY anomaly_score ASC LIMIT 5;\n```", "依據前五筆結果整理。"])):
            response = self.client.post('/api/query', json={'question': '最異常的五筆', 'database': 'S'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()['rows']), 5)
        self.assertEqual(response.json()['rows'][0][1], 0)

    def test_readonly_and_single_statement(self):
        for sql in ['DELETE FROM wafers', 'SELECT * FROM wafers; DELETE FROM wafers;', "SELECT load_extension('x')", 'SELECT * FROM sqlite_master', 'SELECT 1 UNION SELECT name FROM sqlite_master']:
            with self.subTest(sql=sql), self.assertRaises(api.HTTPException):
                api.read_query('S', sql)
        self.assertEqual(api.overview('S')['total'], 205)

    def test_result_limit(self):
        result = api.read_query('S', 'SELECT * FROM wafers')
        self.assertEqual(len(result['rows']), 200)
        self.assertTrue(result['truncated'])

    def test_english_summary_language_and_data(self):
        mock = AsyncMock(side_effect=['SELECT COUNT(*) AS record_count FROM wafers', 'The count is 205. See the table below for full fields and individual records.'])
        with patch.object(api, 'chat', mock):
            response = self.client.post('/api/query', json={'question': 'How many records?', 'language': 'en'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['language'], 'en')
        self.assertEqual(response.json()['rows'], [[205]])
        self.assertIn('English', mock.call_args_list[1].args[0][0]['content'])
        self.assertEqual(mock.call_args_list[1].kwargs['num_predict'], 180)

    def test_english_empty_and_summary_failure(self):
        with patch.object(api, 'chat', AsyncMock(return_value='SELECT * FROM wafers WHERE id = -1')):
            result = self.client.post('/api/query', json={'question': 'Find missing records', 'language': 'en'}).json()
        self.assertEqual(result['answer'], 'Query completed. No matching records were found.')
        with patch.object(api, 'chat', AsyncMock(side_effect=['SELECT COUNT(*) FROM wafers', api.HTTPException(503, '離線')])):
            result = self.client.post('/api/query', json={'question': 'Count records', 'language': 'en'}).json()
        self.assertIn('Retrieved', result['answer'])
        self.assertIn('unavailable', result['warning'])

    def test_english_errors_and_invalid_language(self):
        result = self.client.post('/api/query', json={'question': ' ', 'language': 'en'})
        self.assertEqual(result.json()['detail'], 'Please enter a question.')
        with patch.object(api, 'chat', AsyncMock(side_effect=api.HTTPException(503, 'Ollama 無法回應'))):
            result = self.client.post('/api/query', json={'question': 'Count', 'language': 'en'})
        self.assertEqual(result.status_code, 503)
        self.assertIn('Ollama could not respond', result.json()['detail'])
        with patch.object(api, 'DATABASES', {'S': Path(self.folder.name) / 'missing.db'}):
            result = self.client.get('/api/overview?database=S&language=en')
        self.assertEqual(result.json()['detail'], 'The selected wafer database was not found.')
        self.assertEqual(self.client.post('/api/query', json={'question': 'Count', 'language': 'fr'}).status_code, 422)

    def test_selected_model_used_for_both_stages(self):
        for model in api.ALLOWED_MODELS:
            with self.subTest(model=model):
                mock = AsyncMock(side_effect=['SELECT COUNT(*) FROM wafers', '共205筆'])
                with patch.object(api, 'chat', mock):
                    response = self.client.post('/api/query', json={'question': '總數', 'model': model})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['model'], model)
                self.assertEqual([call.kwargs['model'] for call in mock.call_args_list], [model, model])

    def test_invalid_model_rejected_before_inference(self):
        with patch.object(api, 'chat', AsyncMock()) as mock:
            response = self.client.post('/api/query', json={'question': '總數', 'model': 'unapproved:model'})
        self.assertEqual(response.status_code, 422)
        mock.assert_not_called()

    def test_omitted_model_uses_backend_default(self):
        with patch.object(api, 'OLLAMA_MODEL', 'llama3.2:3b'), patch.object(api, 'chat', AsyncMock(side_effect=['SELECT COUNT(*) FROM wafers', '共205筆'])) as mock:
            response = self.client.post('/api/query', json={'question': '總數'})
        self.assertEqual(response.json()['model'], 'llama3.2:3b')
        self.assertEqual(mock.call_args_list[0].kwargs['model'], 'llama3.2:3b')

    def test_compact_summary_preserves_full_table(self):
        mock = AsyncMock(side_effect=['SELECT image_path, anomaly_score FROM wafers ORDER BY anomaly_score ASC LIMIT 25', '摘要'])
        with patch.object(api, 'chat', mock):
            result = self.client.post('/api/query', json={'question': '分數最低的資料'}).json()
        self.assertEqual(len(result['rows']), 25)
        self.assertEqual(result['rows'][0][0], 'image_0.jpg')
        call = mock.call_args_list[1]
        evidence = json.loads(call.args[0][1]['content'])
        self.assertNotIn('image_0.jpg', call.args[0][1]['content'])
        self.assertEqual(len(evidence['rows']), 5)
        self.assertEqual(evidence['returned_rows'], 25)
        self.assertEqual(evidence['returned_rows_score_range'], {'min': 0, 'max': 0.024})
        self.assertEqual(call.kwargs['num_predict'], 180)

    def test_empty_and_invalid_database(self):
        self.assertEqual(self.client.post('/api/query', json={'question': ' '}).status_code, 422)
        self.assertEqual(self.client.get('/api/overview?database=../../x').status_code, 422)
        with patch.object(api, 'chat', AsyncMock(return_value='SELECT * FROM wafers WHERE id = -1')):
            response = self.client.post('/api/query', json={'question': '不存在的影像'})
        self.assertEqual(response.json()['rows'], [])

    def test_llm_failure_and_summary_fallback(self):
        with patch.object(api, 'chat', AsyncMock(side_effect=api.HTTPException(503, '離線'))):
            self.assertEqual(self.client.post('/api/query', json={'question': '查詢'}).status_code, 503)
        with patch.object(api, 'chat', AsyncMock(side_effect=['SELECT COUNT(*) FROM wafers', api.HTTPException(503, '離線')])):
            result = self.client.post('/api/query', json={'question': '總數'}).json()
        self.assertEqual(result['rows'], [[205]])
        self.assertIsNotNone(result['warning'])


if __name__ == '__main__':
    unittest.main()
