import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from wafer_llm_query import react_api as api, knowledge_store as store


SOURCE = {"id": "K1", "source": "scores.md", "start_line": 1, "end_line": 2,
          "score": 0.8, "text": "Lower scores are more anomalous within the same model."}


def cited_answer(text='Lower scores mean more abnormality.', ids=None):
    return json.dumps({'sentences': [{'text': text, 'source_ids': ids or ['K1'], 'quote': SOURCE['text']}]})


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def tearDown(self):
        self.client.close()

    def post(self, mode="rag", **kwargs):
        return self.client.post('/api/' + mode, json={"question": "Explain scores", "language": "en", **kwargs})

    def test_rag_citations_and_sources(self):
        with patch.object(store, 'retrieve', AsyncMock(return_value=[SOURCE])), patch.object(api, 'chat', AsyncMock(return_value=cited_answer())):
            data = self.post().json()
        self.assertEqual(data['citations'], ['K1'])
        self.assertEqual(data['sources'][0]['start_line'], 1)
        self.assertEqual(data['trace'][0]['tool'], 'search_knowledge')

    def test_unknown_or_missing_citations_withheld(self):
        for reply in [cited_answer(ids=['K99']), 'An unsupported answer.']:
            with patch.object(store, 'retrieve', AsyncMock(return_value=[SOURCE])), patch.object(api, 'chat', AsyncMock(return_value=reply)):
                data = self.post().json()
            self.assertNotEqual(data['answer'], reply)
            self.assertEqual(data['generation_status'], 'extractive_fallback')
            self.assertIn(SOURCE['text'], data['answer'])

    def test_inline_citations_are_rendered_once_in_both_languages(self):
        for language, sentence in [('zh', '分數越低越異常 [K1]。[K1]'),
                                   ('en', 'Lower scores are more anomalous [K1].')]:
            with self.subTest(language=language), patch.object(store, 'retrieve', AsyncMock(return_value=[SOURCE])), patch.object(api, 'chat', AsyncMock(return_value=cited_answer(text=sentence, ids=['K1', 'K1']))):
                data = self.post(language=language).json()
            self.assertEqual(data['generation_status'], 'generated')
            self.assertEqual(data['answer'].count('[K1]'), 1)
            self.assertEqual(data['citations'], ['K1'])
            self.assertIsNone(data['warning'])

    def test_invalid_inline_citation_is_not_hidden_by_deduplication(self):
        with patch.object(store, 'retrieve', AsyncMock(return_value=[SOURCE])), patch.object(api, 'chat', AsyncMock(return_value=cited_answer(text='Lower scores are anomalous [K99].'))):
            data = self.post().json()
        self.assertEqual(data['generation_status'], 'extractive_fallback')
        self.assertIn('inline citation', data['warning'])

    def test_no_matches_skips_generation(self):
        with patch.object(store, 'retrieve', AsyncMock(return_value=[])), patch.object(api, 'chat', AsyncMock()) as chat:
            data = self.post().json()
        chat.assert_not_called()
        self.assertEqual(data['citations'], [])

    def test_fabricated_quote_and_echo_are_withheld(self):
        bad_quote = json.loads(cited_answer())
        bad_quote['sentences'][0]['quote'] = 'This quotation does not exist in the source.'
        for reply in [json.dumps(bad_quote), cited_answer(text='Explain scores')]:
            with patch.object(store, 'retrieve', AsyncMock(return_value=[SOURCE])), patch.object(api, 'chat', AsyncMock(return_value=reply)):
                data = self.post().json()
            self.assertEqual(data['generation_status'], 'extractive_fallback')

    def test_invalid_plan_never_executes(self):
        for plan in ['{"tools":["shell"]}', '{"tools":["search_knowledge","search_knowledge"]}', 'not json', '{"tools":[],"sql":"DELETE"}']:
            with patch.object(api, 'chat', AsyncMock(return_value=plan)), patch.object(store, 'retrieve', AsyncMock()) as retrieve, patch.object(api, 'read_query') as sql:
                self.assertEqual(self.post('agent').status_code, 502)
                retrieve.assert_not_called()
                sql.assert_not_called()

    def test_agent_both_tools_preserves_table_and_model(self):
        replies = ['{"tools":["search_knowledge","query_wafer_sql"]}',
                   'SELECT COUNT(*) FROM wafers', cited_answer()]
        with patch.object(store, 'retrieve', AsyncMock(return_value=[SOURCE])), patch.object(api, 'chat', AsyncMock(side_effect=replies)) as chat, patch.object(api, 'read_query', return_value={'columns': ['count'], 'rows': [[12]], 'truncated': False}):
            data = self.post('agent', model='llama3.2:3b').json()
        self.assertEqual(data['sql_result']['rows'], [[12]])
        self.assertEqual(len(data['trace']), 2)
        self.assertEqual(data['citations'], ['K1', 'SQL1'])
        self.assertTrue(all(c.kwargs['model'] == 'llama3.2:3b' for c in chat.call_args_list))

    def test_aggregate_value_is_not_confused_with_result_row_count(self):
        with patch.object(api, 'chat', AsyncMock(side_effect=['{"tools":["query_wafer_sql"]}', 'SELECT COUNT(*) FROM wafers'])) as chat, patch.object(api, 'read_query', return_value={'columns': ['COUNT(*)'], 'rows': [[7011]], 'truncated': False}):
            data = self.post('agent').json()
        self.assertIn('7,011 records', data['answer'])
        self.assertEqual(chat.await_count, 2)

    def test_agent_unsafe_sql_uses_existing_guard(self):
        with patch.object(api, 'chat', AsyncMock(side_effect=['{"tools":["query_wafer_sql"]}', 'DELETE FROM wafers', 'DELETE FROM wafers'])):
            data = self.post('agent').json()
        self.assertEqual(data['trace'][-1]['status'], 'error')
        self.assertIsNone(data['sql_result'])

    def test_sql_json_repair_is_bounded_and_keeps_guard(self):
        replies = ['{"tools":["query_wafer_sql"]}',
                   json.dumps({'sql': 'SELECT COUNT(*) FROM wafers; SELECT COUNT(*) FROM wafers'}),
                   json.dumps({'sql': 'SELECT COUNT(*) FROM wafers'})]
        with patch.object(api, 'chat', AsyncMock(side_effect=replies)) as chat, patch.object(api, 'read_query', side_effect=[HTTPException(400, 'multiple statements'), {'columns': ['COUNT(*)'], 'rows': [[7011]], 'truncated': False}]) as read:
            data = self.post('agent').json()
        self.assertEqual(read.call_count, 2)
        self.assertEqual(chat.await_count, 3)
        self.assertEqual(data['sql_result']['rows'], [[7011]])
        self.assertEqual(data['trace'][0]['status'], 'rejected')
        self.assertIsNone(data['warning'])

    def test_comparison_omission_falls_back_to_sources(self):
        with patch.object(store, 'retrieve', AsyncMock(return_value=[SOURCE])), patch.object(api, 'chat', AsyncMock(return_value=cited_answer())):
            data = self.post(question='What do negative scores mean? Can S and M scores be compared?').json()
        self.assertEqual(data['generation_status'], 'extractive_fallback')
        self.assertIn('omitted', data['warning'])

    def test_document_question_is_preserved_when_sql_fails(self):
        question = '資料查詢：列出 Donut 五筆。\n文件解釋：負分代表什麼？\n回答時，請引用來源。'
        replies = ['{"tools":["search_knowledge","query_wafer_sql"]}', 'bad', 'bad', cited_answer()]
        with patch.object(store, 'retrieve', AsyncMock(return_value=[SOURCE])) as retrieve, patch.object(api, 'chat', AsyncMock(side_effect=replies)) as chat:
            data = self.post('agent', question=question).json()
        self.assertEqual(retrieve.call_args.args[0], '負分代表什麼？')
        explanation = json.loads(chat.call_args.args[0][1]['content'])
        self.assertEqual(explanation['question'], '負分代表什麼？')
        self.assertIsNone(data['sql_result'])

    def test_count_summary_does_not_call_filtered_count_table_total(self):
        from wafer_llm_query.assistant_api import AssistantRequest, sql_summary
        body = AssistantRequest(question='筆數', language='zh')
        result = {'columns': ['COUNT(*)'], 'rows': [[7011]], 'truncated': False,
                  'sql': 'SELECT COUNT(*) AS total FROM wafers;'}
        self.assertEqual(sql_summary(body, result), '資料集 S 的 wafers 資料表共有 7,011 筆記錄。[SQL1]')
        result['sql'] = "SELECT COUNT(*) FROM wafers WHERE true_label = 'Donut'"
        self.assertNotIn('共有', sql_summary(body, result))

    def test_localized_validation_reasons_and_single_fallback_notice(self):
        bad_quote = json.loads(cited_answer())
        bad_quote['sentences'][0]['quote'] = 'This quote is not in the source.'
        cases = [('bad json', '格式'), (cited_answer(ids=['K99']), '來源代碼'),
                 (json.dumps(bad_quote), '原文摘錄'), (cited_answer(text='Explain scores'), '重複')]
        for reply, reason in cases:
            with self.subTest(reason=reason), patch.object(store, 'retrieve', AsyncMock(return_value=[SOURCE])), patch.object(api, 'chat', AsyncMock(return_value=reply)):
                data = self.post(language='zh').json()
            self.assertIn(reason, data['warning'])
            self.assertEqual(data['generation_status'], 'extractive_fallback')
            self.assertEqual(data['answer'].count('以下顯示檢索來源原文'), 1)
            self.assertNotIn('SQL 查詢結果仍可使用', data['answer'])

    def test_combined_fallback_preserves_sql_and_notice(self):
        replies = ['{"tools":["search_knowledge","query_wafer_sql"]}',
                   'SELECT COUNT(*) FROM wafers', 'bad json']
        with patch.object(store, 'retrieve', AsyncMock(return_value=[SOURCE])), patch.object(api, 'chat', AsyncMock(side_effect=replies)), patch.object(api, 'read_query', return_value={'columns': ['COUNT(*)'], 'rows': [[7011]], 'truncated': False}):
            data = self.post('agent', language='zh').json()
        self.assertIn('7,011', data['answer'])
        self.assertEqual(data['answer'].count('以下顯示檢索來源原文'), 1)
        self.assertIn('SQL 查詢結果仍可使用', data['answer'])
        self.assertEqual(data['citations'], ['K1', 'SQL1'])

    def test_partial_tool_failure_retains_evidence(self):
        with patch.object(store, 'retrieve', AsyncMock(side_effect=HTTPException(503, 'index missing'))), patch.object(api, 'chat', AsyncMock(side_effect=['{"tools":["search_knowledge","query_wafer_sql"]}', 'SELECT COUNT(*) FROM wafers', '12 records [SQL1].'])), patch.object(api, 'read_query', return_value={'columns': ['count'], 'rows': [[12]], 'truncated': False}):
            data = self.post('agent').json()
        self.assertIn('index missing', data['warning'])
        self.assertEqual(data['sql_result']['rows'], [[12]])

    def test_generation_failure_and_zh_abstention(self):
        with patch.object(store, 'retrieve', AsyncMock(return_value=[SOURCE])), patch.object(api, 'chat', AsyncMock(side_effect=HTTPException(503, 'model offline'))):
            data = self.post(language='zh').json()
        self.assertIn('檢索來源原文', data['answer'])
        self.assertEqual(data['sources'], [SOURCE])

    def test_validation(self):
        for kwargs in [{'question': ' '}, {'top_k': 99}, {'min_score': -1}, {'model': 'other'}, {'database': '../x'}]:
            self.assertEqual(self.post(**kwargs).status_code, 422)


class VectorStoreTests(unittest.TestCase):
    def test_persistent_search_threshold_and_stale_corpus(self):
        with tempfile.TemporaryDirectory() as folder:
            chunks = [{'source': 'a.md', 'start_line': 1, 'end_line': 1, 'text': 'alpha'},
                      {'source': 'b.md', 'start_line': 1, 'end_line': 1, 'text': 'beta'}]
            with patch.object(store, 'INDEX_DIR', Path(folder)), patch.object(store, 'documents', return_value=chunks):
                manifest = store.save_index(chunks, [[1., 0.], [0., 1.]])
                self.assertTrue(store.status()['ready'])
                hits = store.search_vector([1., 0.], manifest, 2, 0.9)
                self.assertEqual([h['text'] for h in hits], ['alpha'])
                self.assertEqual(store.search_vector([-1., 0.], manifest, 2, 0.9), [])
                with self.assertRaises(HTTPException):
                    store.search_vector([1., 0., 0.], manifest, 2, 0)
                with patch.object(store, 'documents', return_value=[]):
                    self.assertFalse(store.status()['ready'])

    def test_missing_index(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(store, 'INDEX_DIR', Path(folder)):
            self.assertFalse(store.status()['ready'])

    def test_corrupt_manifest(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(store, 'INDEX_DIR', Path(folder)):
            (Path(folder) / 'manifest.json').write_text('{bad json', encoding='utf-8')
            self.assertFalse(store.status()['ready'])

    def test_chunk_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.md'
            path.write_text('one\n' + 'a' * 1790 + '\nthree', encoding='utf-8')
            chunks = store.documents(Path(folder))
            self.assertEqual(chunks[0]['start_line'], 1)
            self.assertEqual(chunks[-1]['end_line'], 3)
            self.assertEqual('\n'.join(c['text'] for c in chunks), path.read_text())


if __name__ == '__main__':
    unittest.main()
