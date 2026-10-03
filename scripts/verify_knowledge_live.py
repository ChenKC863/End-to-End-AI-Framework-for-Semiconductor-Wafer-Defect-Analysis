"""Explicit live validation against a running local API. Saves actual responses."""
import argparse
import datetime
import json
import sqlite3
from contextlib import closing
from pathlib import Path
import httpx

MODELS = ['llama3.2:3b', 'qwen2.5-coder:7b']
parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8001')
parser.add_argument('--output', help='Report path; defaults to a dataset-and-model-specific filename')
parser.add_argument('--database', choices=['S', 'M'], help='Live dataset (default: S); check-only uses the saved report dataset')
parser.add_argument('--model', choices=MODELS, help='Live model (default: llama3.2:3b); check-only uses the saved report model')
parser.add_argument('--check-only', action='store_true')
args = parser.parse_args()
selected_database = args.database or 'S'
selected_model = args.model or 'llama3.2:3b'
model_filename = selected_model.replace(':', '-')
args.output = args.output or f'docs/knowledge-live-results-{selected_database}-{model_filename}.json'
cases = [
    ('rag', 'What does a negative anomaly score mean? Can scores be compared across S and M?', 'en'),
    ('rag', '負的異常分數代表什麼？可以直接比較 S 和 M 的異常分數嗎？', 'zh'),
    ('agent', 'How many records are in the wafer database?', 'en'),
    ('agent', 'List the five lowest anomaly_score values where true_label and pred_label are both Donut, showing image_path and anomaly_score. Explain what a negative anomaly score means.', 'en'),
]
report = {'executed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'base_url': args.url, 'model': selected_model, 'database': selected_database, 'cases': []}
if args.check_only:
    report = json.loads(Path(args.output).read_text(encoding='utf-8'))
    # Historical reports were produced by the S-only script.
    selected_database = report.get('database', 'S')
    if selected_database not in {'S', 'M'}:
        parser.error('The saved report has an invalid database; expected S or M.')
    if args.database and args.database != selected_database:
        parser.error('--database does not match the saved report dataset.')
    selected_model = report.get('model', 'llama3.2:3b')
    if selected_model not in MODELS:
        parser.error('The saved report has an unsupported model.')
    if args.model and args.model != selected_model:
        parser.error('--model does not match the saved report model.')
    if len(report['cases']) != 4:
        parser.error('Expected a complete report containing all four cases.')
    for case in report['cases']:
        response_model = case['response'].get('model')
        if response_model is not None and response_model != selected_model:
            parser.error('A response model does not match the saved report model.')
        sql_result = case['response'].get('sql_result') or {}
        if sql_result and sql_result.get('database') != selected_database:
            parser.error('A SQL result dataset does not match the saved report dataset.')
    database = Path(__file__).resolve().parents[1] / f'wafer_features_classwise_{selected_database}.db'
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as conn:
        expected_count = conn.execute('SELECT COUNT(*) FROM wafers').fetchone()[0]
        expected_rows = conn.execute("SELECT image_path, anomaly_score FROM wafers WHERE LOWER(true_label) = 'donut' AND LOWER(pred_label) = 'donut' ORDER BY anomaly_score ASC LIMIT 5").fetchall()
    checks = {}
    for i, case in enumerate(report['cases']):
        data = case['response']
        checks[f'case_{i + 1}_http_and_citations'] = (case['status'] == 200 and bool(data.get('citations')))
        checks[f'case_{i + 1}_explicit_answer_status'] = data.get('generation_status') in {'generated', 'extractive_fallback', 'deterministic_sql'}
    count_result = report['cases'][2]['response'].get('sql_result') or {}
    combined = report['cases'][3]['response']
    checks['count_matches_sqlite'] = count_result.get('rows') == [[expected_count]]
    checks['count_answer_contains_value'] = str(expected_count) in report['cases'][2]['response'].get('answer', '').replace(',', '')
    checks['donut_rows_match_sqlite'] = (combined.get('sql_result') or {}).get('rows') == [list(row) for row in expected_rows]
    checks['mixed_question_uses_both_tools'] = {t['tool'] for t in combined.get('trace', []) if t['status'] == 'ok'} == {'search_knowledge', 'query_wafer_sql'}
    report['automated_evidence_checks'] = checks
    report['semantic_review_required'] = True
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)
with httpx.Client(timeout=300) as client:
    for mode, question, language in cases:
        response = client.post(args.url + '/api/' + mode, json={
            'question': question, 'language': language, 'database': selected_database, 'model': selected_model})
        result = response.json()
        report['cases'].append({'mode': mode, 'question': question, 'language': language,
                                'status': response.status_code, 'response': result})
        print(selected_database, selected_model, mode, language, response.status_code, result.get('trace'), flush=True)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
