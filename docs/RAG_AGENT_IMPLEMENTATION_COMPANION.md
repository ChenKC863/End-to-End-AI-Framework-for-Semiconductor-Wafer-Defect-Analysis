# RAG and Agent implementation companion

Repository documentation supporting Slides 46G–46L. The presentation contains the main findings; this file retains the implementation details and reproduction steps. Prepared locally for GitHub; publication is a separate action.

## Evidence scope

The recorded unit-test run passed 36 tests. The supplied live run used dataset S and llama3.2:3b: English/Chinese RAG and two English Agent cases. All 12 automated checks passed; both RAG answers used source-excerpt fallback. Counts and the complete ordered Donut top-five rows matched SQLite. These are recorded results, not a new benchmark run.

Browser demonstrations use two generation models and different prompts. Keep their screenshots and outcomes separate from the live script. The current Qwen Chinese screenshot has one citation marker per sentence; the English Llama Agent screenshot still illustrates a source-attribution limitation. Passing excerpt checks does not establish semantic correctness.

## Repository placement

| Material | Repository destination |
|---|---|
| Architecture, one representative screenshot, measured outcomes and limitations | Root README, linked to this companion |
| Implementation walkthroughs and exact code excerpts | This file and the source files |
| Installation, startup and index maintenance | Reproduction section below; existing RAG_AGENT_GUIDE.md |
| Script prompts and SQLite checks | scripts/verify_knowledge_live.py |
| Recorded responses and check results | docs/knowledge-live-validation.json |
| Browser screenshots | Add images with model, language, prompt and run/version captions |

The existing KNOWLEDGE_VALIDATION.md describes an earlier 28-test run on port 8002. Retain it as historical evidence; do not cite it as the latest 36-test run on port 8001. Before publishing the latest results, record the tested commit and link the matching JSON and terminal log. Do not invent a commit identifier for uncommitted work.

Full code is maintained in the source files. The excerpts below are a dated snapshot for reviewers. Use commit-specific GitHub source links after committing; line numbers can change.

## Optional walkthrough 46G1 — Code: Assistant Requests and Endpoints

**Purpose:** Show how the selected mode, dataset, model, and language reach the backend.

**Display these source selections**

- `frontend/src/AssistantPanel.jsx`, lines 15–22: request payload and response.
- `wafer_llm_query/assistant_api.py`, lines 13–19: supported request settings.
- `wafer_llm_query/assistant_api.py`, lines 268–279: route registration.

**Speaker notes**

The browser sends its selected model explicitly. The supported generation models are llama3.2:3b and qwen2.5-coder:7b. Language is a request setting; a separate terminal script does not inherit the browser selection.

## Optional walkthrough 46H1 — Code: Vector Search

**Display:** `wafer_llm_query/knowledge_store.py`, lines 146–149 and 154–157.

**Explain**

- `limit` controls the number of retrieved chunks.
- `score_threshold` filters weak matches.
- Returned metadata connects each answer citation to the source text and line range.
- No matching documents means no RAG answer-generation call.

**Supporting reference:** `wafer_llm_query/assistant_api.py`, lines 169–175.

## Optional walkthrough 46I1 — Code: Validate the Plan and Report Counts

**Display these source selections**

- `wafer_llm_query/assistant_api.py`, lines 97–105: reject invalid plans.
- `wafer_llm_query/assistant_api.py`, lines 111–120: report a whole-table COUNT result.

**Speaker notes**

The whole-table wording applies only to the recognized unfiltered COUNT pattern. Filtered queries must not be described as the total number of records. The tool record can include rejected SQL candidates and failure details; it is an execution record, not a chain-of-thought transcript.

## Optional walkthrough 46J1 — Code: Render Citations Once

**Display:** `wafer_llm_query/assistant_api.py`, lines 209–218 and 229–235.

**Speaker notes**

The prompt requests plain answer text and separate source IDs. For compatibility, the backend also removes declared inline markers before adding the canonical markers. An invented or undeclared inline reference is rejected rather than silently removed. Existing source-ID and excerpt checks remain active.

**Test reference:** `tests/test_assistant_api.py`, lines 46–60.

## Optional walkthrough 46K1 — Code: Compare Results with SQLite

**Display:** `scripts/verify_knowledge_live.py`, lines 24–38.

**Baseline query**

```sql
SELECT image_path, anomaly_score
FROM wafers
WHERE LOWER(true_label) = 'donut'
  AND LOWER(pred_label) = 'donut'
ORDER BY anomaly_score ASC
LIMIT 5;
```

**Recorded checks**

```text
count_matches_sqlite: true
count_answer_contains_value: true
donut_rows_match_sqlite: true
mixed_question_uses_both_tools: true
```

**Speaker notes**

The comparison checks the complete five returned rows and their order, not only the score range. The recorded range rounded to six decimals is -0.059801 to -0.015675. LIMIT 5 selects five rows; `truncated: false` means the backend did not additionally truncate the result. It does not mean only five records match the filters. Quote and citation checks do not establish that a generated SQL explanation accurately describes the executed query.

## Reproduction Commands

**Terminal D — first-time index setup**

```powershell
cd 'E:\聯成電腦\Semiconductor_wafer'
.\.venv-react\Scripts\python.exe -m pip install -r requirements-rag.in
ollama pull embeddinggemma
$env:OLLAMA_NUM_GPU = '0'
.\.venv-react\Scripts\python.exe -m wafer_llm_query.build_knowledge
```

**Terminal B — start the API**

```powershell
cd 'E:\聯成電腦\Semiconductor_wafer'
$env:OLLAMA_NUM_GPU = '0'
.\.venv-react\Scripts\python.exe -m uvicorn wafer_llm_query.react_api:app --host 127.0.0.1 --port 8001
```

**Terminal D — validation, one command at a time**

```powershell
cd 'E:\聯成電腦\Semiconductor_wafer'
.\.venv-react\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-react\Scripts\python.exe scripts/verify_knowledge_live.py --url http://127.0.0.1:8001 --output docs/knowledge-live-validation.json
.\.venv-react\Scripts\python.exe scripts/verify_knowledge_live.py --check-only --output docs/knowledge-live-validation.json
```

**Speaker notes**

Keep this command slide as backup material or in GitHub. Use a new output filename for a new recorded run and pass the same filename to check-only; the shown filename is overwritten if reused. Ollama and FastAPI must be running for live validation. Startup does not enable automatic Python reload. Rebuild React after frontend changes; restart FastAPI after backend changes. Terminal C is needed only for Vite development mode.

**Implementation references:** `wafer_llm_query/build_knowledge.py`, lines 1–7; `scripts/verify_knowledge_live.py`, lines 9–54.

## Exact code excerpts

These excerpts preserve the inspected source code. The line ranges identify the original files, not this draft.

### Optional walkthrough 46G1 — frontend/src/AssistantPanel.jsx, lines 15–22

```jsx
      const response = await fetch(`/api/${mode}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, database, model, language }),
        signal: AbortSignal.timeout(300000),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
      setResult(data);
```

### Optional walkthrough 46G1 — wafer_llm_query/assistant_api.py, lines 13–19

```python
class AssistantRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1500)
    database: Literal["S", "M"] = "S"
    model: Literal["llama3.2:3b", "qwen2.5-coder:7b"] = "llama3.2:3b"
    language: Literal["zh", "en"] = "zh"
    top_k: int = Field(default=3, ge=1, le=5)
    min_score: float = Field(default=0.4, ge=0, le=1)
```

### Optional walkthrough 46G1 — wafer_llm_query/assistant_api.py, lines 268–279

```python
def register(app, api):
    @app.get("/api/knowledge/status", tags=["Knowledge assistant"])
    def knowledge_status():
        return knowledge_store.status()

    @app.post("/api/rag", tags=["Knowledge assistant"])
    async def rag(body: AssistantRequest):
        return await run(body, api)

    @app.post("/api/agent", tags=["Knowledge assistant"])
    async def agent(body: AssistantRequest):
        return await run(body, api, use_agent=True)
```

### Optional walkthrough 46H1 — wafer_llm_query/knowledge_store.py, lines 146–149

```python
            points = client.query_points(manifest["collection"], query=vector,
                                         limit=top_k, score_threshold=min_score).points
            return [{"id": f"K{i}", "score": round(p.score, 6), **p.payload}
                    for i, p in enumerate(points, 1)]
```

### Optional walkthrough 46H1 — wafer_llm_query/knowledge_store.py, lines 154–157

```python
async def retrieve(question, top_k=3, min_score=0.4):
    manifest = await asyncio.to_thread(read_manifest)
    vector = (await embed([question]))[0]
    return await asyncio.to_thread(search_vector, vector, manifest, top_k, min_score)
```

### Optional walkthrough 46I1 — wafer_llm_query/assistant_api.py, lines 97–105

```python
def checked_plan(raw):
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        plan = Plan.model_validate_json(raw)
        if len(set(plan.tools)) != len(plan.tools):
            raise ValueError("Duplicate tool")
        return plan.tools
    except (ValidationError, ValueError) as exc:
        raise HTTPException(502, "Agent returned an invalid tool plan. No tools were executed.") from exc
```

### Optional walkthrough 46I1 — wafer_llm_query/assistant_api.py, lines 111–120

```python
    # Only an unfiltered, ungrouped COUNT(*) can describe the whole table.
    total_query = re.fullmatch(
        r'\s*SELECT\s+COUNT\s*\(\s*\*\s*\)(?:\s+AS\s+\w+)?\s+FROM\s+wafers\s*;?\s*',
        result.get("sql", ""), re.I)
    if total_query and count == 1 and len(result["rows"][0]) == 1:
        total = result["rows"][0][0]
        if isinstance(total, int) and not isinstance(total, bool):
            return text(body,
                        f"The wafers table in dataset {body.database} contains {total:,} records. [SQL1]",
                        f"資料集 {body.database} 的 wafers 資料表共有 {total:,} 筆記錄。[SQL1]")
```

### Optional walkthrough 46J1 — wafer_llm_query/assistant_api.py, lines 209–218

```python
                for sentence in parsed.sentences:
                    # Never silently hide an invented or undeclared inline citation.
                    inline_ids = re.findall(r'\[(K\d+|SQL\d+)\]', sentence.text)
                    if any(c not in sentence.source_ids or c not in valid_ids for c in inline_ids):
                        failures.append(text(body, "The response contains an undeclared or invalid inline citation.",
                                             "回答文字含未宣告或無效的引用代碼，因此未採用。"))
                    sentence.source_ids = list(dict.fromkeys(sentence.source_ids))
                    for c in sentence.source_ids:
                        sentence.text = sentence.text.replace(f'[{c}]', '')
                    sentence.text = re.sub(r'\s+([，。！？,.;:!?])', r'\1', sentence.text).strip()
```

### Optional walkthrough 46J1 — wafer_llm_query/assistant_api.py, lines 229–235

```python
            if failures:
                trace.append({"tool": "generate_explanation", "status": "rejected", "candidate": candidate, "reasons": failures})
                warnings.extend(failures)
            else:
                answer = "\n".join(s.text + " " + " ".join(f"[{c}]" for c in s.source_ids) for s in parsed.sentences)
                citations = sorted(set(cited))
                generation_status = "generated"
```

### Optional walkthrough 46K1 — scripts/verify_knowledge_live.py, lines 24–38

```python
    database = Path(__file__).resolve().parents[1] / 'wafer_features_classwise_S.db'
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as conn:
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
```

## Source snapshot

Line references were checked on September 27, 2026. SHA-256 values identify the source versions used for this draft.

| File | SHA-256 |
|---|---|
| frontend/src/AssistantPanel.jsx | `c79bd9ad2320576f52f1bb6277a68b7bf8f4aad06bf05d69a2609da0fad07066` |
| frontend/src/main.jsx | `94f34f5775de3293fa527b8f9a0f7b1b71a3b1288f6b43a46c9d2f81715d0363` |
| scripts/verify_knowledge_live.py | `48a930b75b005f4b233dc1b8410cb4a2b340a9e08be4573f16bc030e4d8c4eb9` |
| tests/test_assistant_api.py | `cb34af57da6ff95d9c53486c3c87298a24dbf52d0536cac7a4895777ebc3dbe5` |
| wafer_llm_query/assistant_api.py | `a3bfe66bdeadc3a2273909fd117116005f555d521f965eaefa01aab0eea34703` |
| wafer_llm_query/build_knowledge.py | `677ad48b6c4bf756c63e4d9aeaac808f16a1069026d7f02bdfb8eaa14476041d` |
| wafer_llm_query/knowledge_store.py | `6625e39e2bebc69f77f765e29f7b2e7a168ba3b93432d99b576725fdbd080276` |
| wafer_llm_query/react_api.py | `cb1e1261ef18cfb53afabe1184b2b5a0fe9ec3df55372f2990efddf6d66f00e4` |
