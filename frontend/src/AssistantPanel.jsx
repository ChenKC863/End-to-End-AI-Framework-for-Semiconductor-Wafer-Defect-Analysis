import React, { useState } from 'react';

export default function AssistantPanel({ database, model, language }) {
  const en = language === 'en';
  const [mode, setMode] = useState('rag');
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function submit(event) {
    event.preventDefault();
    if (busy || !question.trim()) return;
    setBusy(true); setError(''); setResult(null);
    try {
      const response = await fetch(`/api/${mode}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, database, model, language }),
        signal: AbortSignal.timeout(300000),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  return <section className="panel">
    <h2>{en ? 'Knowledge assistant' : '知識助理'}</h2>
    <p>{en ? 'RAG searches project documents. Agent chooses document retrieval, read-only SQL, or both.' : 'RAG 檢索專案文件；Agent 選擇文件檢索、唯讀 SQL 或兩者。'}</p>
    <form onSubmit={submit}>
      <label htmlFor="assistant-mode">{en ? 'Mode' : '模式'}</label>
      <select id="assistant-mode" value={mode} disabled={busy} onChange={e => { setMode(e.target.value); setResult(null); }}>
        <option value="rag">RAG</option><option value="agent">Agent</option>
      </select>
      <label htmlFor="assistant-question">{en ? 'Question' : '問題'}</label>
      <textarea id="assistant-question" value={question} maxLength={1500} required disabled={busy}
        placeholder={en ? 'What does a negative anomaly score mean?' : '負的異常分數代表什麼？'}
        onChange={e => setQuestion(e.target.value)} />
      <button disabled={busy || !question.trim()}>{busy ? (en ? 'Working…' : '處理中…') : (en ? 'Ask assistant' : '詢問助理')}</button>
    </form>
    {error && <p role="alert" className="error">{error}</p>}
    {result && <div aria-live="polite">
      <p>{result.mode} · {result.model} · {result.language}</p>
      <p style={{ whiteSpace: 'pre-wrap' }}>{result.answer}</p>
      {result.warning && <p role="alert" className="error">{result.warning}</p>}
      <h3>{en ? 'Sources and query evidence' : '來源與查詢證據'}</h3>
      {result.sources.map(s => <details key={s.id}><summary>[{s.id}] {s.source}:{s.start_line}–{s.end_line} ({s.score})</summary><pre style={{ whiteSpace: 'pre-wrap' }}>{s.text}</pre></details>)}
      {result.sql_result && <details open><summary>[SQL1] {en ? 'SQL evidence' : 'SQL 證據'} ({result.sql_result.database})</summary>
        <pre>{result.sql_result.sql}</pre>
        <p>{en ? `SQL returned ${result.sql_result.rows.length} row(s).` : `SQL 回傳 ${result.sql_result.rows.length} 列結果。`}</p>
        <div style={{ overflowX: 'auto' }}><table>
          <thead><tr>{result.sql_result.columns.map((column, i) => <th key={i} scope="col">{column}</th>)}</tr></thead>
          <tbody>{result.sql_result.rows.map((row, i) => <tr key={i}>{row.map((value, j) =>
            <td key={j} style={{ overflowWrap: 'anywhere' }}>{value === null ? 'NULL' : String(value)}</td>)}</tr>)}</tbody>
        </table></div>
        <details><summary>{en ? 'Raw JSON' : '原始 JSON'}</summary><pre>{JSON.stringify({ columns: result.sql_result.columns, rows: result.sql_result.rows }, null, 2)}</pre></details>
        <p>truncated: {String(result.sql_result.truncated)}</p></details>}
      <details><summary>{en ? 'Tool execution record' : '工具執行紀錄'}</summary><pre>{JSON.stringify(result.trace, null, 2)}</pre></details>
      <p className="footnote">{en ? 'Citation IDs are checked. Review whether the sources support each claim.' : '系統檢查引用代碼，仍須人工確認來源是否支持回答內容。'}</p>
    </div>}
  </section>;
}
