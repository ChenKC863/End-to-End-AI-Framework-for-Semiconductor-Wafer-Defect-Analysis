import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';
import { translations } from './translations';
import AssistantPanel from './AssistantPanel';

async function request(path, options = {}, language = 'zh') {
  const t = translations[language];
  const response = await fetch(path, { ...options, signal: options.signal ?? AbortSignal.timeout(180000) });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `${t.failure} (${response.status})`);
  return data;
}

function ResultTable({ columns, rows, t }) {
  if (!rows.length) return <p className="empty">{t.empty}</p>;
  return <div className="table-scroll" tabIndex="0" aria-label={t.table}><table><thead><tr>{columns.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((value, j) => <td key={j}>{value == null ? '—' : typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(6) : String(value)}</td>)}</tr>)}</tbody></table></div>;
}

function App() {
  const [language, setLanguage] = useState(() => {
    try { return localStorage.getItem('wafer-language') === 'en' ? 'en' : 'zh'; }
    catch { return 'zh'; }
  });
  const t = translations[language];
  useEffect(() => {
    document.documentElement.lang = language === 'en' ? 'en' : 'zh-Hant';
    document.title = t.title;
    try { localStorage.setItem('wafer-language', language); } catch {}
  }, [language, t.title]);
  function changeLanguage(next) {
    const index = t.examples.indexOf(question);
    if (index >= 0) setQuestion(translations[next].examples[index]);
    setLanguage(next); setResult(null); setError('');
  }
  const [database, setDatabase] = useState('S');
  const [model, setModel] = useState('qwen2.5-coder:7b');
  const [question, setQuestion] = useState(() => translations[language].examples[1]);
  const [stats, setStats] = useState(null);
  const [statsError, setStatsError] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [reload, setReload] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setStats(null); setStatsError(''); setResult(null); setError('');
    request(`/api/overview?database=${database}&language=${language}`, { signal: controller.signal }, language)
      .then(setStats).catch(e => { if (!controller.signal.aborted) setStatsError(e instanceof TypeError ? translations[language].network : e.message); });
    return () => controller.abort();
  }, [database, reload, language]);

  async function submit(event) {
    event.preventDefault();
    if (!question.trim() || busy) return;
    setBusy(true); setError(''); setResult(null);
    try {
      setResult(await request('/api/query', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim(), database, model, language }),
      }, language));
    } catch (e) { setError(e.name === 'TimeoutError' ? t.timeout : e instanceof TypeError ? t.network : e.message); }
    finally { setBusy(false); }
  }

  return <div className="workspace">
    <aside><p className="eyebrow">{t.workspace}</p><div className="active-nav">{t.nav}</div><label htmlFor="language">{t.language}</label><select id="language" value={language} disabled={busy} onChange={e => changeLanguage(e.target.value)}><option value="zh">繁體中文</option><option value="en">English</option></select><label htmlFor="database">{t.datasetLabel}</label><select id="database" value={database} disabled={busy} onChange={e => setDatabase(e.target.value)}><option value="S">EfficientNetV2-S</option><option value="M">EfficientNetV2-M</option></select><label htmlFor="model">{t.model}</label><select id="model" value={model} disabled={busy} onChange={e => { setModel(e.target.value); setError(''); }}><option value="qwen2.5-coder:7b">qwen2.5-coder:7b</option><option value="llama3.2:3b">llama3.2:3b</option></select><p className="sidebar-note">{t.sidebar}</p><footer>React · FastAPI · Ollama<br/>{t.footer}</footer></aside>
    <main><header><div><h1>{t.title}</h1><p>{t.subtitle}</p></div><span className="badge">{t.dataset} {database}</span></header>
      {statsError ? <div role="alert" className="error">{t.loadError}{statsError} <button onClick={() => setReload(v => v + 1)}>{t.retry}</button></div> : <section className="metrics" aria-label={t.overview}>{[[t.total, stats?.total], [t.anomalies, stats?.anomalies], [t.classes, stats?.classes?.length]].map(([name, value]) => <div key={name}><span>{name}</span><strong>{value == null ? '…' : value.toLocaleString()}</strong></div>)}</section>}
      <section className="panel"><h2>{t.ask}</h2><form onSubmit={submit}><label className="sr-only" htmlFor="question">{t.question}</label><textarea id="question" maxLength={1500} value={question} disabled={busy} onChange={e => setQuestion(e.target.value)} rows="3" required/><div className="form-bottom"><span>{t.score}</span><button disabled={busy || !question.trim()} type="submit">{busy ? t.busy : t.submit}</button></div></form><div className="examples">{t.examples.map(q => <button className="example" key={q} disabled={busy} onClick={() => setQuestion(q)}>{q}</button>)}</div></section>
      <div aria-live="polite">{busy && <p className="status">{t.waiting}</p>}{error && <p role="alert" className="error">{error}</p>}</div>
      {result && <section className="panel result"><div className="section-heading"><h2>{t.results}</h2><span>{result.rows.length} {t.rows}{result.truncated ? t.limit : ''}</span></div><p className="footnote">{t.usedModel}<strong>{result.model ?? t.missingModel}</strong></p><p className="answer">{result.answer}</p>{result.warning && <p className="error">{result.warning}</p>}<details open><summary>{t.sql}</summary><pre><code>{result.sql}</code></pre></details><ResultTable columns={result.columns} rows={result.rows} t={t}/><p className="footnote">{t.note}</p></section>}
      {stats?.classes?.length > 0 && <section className="panel"><h2>{t.distribution}</h2><p className="footnote">{t.distributionNote}</p><div className="distribution">{stats.classes.map(c => <div className="bar-row" key={c.label}><span>{c.label ?? t.unknown}</span><div className="track"><div style={{ width: `${100 * c.count / Math.max(...stats.classes.map(x => x.count))}%` }}/></div><strong>{c.count.toLocaleString()}</strong></div>)}</div></section>}
      <AssistantPanel key={`${database}-${model}-${language}`} database={database} model={model} language={language}/>
    </main></div>;
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);
