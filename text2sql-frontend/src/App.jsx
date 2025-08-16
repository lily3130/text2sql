import { useState } from 'react';
import QueryForm from './components/QueryForm';
import ResultPanel from './components/ResultPanel';
import SqlPanel from './components/SqlPanel';
import UploadPanel from './components/UploadPanel';

const API_BASE = import.meta.env.VITE_API_BASE_URL;
const isNumericCol = (col, rows) =>
  rows.every(r => {
    const v = r?.[col];
    return v == null || v === '' || typeof v === 'number' || !Number.isNaN(Number(v));
  });

const pickDimAndMeasure = (cols, rows) => {
  const L = s => (s || '').toLowerCase();
  const aggRe = /(sum|total|avg|mean|count|min|max|value|amount|consumption|ktoe|kwh|gwh|mwh)$/i;
  const dimRe = /(year|month|date|quarter|category|type|sector|region|fuel|name|country|city|area)$/i;

  const numeric = cols.filter(c => isNumericCol(c, rows));
  const aggCandidates = cols.filter(c => aggRe.test(L(c)));
  const dimCandidates = cols.filter(c => !numeric.includes(c) || dimRe.test(L(c)));

  const xKey = dimCandidates[0] || cols.find(c => !numeric.includes(c)) || cols[0];
  const yKey =
    aggCandidates.find(c => c !== xKey) ||
    numeric.find(c => c !== xKey) ||
    numeric[0] ||
    cols.find(c => c !== xKey) ||
    xKey;

  return { xKey, yKey };
};

const fmtNum = n =>
  Number.isFinite(Number(n)) ? Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—';

export default function App() {
  const [query, setQuery] = useState('');
  const [useEnrichment, setUseEnrichment] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const [columns, setColumns] = useState([]);
  const [rows, setRows] = useState([]);
  const [sql, setSql] = useState('');
  const [summary, setSummary] = useState('');

  const onSubmit = async () => {
    setErr('');
    setSql('');
    setColumns([]);
    setRows([]);
    setSummary('');
    if (!query.trim()) return;

    setLoading(true);
    try {
      const payload = {
        query: query.trim(),
        use_enrichment: !!useEnrichment,
      };

      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(payload),
      });

      const text = await res.text();
      let data; try { data = JSON.parse(text); } catch {}
      if (!res.ok || data?.ok === false) {
        const msg = Array.isArray(data?.detail)
          ? data.detail.map(d => `${d.msg}${d.loc ? ` @ ${d.loc.join('.')}` : ''}`).join('; ')
          : (data?.detail || data?.error || `HTTP ${res.status} ${text.slice(0,200)}`);
        throw new Error(msg);
      }

      setSql(data.sql || '');
      const cols = Array.isArray(data.columns) ? data.columns : [];
      const rs   = Array.isArray(data.rows) ? data.rows : [];
      setColumns(cols);
      setRows(rs);

      if (rs.length && cols.length >= 2) {
        const { xKey, yKey } = pickDimAndMeasure(cols, rs);
        const top = rs.slice()
          .sort((a, b) => (Number(b?.[yKey]) || 0) - (Number(a?.[yKey]) || 0))
          .slice(0, Math.min(3, rs.length));
        setSummary(
          `Top ${top.length} by ${xKey}: ` +
          top.map(r => `${r?.[xKey]} (${fmtNum(r?.[yKey])})`).join(', ') + '.'
        );
      } else if (!rs.length) {
        setSummary('No rows returned.');
      }

    } catch (e) {
      setErr(e.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header>
        <h1>Text-to-SQL</h1>
        <p className="muted">I'm your text-to-SQL agent. Ask me a question!</p>
      </header>

      <QueryForm
        query={query}
        setQuery={setQuery}
        useEnrichment={useEnrichment}
        setUseEnrichment={setUseEnrichment}
        loading={loading}
        error={err}
        onSubmit={onSubmit}
      />

      <UploadPanel apiBase={API_BASE} />

      {(rows.length || sql) && (
        <section className="grid">
          <ResultPanel columns={columns} rows={rows} summary={summary} />
          <SqlPanel sql={sql} />
        </section>
      )}

      <footer>
        <span className="muted">API: <code>{API_BASE}</code></span>
      </footer>
    </div>
  );
}
