import { useState } from 'react';
import QueryForm from './components/QueryForm';
import ResultPanel from './components/ResultPanel';
import SqlPanel from './components/SqlPanel';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export default function App() {
  const [query, setQuery] = useState('');
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
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || data?.error || `HTTP ${res.status}`);
      if (!data.ok) throw new Error(data?.error || 'Backend returned ok=false');

      setSql(data.sql || '');
      const cols = Array.isArray(data.columns) ? data.columns : [];
      const rs = Array.isArray(data.rows) ? data.rows : [];
      setColumns(cols);
      setRows(rs);

      if (rs.length && cols.length >= 2) {
        const labelKey = cols[0];
        const valueKey = cols.find(c => typeof rs[0]?.[c] === 'number') || cols[1];
        const top = rs.slice().sort((a,b)=> (Number(b?.[valueKey])||0)-(Number(a?.[valueKey])||0))
                       .slice(0, Math.min(3, rs.length));
        setSummary(
          `Top ${top.length} by ${valueKey}: ` +
          top.map(r => `${r[labelKey]} (${r[valueKey]})`).join(', ') + '.'
        );
      } else {
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
        <p className="muted">Ask a question in natural language. I’ll query your Azure SQL DB.</p>
      </header>

      <QueryForm
        query={query}
        setQuery={setQuery}
        loading={loading}
        error={err}
        onSubmit={onSubmit}
      />

      {(rows.length || sql) && (
        <section className="grid">
          <ResultPanel
            columns={columns}
            rows={rows}
            summary={summary}
          />
          <SqlPanel sql={sql} />
        </section>
      )}

      <footer>
        <span className="muted">API: <code>{API_BASE}</code></span>
      </footer>
    </div>
  );
}
