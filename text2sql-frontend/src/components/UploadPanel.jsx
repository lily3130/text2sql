import { useState } from 'react';

export default function UploadPanel({ apiBase }) {
  const [file, setFile] = useState(null);
  const [ifExists, setIfExists] = useState('replace'); // fail | replace | append
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState('');

  const onUpload = async () => {
    setMsg('');
    if (!file) {
      setMsg('Please choose a CSV/XLSX/XLS file.');
      return;
    }
    setLoading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('if_exists', ifExists); // 僅保留 if_exists

      const res = await fetch(`${apiBase}/upload`, { method: 'POST', body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || data?.error || `HTTP ${res.status}`);

      setMsg(data?.message || 'Upload success.');
    } catch (e) {
      setMsg(e.message || 'Upload failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card" style={{ marginTop: 16 }}>
      <h3>Upload CSV/Excel to Azure SQL</h3>
      <div className="row" style={{ gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          disabled={loading}
        />
        <select value={ifExists} onChange={(e) => setIfExists(e.target.value)} disabled={loading}>
          <option value="replace">replace</option>
          <option value="append">append</option>
          <option value="fail">fail</option>
        </select>
        <button onClick={onUpload} disabled={loading || !file}>
          {loading ? 'Uploading...' : 'Upload'}
        </button>
      </div>
      {!!msg && <p className="muted" style={{ marginTop: 8 }}>{msg}</p>}
    </section>
  );
}
