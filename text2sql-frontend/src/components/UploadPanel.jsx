// components/UploadPanel.jsx
import { useState } from 'react';

export default function UploadPanel({ apiBase }) {
  const [file, setFile] = useState(null);
  const [tableName, setTableName] = useState('');
  const [ifExists, setIfExists] = useState('replace'); // fail | replace | append
  const [sheetName, setSheetName] = useState('');
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
      if (tableName.trim()) form.append('table_name', tableName.trim());
      if (sheetName.trim()) form.append('sheet_name', sheetName.trim());
      form.append('if_exists', ifExists);

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
        <input
          type="text"
          placeholder="table_name (optional)"
          value={tableName}
          onChange={(e) => setTableName(e.target.value)}
          disabled={loading}
        />
        <input
          type="text"
          placeholder="sheet_name (Excel only, optional)"
          value={sheetName}
          onChange={(e) => setSheetName(e.target.value)}
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
