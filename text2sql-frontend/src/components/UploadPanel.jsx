import { useState } from 'react';

// 後端策略固定在這裡（如不想帶，刪掉下一段 append 即可）
const IF_EXISTS_DEFAULT = 'fail';

export default function UploadPanel({ apiBase }) {
  const [file, setFile] = useState(null);
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
      form.append('if_exists', IF_EXISTS_DEFAULT); // 不想帶就刪掉這行

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
    // 移除 className="card" → 不會有外框
    <section style={{ marginTop: 16 }}>
      <h3>Upload CSV/Excel to Azure SQL</h3>
      <div className="row" style={{ gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          disabled={loading}
        />
        <button onClick={onUpload} disabled={loading || !file}>
          {loading ? 'Uploading...' : 'Upload'}
        </button>
      </div>
      {!!msg && <p className="muted" style={{ marginTop: 8 }}>{msg}</p>}
    </section>
  );
}
