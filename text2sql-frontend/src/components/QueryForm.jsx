export default function QueryForm({
  query,
  setQuery,
  useEnrichment,
  setUseEnrichment,
  loading,
  error,
  onSubmit,
}) {
  const submit = (e) => { e.preventDefault(); onSubmit(); };

  return (
    <form onSubmit={submit} className="card">
      <label htmlFor="q">Your question</label>
      <textarea
        id="q"
        rows={3}
        placeholder='e.g., "List the top 5 years with the highest energy consumption."'
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        disabled={loading}
      />

      {/* 只保留 Enrichment 勾選；已移除 Table whitelist */}
      <div className="row" style={{ marginTop: 8, gap: 12, alignItems: 'center' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input
            type="checkbox"
            checked={useEnrichment}
            onChange={(e) => setUseEnrichment(e.target.checked)}
            disabled={loading}
          />
          Use Text Enrichment
        </label>
      </div>

      <div className="actions">
        <button type="submit" disabled={loading}>
          {loading ? 'Thinking…' : 'Ask'}
        </button>
      </div>

      {error && <div className="error">⚠ {error}</div>}
    </form>
  );
}
