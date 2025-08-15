// components/QueryForm.jsx
export default function QueryForm({
  query,
  setQuery,
  useEnrichment,
  setUseEnrichment,
  tableWhitelist,
  setTableWhitelist,
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

      {/* 新增：Enrichment 切換 & 白名單 */}
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

        <input
          type="text"
          placeholder="Table whitelist (comma-separated), optional"
          value={tableWhitelist}
          onChange={(e) => setTableWhitelist(e.target.value)}
          disabled={loading}
          style={{ flex: 1 }}
          aria-label="Table whitelist"
        />
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
