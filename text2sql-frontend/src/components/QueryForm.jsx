export default function QueryForm({ query, setQuery, loading, error, onSubmit }) {
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
      />
      <div className="actions">
        <button type="submit" disabled={loading}>
          {loading ? 'Thinking…' : 'Ask'}
        </button>
      </div>
      {error && <div className="error">⚠ {error}</div>}
    </form>
  );
}
