export default function SqlPanel({ sql = '' }) {
  if (!sql) return null;
  return (
    <div className="card">
      <h3>Generated SQL</h3>
      <pre>{sql}</pre>
    </div>
  );
}
