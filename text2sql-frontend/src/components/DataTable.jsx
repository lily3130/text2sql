export default function DataTable({ columns = [], rows = [] }) {
  return (
    <table>
      <thead>
        <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((r, idx) => (
          <tr key={idx}>
            {columns.map((c) => <td key={c}>{String(r[c])}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}