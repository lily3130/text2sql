import BarChart from './Chart';
import DataTable from './DataTable';

export default function ResultPanel({ columns = [], rows = [], summary = '' }) {
  if (!rows?.length) return null;

  const labelKey = columns[0];
  const valueKey = columns.find(c => typeof rows[0]?.[c] === 'number') || columns[1];
  const formattedRows = rows.map(row => {
    const newRow = { ...row };
    if (typeof newRow[valueKey] === 'number') {
        newRow[valueKey] = Number(newRow[valueKey].toFixed(2));
    }
    return newRow;
 });

  return (
    <div className="card">
      <h3>Answer</h3>
      <p className="muted" style={{ marginTop: 0 }}>{summary}</p>

      {columns.length >= 2 && valueKey && (
        <BarChart
          data={formattedRows}
        />
      )}

      <div className="table-wrap">
        <DataTable columns={columns} rows={formattedRows} />
      </div>
    </div>
  );
}
