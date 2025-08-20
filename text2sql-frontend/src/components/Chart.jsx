// components/BarChart.jsx

// 顏色常數
const AXIS_COLOR = "#000";
const TEXT_COLOR = "#000";
const GRID_COLOR = "#000"; // 若之後要加格線可用

// 判斷欄位型別
const isNumericCol = (col, rows) =>
  rows.every(r => {
    const v = r?.[col];
    return v == null || v === '' || typeof v === 'number' || !Number.isNaN(Number(v));
  });

// 自動挑 x(維度)/y(度量)
const pickDimAndMeasure = (keys, rows) => {
  const toL = s => (s || '').toLowerCase();
  const aggRe = /(sum|total|avg|mean|count|min|max|value|amount|consumption|ktoe|kwh|gwh|mwh)$/i;
  const dimRe = /(year|month|date|quarter|category|type|sector|region|fuel|name|country|city|area)$/i;

  const numeric = keys.filter(k => isNumericCol(k, rows));
  const aggCandidates = keys.filter(k => aggRe.test(toL(k)));
  const dimCandidates = keys.filter(k => !numeric.includes(k) || dimRe.test(toL(k)));

  const xKey = dimCandidates[0] || keys.find(k => !numeric.includes(k)) || keys[0];
  const yKey =
    aggCandidates.find(k => k !== xKey) ||
    numeric.find(k => k !== xKey) ||
    numeric[0] ||
    keys.find(k => k !== xKey) ||
    xKey;

  return { xKey, yKey };
};

const prettyLabel = key =>
  (key || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

const unitFromKey = key => {
  if (/ktoe/i.test(key)) return ' (ktoe)';
  if (/\bgwh\b/i.test(key)) return ' (GWh)';
  if (/\bkwh\b/i.test(key)) return ' (kWh)';
  return '';
};

const fmtNumber = n => {
  const num = Number(n);
  if (!Number.isFinite(num)) return '—';
  const a = Math.abs(num);
  if (a >= 1e9) return (num / 1e9).toFixed(2) + 'B';
  if (a >= 1e6) return (num / 1e6).toFixed(2) + 'M';
  if (a >= 1e3) return (num / 1e3).toFixed(2) + 'K';
  return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
};

export default function BarChart({
  data,
  labelKey: propLabelKey,
  valueKey: propValueKey,
  height = 260,
  width = 520,
  margin = 48,
}) {
  if (!data?.length) return null;

  const keys = Object.keys(data[0] || {});
  let labelKey = propLabelKey;
  let valueKey = propValueKey;
  if (!labelKey || !valueKey) {
    const picked = pickDimAndMeasure(keys, data);
    labelKey = labelKey || picked.xKey;
    valueKey = valueKey || picked.yKey;
  }

  const sorted = [...data].sort(
    (a, b) => (Number(b?.[valueKey]) || 0) - (Number(a?.[valueKey]) || 0)
  );

  const values = sorted.map(d => Number(d?.[valueKey]) || 0);
  const maxV = Math.max(...values, 1);
  const innerW = width - margin * 2;
  const innerH = height - margin * 2;
  const gap = 16;
  const barW = Math.max(8, (innerW - gap * (sorted.length - 1)) / sorted.length);

  return (
    <svg width={width} height={height} role="img" aria-label="bar chart">
      {/* y 軸標題（旋轉） */}
      <text
        x={-(height / 2)}
        y={16}
        fontSize="12"
        fill={TEXT_COLOR}
        textAnchor="middle"
        transform="rotate(-90)"
      >
        {prettyLabel(valueKey)}
        {unitFromKey(valueKey)}
      </text>

      {/* x 軸標題 */}
      <text x={width / 2} y={height - 6} textAnchor="middle" fontSize="12" fill={TEXT_COLOR}>
        {prettyLabel(labelKey)}
      </text>

      {/* y 軸 & x 軸（黑色） */}
      <line x1={margin - 6} y1={margin} x2={margin - 6} y2={height - margin} stroke={AXIS_COLOR} />
      <line x1={margin - 6} y1={height - margin} x2={width - margin + 6} y2={height - margin} stroke={AXIS_COLOR} />

      {sorted.map((d, i) => {
        const v = Number(d?.[valueKey]) || 0;
        const h = (v / (maxV * 1.1)) * innerH; // 多留 10% 頂部空間
        const x = margin + i * (barW + gap);
        const y = height - margin - h;
        return (
          <g key={i}>
            <rect x={x} y={y} width={barW} height={h} fill="yellow">
              <title>{`${d?.[labelKey]}: ${fmtNumber(v)}`}</title>
            </rect>
            {/* x 軸標籤 */}
            <text x={x + barW / 2} y={height - margin + 14} textAnchor="middle" fontSize="11" fill={TEXT_COLOR}>
              {String(d?.[labelKey])}
            </text>
            {/* 數值 */}
            <text x={x + barW / 2} y={Math.min(y - 6, height - margin - 6)} textAnchor="middle" fontSize="11" fill={TEXT_COLOR}>
              {fmtNumber(v)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
