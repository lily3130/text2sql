export default function BarChart({ data, labelKey, valueKey, height = 260, width = 520, margin = 48 }) {
  if (!data?.length || !labelKey || !valueKey) return null;

  const values = data.map(d => Number(d?.[valueKey]) || 0);
  const maxV = Math.max(...values, 1);
  const innerW = width - margin * 2;
  const innerH = height - margin * 2;
  const gap = 16;
  const barW = Math.max(8, (innerW - gap * (data.length - 1)) / data.length);

  return (
    <svg width={width} height={height} role="img" aria-label="bar chart">
      {/* y 軸標題（旋轉） */}
      <text
        x={-(height / 2)}
        y={16}
        fontSize="12"
        fill="white"
        textAnchor="middle"
        transform="rotate(-90)"
      >
        {valueKey}
      </text>

      {/* x 軸標題 */}
      <text x={width / 2} y={height - 6} textAnchor="middle" fontSize="12" fill="white">
        {labelKey}
      </text>

      {/* y 軸 & x 軸 */}
      <line x1={margin - 6} y1={margin} x2={margin - 6} y2={height - margin} stroke="white" />
      <line x1={margin - 6} y1={height - margin} x2={width - margin + 6} y2={height - margin} stroke="white" />

      {data.map((d, i) => {
        const v = Number(d?.[valueKey]) || 0;
        const h = (v / maxV) * innerH;
        const x = margin + i * (barW + gap);
        const y = height - margin - h;
        return (
          <g key={i}>
            <rect x={x} y={y} width={barW} height={h} fill="yellow">
              <title>{`${d?.[labelKey]}: ${v}`}</title>
            </rect>
            {/* x 軸標籤 */}
            <text x={x + barW / 2} y={height - margin + 14} textAnchor="middle" fontSize="11" fill="white">
              {String(d?.[labelKey])}
            </text>
            {/* 數值：在柱頂上方留距 */}
            <text x={x + barW / 2} y={y - 8} textAnchor="middle" fontSize="11" fill="white">
              {v.toFixed(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
