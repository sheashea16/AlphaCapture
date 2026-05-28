function fmt(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

export default function StatsBar({ stats, thinking }) {
  if (thinking) {
    return (
      <div style={containerStyle}>
        <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>computing…</span>
      </div>
    );
  }
  if (!stats) return <div style={containerStyle} />;

  const items = [];

  items.push({ label: "agent", value: stats.agent });

  if (stats.depth != null) items.push({ label: "depth", value: stats.depth });
  if (stats.nodes != null) items.push({ label: "positions", value: fmt(stats.nodes) });
  if (stats.ms != null)    items.push({ label: "time", value: `${stats.ms}ms` });
  if (stats.value != null) items.push({ label: "eval", value: stats.value > 0 ? `+${stats.value}` : stats.value });
  if (stats.sequence)      items.push({ label: "sequence", value: stats.sequence.join(" → ") });

  return (
    <div style={containerStyle}>
      {items.map(({ label, value }) => (
        <span key={label} style={chipStyle}>
          <span style={{ color: "var(--text-muted)", marginRight: "0.25rem" }}>{label}</span>
          <span style={{ color: "var(--text)", fontVariantNumeric: "tabular-nums" }}>{value}</span>
        </span>
      ))}
    </div>
  );
}

const containerStyle = {
  display: "flex",
  flexWrap: "wrap",
  gap: "0.5rem",
  justifyContent: "center",
  minHeight: "2rem",
  alignItems: "center",
};

const chipStyle = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  padding: "0.25rem 0.6rem",
  fontSize: "0.78rem",
  fontFamily: "ui-monospace, monospace",
  display: "inline-flex",
};
