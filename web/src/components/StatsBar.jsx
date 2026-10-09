const fmt = (n) =>
  n == null
    ? "—"
    : new Intl.NumberFormat("en", {
        notation: "compact",
        maximumFractionDigits: 1,
      }).format(n);

export default function StatsBar({ stats, thinking, depth, mode }) {
  const neural = stats ? stats.agent === "CaptureZero" : mode === "hvC";
  return (
    <section className="inspector" aria-label="Computer move stats">
      <div className="section-heading">
        <span className="eyebrow">LAST COMPUTER MOVE</span>
        <span className={`live-indicator ${thinking ? "busy" : ""}`}>
          {thinking ? "THINKING" : "READY"}
        </span>
      </div>
      <h3>{stats ? `Pit ${stats.pit}` : "No move yet."}</h3>
      <p className="inspector-intro">
        {stats
          ? `${stats.agent} picked this pit.`
          : "Stats appear after the computer plays."}
      </p>
      <div className="metrics">
        <div>
          <span>{neural ? "Moves rated" : "Boards checked"}</span>
          <strong>{fmt(neural ? stats?.choices : stats?.nodes)}</strong>
        </div>
        <div>
          <span>Time taken</span>
          <strong>
            {stats ? (stats.ms < 1 ? "<1" : stats.ms) : "—"}
            <small>ms</small>
          </strong>
        </div>
      </div>
      <details className="extra-stats">
        <summary>More stats</summary>
        {!neural && (
          <p>
            Moves ahead: <strong>{stats?.depth ?? depth}</strong>
          </p>
        )}
        <p>
          Move score:{" "}
          <strong>
            {stats?.value == null
              ? "—"
              : `${stats.value > 0 ? "+" : ""}${neural ? stats.value.toFixed(2) : stats.value}`}
          </strong>
        </p>
        <p>
          Higher scores mean this player prefers the move. They aren’t win odds.
        </p>
      </details>
    </section>
  );
}
