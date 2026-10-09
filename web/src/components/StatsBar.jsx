import Icon from "./Icon.jsx";
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
    <section className="inspector" aria-label="Live AI statistics">
      <div className="section-heading">
        <span className="eyebrow">INSIDE THE MACHINE</span>
        <span className={`live-indicator ${thinking ? "busy" : ""}`}>
          {thinking ? "COMPUTING" : "LIVE"}
        </span>
      </div>
      <h3>Every move has a reason.</h3>
      <p className="inspector-intro">
        {stats
          ? `${stats.agent} chose pit ${stats.pit}. ${stats.agent === "AlphaCapture" ? "Here’s what the search explored." : "One forward pass through the trained network."}`
          : "Make your first move to see the AI’s decision, measured in real time."}
      </p>
      <div className="metrics">
        <div>
          <span>{neural ? "Legal moves scored" : "Positions explored"}</span>
          <strong>{fmt(neural ? stats?.choices : stats?.nodes)}</strong>
        </div>
        <div>
          <span>Decision time</span>
          <strong>
            {stats ? (stats.ms < 1 ? "<1" : stats.ms) : "—"}
            <small>ms</small>
          </strong>
        </div>
        <div>
          <span>{neural ? "Network depth" : "Search depth"}</span>
          <strong>
            {neural ? 3 : (stats?.depth ?? depth)}
            <small>{neural ? "layers" : "plies"}</small>
          </strong>
        </div>
        <div>
          <span>{neural ? "Selected Q-value" : "Position value"}</span>
          <strong>
            {stats?.value == null
              ? "—"
              : `${stats.value > 0 ? "+" : ""}${neural ? stats.value.toFixed(2) : stats.value}`}
          </strong>
        </div>
      </div>
      <div className="inspector-note">
        <Icon name="spark" size={16} />
        <span>
          {neural
            ? "15 inputs → 128 → 128 → 6 Q-values. The highest legal Q-value wins the choice; it is not a win probability."
            : "Alpha-beta skips branches that cannot improve the decision. Value is a heuristic score from the AI’s perspective, not a win probability."}
        </span>
      </div>
    </section>
  );
}
