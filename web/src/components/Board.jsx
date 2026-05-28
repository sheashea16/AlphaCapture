// Board layout:
//   P1 store (13) | P1 pits 12..7 (top row, right-to-left) | P0 store (6)
//   P0 pits 0..5  (bottom row, left-to-right)

function Stone({ color }) {
  return (
    <div style={{
      width: 10, height: 10,
      borderRadius: "50%",
      background: color,
      opacity: 0.9,
      flexShrink: 0,
    }} />
  );
}

function Pit({ index, count, isLegal, isHighlighted, isStore, player, onClick }) {
  const stoneColor = player === 0 ? "var(--stone-p0)" : "var(--stone-p1)";
  const storeColor = isStore
    ? (index === 6 ? "var(--stone-p0)" : "var(--stone-p1)")
    : stoneColor;

  const displayCount = Math.min(count, 24);
  const stones = Array.from({ length: displayCount });

  const base = {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "0.3rem",
    borderRadius: isStore ? "var(--radius)" : "50%",
    background: "var(--pit-empty)",
    border: isHighlighted
      ? "2px solid var(--accent)"
      : isLegal
      ? "2px solid var(--accent-dim)"
      : "2px solid var(--border)",
    cursor: isLegal ? "pointer" : "default",
    transition: "border-color 0.15s, transform 0.1s",
    position: "relative",
    overflow: "hidden",
    userSelect: "none",
    padding: "0.4rem",
  };

  return (
    <div
      style={{
        ...base,
        ...(isLegal ? { transform: "scale(1.05)" } : {}),
      }}
      onClick={isLegal ? onClick : undefined}
      title={`Pit ${index}: ${count} stone${count !== 1 ? "s" : ""}`}
    >
      <span style={{
        fontSize: "0.85rem",
        fontWeight: 700,
        color: count > 0 ? "var(--text)" : "var(--text-muted)",
        lineHeight: 1,
      }}>
        {count}
      </span>
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        gap: 3,
        justifyContent: "center",
        alignContent: "center",
        flex: 1,
        width: "100%",
      }}>
        {stones.map((_, i) => <Stone key={i} color={storeColor} />)}
        {count > 24 && <span style={{ fontSize: "0.6rem", color: "var(--text-muted)" }}>+{count - 24}</span>}
      </div>
    </div>
  );
}

export default function Board({ board, currentPlayer, legalPits, highlighted, onPitClick, mode }) {
  const pitSize = 72;
  const storeW = 64;
  const storeH = pitSize * 2 + 10;

  const pitStyle = { width: pitSize, height: pitSize };
  const storeStyle = { width: storeW, height: storeH };

  // Labels
  const p0Label = mode === "AvC" ? "AlphaCapture (P0)" : "You (P0)";
  const p1Label = mode === "hvA" ? "AlphaCapture (P1)" : mode === "hvC" ? "CaptureZero (P1)" : "CaptureZero (P1)";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", alignItems: "center" }}>
      {/* P1 label */}
      <div style={{ width: "100%", display: "flex", justifyContent: "center" }}>
        <span style={{
          fontSize: "0.75rem",
          color: currentPlayer === 1 && !highlighted.length ? "var(--stone-p1)" : "var(--text-muted)",
          fontWeight: 500,
          transition: "color 0.2s",
        }}>
          {p1Label}
        </span>
      </div>

      {/* Main board */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: "0.5rem",
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        padding: "0.75rem",
        width: "100%",
        justifyContent: "center",
      }}>
        {/* P1 store (index 13) */}
        <div style={storeStyle}>
          <Pit
            index={13}
            count={board[13]}
            isStore
            player={1}
            isLegal={false}
            isHighlighted={false}
          />
        </div>

        {/* Pits grid: 2 rows × 6 cols */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          {/* P1 top row: pits 12 down to 7 (right-to-left) */}
          <div style={{ display: "flex", gap: "0.5rem" }}>
            {[12,11,10,9,8,7].map(i => (
              <div key={i} style={pitStyle}>
                <Pit
                  index={i}
                  count={board[i]}
                  player={1}
                  isLegal={legalPits.includes(i)}
                  isHighlighted={highlighted.includes(i)}
                  onClick={() => onPitClick(i)}
                />
              </div>
            ))}
          </div>
          {/* P0 bottom row: pits 0 to 5 */}
          <div style={{ display: "flex", gap: "0.5rem" }}>
            {[0,1,2,3,4,5].map(i => (
              <div key={i} style={pitStyle}>
                <Pit
                  index={i}
                  count={board[i]}
                  player={0}
                  isLegal={legalPits.includes(i)}
                  isHighlighted={highlighted.includes(i)}
                  onClick={() => onPitClick(i)}
                />
              </div>
            ))}
          </div>
        </div>

        {/* P0 store (index 6) */}
        <div style={storeStyle}>
          <Pit
            index={6}
            count={board[6]}
            isStore
            player={0}
            isLegal={false}
            isHighlighted={false}
          />
        </div>
      </div>

      {/* P0 label */}
      <div style={{ width: "100%", display: "flex", justifyContent: "center" }}>
        <span style={{
          fontSize: "0.75rem",
          color: currentPlayer === 0 && !highlighted.length ? "var(--stone-p0)" : "var(--text-muted)",
          fontWeight: 500,
          transition: "color 0.2s",
        }}>
          {p0Label}
        </span>
      </div>
    </div>
  );
}
