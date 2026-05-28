const MODES = [
  { id: "hvA", label: "vs AlphaCapture", sub: "you vs minimax" },
  { id: "hvC", label: "vs CaptureZero",  sub: "you vs RL agent" },
  { id: "AvC", label: "AI vs AI",         sub: "minimax vs RL" },
];

export default function ModeSelector({ mode, onSelect }) {
  return (
    <div style={{
      display: "flex",
      gap: "0.6rem",
      justifyContent: "center",
      flexWrap: "wrap",
    }}>
      {MODES.map(m => {
        const active = m.id === mode;
        return (
          <button
            key={m.id}
            onClick={() => onSelect(m.id)}
            style={{
              padding: "0.55rem 1rem",
              borderRadius: 8,
              background: active ? "var(--accent)" : "var(--surface)",
              color: active ? "#fff" : "var(--text-muted)",
              border: `1px solid ${active ? "var(--accent)" : "var(--border)"}`,
              fontWeight: active ? 600 : 400,
              fontSize: "0.85rem",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "0.1rem",
              transition: "all 0.15s",
              minWidth: 130,
            }}
          >
            <span>{m.label}</span>
            <span style={{ fontSize: "0.72rem", opacity: 0.7 }}>{m.sub}</span>
          </button>
        );
      })}
    </div>
  );
}
