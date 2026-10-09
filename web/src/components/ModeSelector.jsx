const MODES = [
  {
    id: "hvA",
    label: "Play AlphaCapture",
    sub: "Looks ahead before choosing",
    number: "01",
  },
  {
    id: "hvC",
    label: "Play CaptureZero",
    sub: "Learned by playing games",
    number: "02",
  },
  {
    id: "AvC",
    label: "Watch them compete",
    sub: "Let the computers play",
    number: "03",
  },
];
export default function ModeSelector({ mode, onSelect, weightsReady }) {
  return (
    <div className="mode-selector" role="group" aria-label="Game mode">
      {MODES.map((m) => (
        <button
          key={m.id}
          className={`mode-button ${m.id === mode ? "selected" : ""}`}
          onClick={() => onSelect(m.id)}
          disabled={m.id !== "hvA" && !weightsReady}
          aria-pressed={m.id === mode}
        >
          <span className="mode-number">{m.number}</span>
          <span>
            <strong>{m.label}</strong>
            <small>{m.sub}</small>
          </span>
          <span className="mode-indicator" />
        </button>
      ))}
    </div>
  );
}
