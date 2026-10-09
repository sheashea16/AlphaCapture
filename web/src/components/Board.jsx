import { pitNumber, playerName } from "../logic/game.js";

function Stones({ count, side, store = false }) {
  const shown = Math.min(count, store ? 20 : 14);
  return (
    <span className={`stones side-${side}`} aria-hidden="true">
      {Array.from({ length: shown }, (_, i) => {
        const angle = i * 2.39996;
        const radius =
          Math.sqrt((i + 0.5) / Math.max(shown, 1)) * (store ? 32 : 29);
        return (
          <span
            className="stone"
            key={i}
            style={{
              left: `${50 + Math.cos(angle) * radius}%`,
              top: `${50 + Math.sin(angle) * radius * (store ? 1.9 : 1)}%`,
              transform: `rotate(${i * 47}deg)`,
            }}
          />
        );
      })}
    </span>
  );
}

export default function Board({
  board,
  currentPlayer,
  legalPits,
  activePit,
  lastPit,
  onPitClick,
  onPreview,
  mode,
  done,
}) {
  const pit = (index) => (
    <div className="pit-cell" key={index}>
      <button
        className={`pit side-${index < 6 ? 0 : 1} ${legalPits.includes(index) ? "legal" : ""} ${activePit === index ? "sowing" : ""} ${lastPit === index ? "last-move" : ""}`}
        disabled={!legalPits.includes(index)}
        onClick={() => onPitClick(index)}
        onMouseEnter={() => onPreview(index)}
        onMouseLeave={() => onPreview(null)}
        onFocus={() => onPreview(index)}
        onBlur={() => onPreview(null)}
        aria-label={`${index < 6 ? "Your" : "Opponent"} pit ${pitNumber(index)}, ${board[index]} stones`}
      >
        <Stones count={board[index]} side={index < 6 ? 0 : 1} />
        <span className="pit-count">{board[index]}</span>
      </button>
      <span className="pit-number">{pitNumber(index)}</span>
    </div>
  );
  return (
    <div className="board-scene">
      <div
        className={`player-label top ${currentPlayer === 1 && !done ? "current" : ""}`}
      >
        <span className="player-dot side-1" />
        <span>{playerName(1, mode)}</span>
        <span className="player-type">
          {mode === "hvA" ? "SEARCH AGENT" : "NEURAL AGENT"}
        </span>
      </div>
      <div className="mancala-board" aria-label="Mancala board">
        <div
          className={`store side-1 ${activePit === 13 ? "sowing" : ""}`}
          aria-label={`Opponent store: ${board[13]}`}
        >
          <Stones count={board[13]} side={1} store />
          <span className="store-count">{board[13]}</span>
          <span className="store-label">
            {mode === "AvC" ? "ZERO" : "THEIRS"}
          </span>
        </div>
        <div className="pit-rows">
          <div className="pit-row opponent-row">
            {[12, 11, 10, 9, 8, 7].map(pit)}
          </div>
          <div className="board-seam">
            <span>←</span>
            <span>CAPTURE / KALAH</span>
            <span>→</span>
          </div>
          <div className="pit-row human-row">{[0, 1, 2, 3, 4, 5].map(pit)}</div>
        </div>
        <div
          className={`store side-0 ${activePit === 6 ? "sowing" : ""}`}
          aria-label={`Your store: ${board[6]}`}
        >
          <Stones count={board[6]} side={0} store />
          <span className="store-count">{board[6]}</span>
          <span className="store-label">
            {mode === "AvC" ? "ALPHA" : "YOURS"}
          </span>
        </div>
      </div>
      <div
        className={`player-label bottom ${currentPlayer === 0 && !done ? "current" : ""}`}
      >
        <span className="player-dot side-0" />
        <span>{playerName(0, mode)}</span>
        <span className="player-type">
          {mode === "AvC" ? "SEARCH AGENT" : "HUMAN PLAYER"}
        </span>
      </div>
    </div>
  );
}
