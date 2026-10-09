import { playerName } from "../logic/game.js";
export default function StatusBanner({
  gameState,
  mode,
  thinking,
  animating,
  paused,
}) {
  const { board, player, done } = gameState;
  let title, detail;
  if (done) {
    title =
      board[6] === board[13]
        ? "It’s a tie."
        : `${playerName(board[6] > board[13] ? 0 : 1, mode)} ${board[6] > board[13] && mode !== "AvC" ? "win" : "wins"}.`;
    detail = `Final score: ${board[6]} — ${board[13]}.`;
  } else if (animating) {
    title = "Moving stones…";
    detail = "One stone goes into each pit along the way.";
  } else if (paused) {
    title = mode === "AvC" ? "Paused." : "Couldn’t choose a move.";
    detail =
      mode === "AvC"
        ? "Press Resume to keep playing."
        : "Start a new game to try again.";
  } else if (thinking) {
    title = `${playerName(player, mode)} is thinking.`;
    detail = "Choosing a pit…";
  } else {
    title = "Your turn.";
    detail = "Pick one of your pits along the bottom.";
  }
  return (
    <div className="turn-banner" role="status" aria-live="polite">
      <span className={`turn-orb ${thinking ? "thinking" : ""}`} />
      <div>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>
      {!done && mode !== "AvC" && !thinking && !animating && (
        <span className="keyboard-tip">KEYS 1–6</span>
      )}
    </div>
  );
}
