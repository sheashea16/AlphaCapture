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
        ? "A perfect stalemate."
        : `${playerName(board[6] > board[13] ? 0 : 1, mode)} ${board[6] > board[13] && mode !== "AvC" ? "win" : "wins"}.`;
    detail = `Final score ${board[6]} — ${board[13]}. All remaining stones have been collected.`;
  } else if (animating) {
    title = "Stones in motion.";
    detail = "One stone in each pit. Keep an eye on where the last one lands.";
  } else if (paused) {
    title = mode === "AvC" ? "The arena is paused." : "The search stopped.";
    detail =
      mode === "AvC"
        ? "Press resume to watch the next decision unfold."
        : "Start a new game to retry the search.";
  } else if (thinking) {
    title = `${playerName(player, mode)} is thinking.`;
    detail =
      playerName(player, mode) === "AlphaCapture"
        ? "Exploring future positions with alpha-beta pruning."
        : "Evaluating this position with the trained Q-network.";
  } else {
    title = "Your move.";
    detail =
      "Choose a pit on your side. Collect more stones than the AI to win.";
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
