function winner(board) {
  const p0 = board[6], p1 = board[13];
  if (p0 > p1) return "AlphaCapture (P0) wins!";
  if (p1 > p0) return "CaptureZero (P1) wins!";
  return "Draw!";
}

function winnerHuman(board, mode) {
  const p0 = board[6], p1 = board[13];
  if (mode === "hvA") {
    if (p0 > p1) return "You win!";
    if (p1 > p0) return "AlphaCapture wins.";
    return "Draw.";
  }
  if (mode === "hvC") {
    if (p0 > p1) return "You win!";
    if (p1 > p0) return "CaptureZero wins.";
    return "Draw.";
  }
  return winner(board);
}

export default function StatusBanner({ gameState, mode, thinking, humanPlayer }) {
  const { board, player, done } = gameState;

  let text = "";
  let color = "var(--text-muted)";

  if (done) {
    text = humanPlayer !== null ? winnerHuman(board, mode) : winner(board);
    color = board[6] > board[13]
      ? (mode !== "AvC" ? "#7cf794" : "var(--stone-p0)")
      : board[13] > board[6]
      ? "var(--stone-p1)"
      : "var(--text-muted)";
  } else if (thinking) {
    const agent = mode === "hvC" || (mode === "AvC" && player === 1)
      ? "CaptureZero"
      : "AlphaCapture";
    text = `${agent} is thinking…`;
  } else if (humanPlayer !== null && player === humanPlayer) {
    text = "Your turn — click a pit";
  } else {
    text = mode === "AvC"
      ? (player === 0 ? "AlphaCapture's turn (P0)" : "CaptureZero's turn (P1)")
      : "";
  }

  return (
    <div style={{
      textAlign: "center",
      minHeight: "1.4rem",
      fontSize: "0.9rem",
      fontWeight: 500,
      color,
      transition: "color 0.2s",
    }}>
      {text}
    </div>
  );
}
