import { useState, useEffect, useRef, useCallback } from "react";
import { actions, result, terminal, bestSequence } from "./logic/alphacapture.js";
import { loadWeights, bestAction as czBestAction } from "./logic/capturezero.js";
import Board from "./components/Board.jsx";
import ModeSelector from "./components/ModeSelector.jsx";
import StatsBar from "./components/StatsBar.jsx";
import StatusBanner from "./components/StatusBanner.jsx";

const INITIAL_BOARD = [4,4,4,4,4,4,0, 4,4,4,4,4,4,0];
const ALPHA_DEPTH = 8;

// Modes: "hvA" (human vs AlphaCapture), "hvC" (human vs CaptureZero), "AvC" (AI vs AI)
// In hvA: human = P0, AlphaCapture = P1
// In hvC: human = P0, CaptureZero = P1
// In AvC: AlphaCapture = P0, CaptureZero = P1

function freshState() {
  return { board: INITIAL_BOARD.slice(), player: 0, done: false };
}

export default function App() {
  const [mode, setMode] = useState("hvA");
  const [gameState, setGameState] = useState(freshState());
  const [highlighted, setHighlighted] = useState([]);  // pit indices to highlight as AI's move
  const [stats, setStats] = useState(null);
  const [weightsReady, setWeightsReady] = useState(false);
  const [thinking, setThinking] = useState(false);
  const aiTimerRef = useRef(null);

  // Load CaptureZero weights once
  useEffect(() => {
    loadWeights().then(() => setWeightsReady(true));
  }, []);

  const resetGame = useCallback((newMode = mode) => {
    clearTimeout(aiTimerRef.current);
    setMode(newMode);
    setGameState(freshState());
    setHighlighted([]);
    setStats(null);
    setThinking(false);
  }, [mode]);

  // Determine if it's an AI's turn
  const isAiTurn = useCallback((gs, m) => {
    if (gs.done) return false;
    if (m === "AvC") return true;          // always AI
    if (m === "hvA") return gs.player === 1; // AlphaCapture is P1
    if (m === "hvC") return gs.player === 1; // CaptureZero is P1
    return false;
  }, []);

  const makeAiMove = useCallback((gs, m) => {
    if (gs.done) return;
    const state = [gs.board, gs.player];
    let action = null;
    let moveStats = null;

    if (m === "hvC" || (m === "AvC" && gs.player === 1)) {
      // CaptureZero (P1)
      if (!weightsReady) return;
      action = czBestAction(state, 1);
      moveStats = { agent: "CaptureZero", depth: null, nodes: null, ms: null, value: null };
    } else {
      // AlphaCapture (P0 in AvC, P1 in hvA)
      const maxPlayer = m === "AvC" ? 0 : 1;
      const { sequence, stats: s } = bestSequence(state, maxPlayer, ALPHA_DEPTH);
      action = sequence[0] ?? null;
      moveStats = {
        agent: "AlphaCapture",
        depth: ALPHA_DEPTH,
        nodes: s.nodes,
        ms: s.ms,
        value: s.value,
        sequence,
      };
    }

    if (action === null) return;

    const next = result(state, action);
    const done = terminal(next);
    const newGs = { board: next[0], player: next[1], done };
    setHighlighted([action]);
    setStats(moveStats);
    setGameState(newGs);
  }, [weightsReady]);

  // Trigger AI moves
  useEffect(() => {
    if (!isAiTurn(gameState, mode)) return;
    setThinking(true);
    // Small delay so React can render "thinking" state before blocking minimax
    aiTimerRef.current = setTimeout(() => {
      makeAiMove(gameState, mode);
      setThinking(false);
    }, mode === "AvC" ? 1500 : 200);
    return () => clearTimeout(aiTimerRef.current);
  }, [gameState, mode, isAiTurn, makeAiMove]);

  const handlePitClick = useCallback((pitIndex) => {
    if (gameState.done || isAiTurn(gameState, mode) || thinking) return;
    const state = [gameState.board, gameState.player];
    const legal = actions(state);
    if (!legal.includes(pitIndex)) return;
    setHighlighted([]);
    setStats(null);
    const next = result(state, pitIndex);
    setGameState({ board: next[0], player: next[1], done: terminal(next) });
  }, [gameState, mode, isAiTurn, thinking]);

  const humanPlayer = mode === "AvC" ? null : 0;
  const isHumanTurn = !gameState.done && !thinking && humanPlayer !== null && gameState.player === humanPlayer;
  const legalPits = isHumanTurn ? actions([gameState.board, gameState.player]) : [];

  return (
    <div style={{ width: "100%", maxWidth: 680, display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      <header style={{ textAlign: "center" }}>
        <h1 style={{ fontSize: "1.75rem", fontWeight: 700, letterSpacing: "-0.03em", color: "var(--text)" }}>
          AlphaCapture
        </h1>
        <p style={{ marginTop: "0.35rem", color: "var(--text-muted)", fontSize: "0.9rem" }}>
          Classical minimax vs reinforcement learning — on Mancala.
        </p>
      </header>

      <ModeSelector mode={mode} onSelect={resetGame} />

      <StatusBanner
        gameState={gameState}
        mode={mode}
        thinking={thinking}
        humanPlayer={humanPlayer}
      />

      <Board
        board={gameState.board}
        currentPlayer={gameState.player}
        legalPits={legalPits}
        highlighted={highlighted}
        onPitClick={handlePitClick}
        mode={mode}
      />

      <StatsBar stats={stats} thinking={thinking} />

      {gameState.done && (
        <div style={{ textAlign: "center" }}>
          <button
            onClick={() => resetGame(mode)}
            style={{
              background: "var(--accent)",
              color: "#fff",
              padding: "0.6rem 1.5rem",
              borderRadius: 8,
              fontWeight: 600,
              fontSize: "0.95rem",
              transition: "opacity 0.15s",
            }}
            onMouseOver={e => e.target.style.opacity = 0.85}
            onMouseOut={e => e.target.style.opacity = 1}
          >
            Play again
          </button>
        </div>
      )}
    </div>
  );
}
