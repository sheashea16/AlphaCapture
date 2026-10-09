import { useState, useEffect, useRef, useCallback } from "react";
import { actions } from "./logic/alphacapture.js";
import { loadWeights, analyze as czAnalyze } from "./logic/capturezero.js";
import {
  INITIAL_BOARD,
  resolveMove,
  pitNumber,
  playerName,
} from "./logic/game.js";
import Board from "./components/Board.jsx";
import ModeSelector from "./components/ModeSelector.jsx";
import StatsBar from "./components/StatsBar.jsx";
import StatusBanner from "./components/StatusBanner.jsx";
import Icon, { Mark } from "./components/Icon.jsx";

const freshState = () => ({
  board: INITIAL_BOARD.slice(),
  player: 0,
  done: false,
});
const REPO = "https://github.com/sheashea16/AlphaCapture";

export default function App() {
  const [mode, setMode] = useState("hvA");
  const [gameState, setGameState] = useState(freshState);
  const [displayBoard, setDisplayBoard] = useState(null);
  const [activePit, setActivePit] = useState(null);
  const [lastPit, setLastPit] = useState(null);
  const [previewPit, setPreviewPit] = useState(null);
  const [stats, setStats] = useState(null);
  const [weightsReady, setWeightsReady] = useState(false);
  const [error, setError] = useState(null);
  const [animating, setAnimating] = useState(false);
  const [paused, setPaused] = useState(false);
  const [depth, setDepth] = useState(8);
  const [moves, setMoves] = useState([]);
  const [history, setHistory] = useState([]);
  const [rulesOpen, setRulesOpen] = useState(false);
  const timers = useRef([]);
  const rulesRef = useRef(null);
  const isAiTurn =
    !gameState.done && (mode === "AvC" || gameState.player === 1);
  const thinking = isAiTurn && !animating && !paused;

  const initWeights = useCallback(() => {
    setError(null);
    loadWeights()
      .then(() => setWeightsReady(true))
      .catch(() =>
        setError(
          "CaptureZero couldn’t load. You can still play AlphaCapture, or retry the model.",
        ),
      );
  }, []);
  useEffect(() => {
    initWeights();
  }, [initWeights]);
  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  const clearAnimation = useCallback(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
    setDisplayBoard(null);
    setActivePit(null);
    setAnimating(false);
  }, []);
  const resetGame = useCallback(
    (newMode = mode) => {
      clearAnimation();
      setMode(newMode);
      setGameState(freshState());
      setLastPit(null);
      setPreviewPit(null);
      setStats(null);
      setMoves([]);
      setHistory([]);
      setPaused(false);
      if (weightsReady) setError(null);
    },
    [mode, clearAnimation, weightsReady],
  );

  const makeMove = useCallback(
    (action, moveStats = null) => {
      const next = resolveMove([gameState.board, gameState.player], action);
      const actor = playerName(gameState.player, mode);
      const outcome = next.captured
        ? `Captured ${next.captured} stones`
        : next.extraTurn
          ? "Earned an extra turn"
          : next.done
            ? "Collected the remaining stones"
            : next.gain
              ? `Added ${next.gain} to the store`
              : `Sowed ${gameState.board[action]} stone${gameState.board[action] === 1 ? "" : "s"}`;
      setHistory((h) => [...h, { gameState, stats, moves, lastPit }]);
      setAnimating(true);
      setLastPit(action);
      setPreviewPit(null);
      if (moveStats) setStats({ ...moveStats, pit: pitNumber(action) });
      const reduced = window.matchMedia(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      const frames = reduced ? [] : next.frames;
      frames.forEach((frame, i) =>
        timers.current.push(
          setTimeout(() => {
            setDisplayBoard(frame.board);
            setActivePit(frame.active);
          }, i * 65),
        ),
      );
      timers.current.push(
        setTimeout(
          () => {
            setGameState({
              board: next.board,
              player: next.player,
              done: next.done,
            });
            setDisplayBoard(null);
            setActivePit(null);
            setAnimating(false);
            setMoves((m) => [
              ...m,
              {
                actor,
                pit: pitNumber(action),
                outcome,
                extraTurn: next.extraTurn,
                captured: next.captured,
              },
            ]);
            timers.current = [];
          },
          frames.length * 65 + (reduced ? 0 : 180),
        ),
      );
    },
    [gameState, mode, stats, moves, lastPit],
  );

  useEffect(() => {
    if (
      !isAiTurn ||
      animating ||
      paused ||
      (gameState.player === 1 && mode !== "hvA" && !weightsReady)
    )
      return;
    let worker;
    const timer = setTimeout(
      () => {
        if (mode === "hvC" || (mode === "AvC" && gameState.player === 1)) {
          const start = performance.now();
          const decision = czAnalyze([gameState.board, gameState.player], 1);
          if (decision != null)
            makeMove(decision.action, {
              agent: "CaptureZero",
              value: decision.value,
              choices: decision.choices,
              ms: Math.round(performance.now() - start),
            });
        } else {
          worker = new Worker(
            new URL("./logic/search.worker.js", import.meta.url),
            { type: "module" },
          );
          worker.onmessage = ({ data }) => {
            if (data.error) {
              setError(
                "The search hit an error. Restart the game to try again.",
              );
              setPaused(true);
              return;
            }
            if (data.action != null)
              makeMove(data.action, {
                agent: "AlphaCapture",
                ...data.stats,
                depth: data.depth,
              });
          };
          worker.onerror = () => {
            setError(
              "The search couldn’t start. Restart the game to try again.",
            );
            setPaused(true);
          };
          worker.postMessage({
            board: gameState.board,
            player: gameState.player,
            depth,
          });
        }
      },
      mode === "AvC" ? 950 : 400,
    );
    return () => {
      clearTimeout(timer);
      worker?.terminate();
    };
  }, [
    isAiTurn,
    animating,
    paused,
    gameState,
    mode,
    weightsReady,
    depth,
    makeMove,
  ]);

  const legalPits =
    !gameState.done && !isAiTurn && !animating
      ? actions([gameState.board, gameState.player])
      : [];
  const handlePit = useCallback(
    (index) => {
      if (
        !gameState.done &&
        !isAiTurn &&
        !animating &&
        actions([gameState.board, gameState.player]).includes(index)
      )
        makeMove(index);
    },
    [gameState, isAiTurn, animating, makeMove],
  );
  useEffect(() => {
    const onKey = (event) => {
      if (event.target.closest("select, input, textarea, dialog") || rulesOpen)
        return;
      if (/^[1-6]$/.test(event.key)) handlePit(Number(event.key) - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handlePit, rulesOpen]);
  useEffect(() => {
    if (rulesOpen) rulesRef.current.showModal();
    else rulesRef.current.close();
  }, [rulesOpen]);

  function undo() {
    let index = history.length - 1;
    while (index > 0 && history[index].gameState.player !== 0) index--;
    if (index < 0) return;
    const previous = history[index];
    clearAnimation();
    setGameState(previous.gameState);
    setStats(previous.stats);
    setMoves(previous.moves);
    setLastPit(previous.lastPit);
    setHistory(history.slice(0, index));
  }
  const preview =
    previewPit != null && legalPits.includes(previewPit)
      ? resolveMove([gameState.board, gameState.player], previewPit)
      : null;
  const latest = moves.at(-1);
  const description = preview
    ? `Pit ${pitNumber(previewPit)} → ${preview.captured ? `capture ${preview.captured} stones` : preview.extraTurn ? "an extra turn" : `${preview.gain} stone${preview.gain === 1 ? "" : "s"} into your store`}`
    : latest
      ? `${latest.actor} played pit ${latest.pit}. ${latest.outcome}.`
      : mode === "AvC"
        ? "Watch the agents choose their moves. Orange is search; green is learning."
        : "Try pit 3 for an extra turn. Your last stone will land in your store.";

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#">
          <Mark />
          <span>
            AlphaCapture<span className="brand-period">.</span>
          </span>
        </a>
        <nav aria-label="Main navigation">
          <button className="text-button" onClick={() => setRulesOpen(true)}>
            <Icon name="book" />
            How to play
          </button>
          <a href={REPO} target="_blank" rel="noreferrer">
            View source <Icon name="external" size={15} />
          </a>
        </nav>
      </header>
      <main>
        <section className="intro">
          <div>
            <div className="eyebrow intro-eyebrow">
              <span className="orange-line" />
              AN INTERACTIVE AI EXPERIMENT
            </div>
            <h1>
              A small board.
              <br />A big battle of <em>brains.</em>
            </h1>
            <p>
              Six pits. Forty-eight stones. Two very different ways to think.
              <br className="desktop-break" /> Play against an AI—or watch
              search take on learning.
            </p>
          </div>
          <div className="intro-aside">
            <span className="experiment-number">α / 01</span>
            <span>MANCALA, CAPTURE MODE</span>
            <span>HUMAN INTUITION WELCOME.</span>
          </div>
        </section>
        <ModeSelector
          mode={mode}
          onSelect={resetGame}
          weightsReady={weightsReady}
        />
        {error && (
          <div className="error-notice" role="alert">
            {error}{" "}
            {!weightsReady && (
              <button onClick={initWeights}>Retry model</button>
            )}
          </div>
        )}
        <div className="arena-layout">
          <section className="arena" aria-label="Game arena">
            <div className="arena-toolbar">
              <div className="eyebrow">
                <span className="arena-square" />
                THE ARENA{" "}
                <span className="round-label">
                  / MOVE{" "}
                  {String(moves.length + (gameState.done ? 0 : 1)).padStart(
                    2,
                    "0",
                  )}
                </span>
              </div>
              <div className="game-actions">
                {mode === "AvC" ? (
                  <button
                    onClick={() => setPaused((p) => !p)}
                    disabled={animating || gameState.done}
                  >
                    <Icon name={paused ? "play" : "pause"} size={16} />
                    {paused ? "Resume" : "Pause"}
                  </button>
                ) : (
                  <button
                    onClick={undo}
                    disabled={!history.length || animating}
                  >
                    <Icon name="undo" size={16} />
                    Undo
                  </button>
                )}
                <button onClick={() => resetGame()}>
                  <Icon name="reset" size={16} />
                  New game
                </button>
              </div>
            </div>
            <StatusBanner
              gameState={gameState}
              mode={mode}
              thinking={thinking}
              animating={animating}
              paused={paused}
            />
            <Board
              board={displayBoard || gameState.board}
              currentPlayer={gameState.player}
              legalPits={legalPits}
              activePit={activePit}
              lastPit={lastPit}
              onPitClick={handlePit}
              onPreview={setPreviewPit}
              mode={mode}
              done={gameState.done}
            />
            <div className="move-explanation">
              <Icon
                name={
                  preview?.extraTurn || latest?.extraTurn ? "spark" : "arrow"
                }
                size={17}
              />
              <span>{description}</span>
            </div>
            <div className="score-strip">
              <span>
                {mode === "AvC" ? "ALPHACAPTURE" : "YOUR SCORE"}{" "}
                <strong>{gameState.board[6]}</strong>
              </span>
              <div
                className="score-track"
                aria-label={`Score ${gameState.board[6]} to ${gameState.board[13]}`}
              >
                <span
                  style={{ width: `${(gameState.board[6] / 48) * 100}%` }}
                />
                <span
                  style={{ width: `${(gameState.board[13] / 48) * 100}%` }}
                />
              </div>
              <span>
                <strong>{gameState.board[13]}</strong>{" "}
                {mode === "hvA" ? "ALPHACAPTURE" : "CAPTUREZERO"}
              </span>
            </div>
            {gameState.done && (
              <div className="end-game">
                <button className="primary-button" onClick={() => resetGame()}>
                  Play another round <Icon name="arrow" />
                </button>
              </div>
            )}
            <div className="arena-bottom">
              <span>
                <span className="status-dot" />
                {gameState.done ? "ROUND COMPLETE" : "48 STONES IN PLAY"}
              </span>
              <label>
                Search depth{" "}
                <select
                  value={depth}
                  onChange={(e) => setDepth(Number(e.target.value))}
                  disabled={animating || thinking || mode === "hvC"}
                >
                  <option value={4}>4 · Casual</option>
                  <option value={6}>6 · Challenging</option>
                  <option value={8}>8 · Expert</option>
                </select>
              </label>
            </div>
          </section>
          <aside className="lab-sidebar">
            <StatsBar
              stats={stats}
              thinking={thinking}
              depth={depth}
              mode={mode}
            />
            <section className="move-log">
              <div className="section-heading">
                <span className="eyebrow">MOVE JOURNAL</span>
                <span className="journal-count">
                  {String(moves.length).padStart(2, "0")}
                </span>
              </div>
              {moves.length ? (
                <ol>
                  {moves
                    .slice(-4)
                    .reverse()
                    .map((move, i) => (
                      <li key={moves.length - i}>
                        <span className="move-index">
                          {String(moves.length - i).padStart(2, "0")}
                        </span>
                        <div>
                          <strong>
                            {move.actor} <span>→ pit {move.pit}</span>
                          </strong>
                          <p>{move.outcome}</p>
                        </div>
                        {(move.extraTurn || move.captured > 0) && (
                          <span className="move-badge">
                            {move.captured ? "CAPTURE" : "+ TURN"}
                          </span>
                        )}
                      </li>
                    ))}
                </ol>
              ) : (
                <div className="empty-journal">
                  <span className="journal-lines">↗</span>
                  <p>The story starts with your first move.</p>
                </div>
              )}
            </section>
          </aside>
        </div>
        <section className="research" id="experiment">
          <div className="research-heading">
            <span className="eyebrow">THE EXPERIMENT</span>
            <h2>
              Same game.
              <br />
              <em>Different minds.</em>
            </h2>
            <p>
              How far does looking ahead get you?
              <br />
              And can a neural network learn to catch up?
            </p>
            <a href={`${REPO}#the-thesis`} target="_blank" rel="noreferrer">
              Explore the research <Icon name="arrow" size={17} />
            </a>
          </div>
          <article className="agent-story">
            <span className="agent-symbol alpha-symbol">α</span>
            <div className="eyebrow">01 / CLASSICAL SEARCH</div>
            <h3>AlphaCapture</h3>
            <p>
              Thinks ahead. A minimax search explores future moves, while
              alpha-beta pruning cuts branches that won’t change the result.
            </p>
            <div className="agent-footnote">
              <strong>LOOK AHEAD</strong>
              <span>Depth {depth} · heuristic evaluation</span>
            </div>
          </article>
          <article className="agent-story">
            <span className="agent-symbol zero-symbol">0</span>
            <div className="eyebrow">02 / REINFORCEMENT LEARNING</div>
            <h3>CaptureZero</h3>
            <p>
              Learns from experience. A deep Q-network trained against
              AlphaCapture scores the legal moves in a single forward pass.
            </p>
            <div className="agent-footnote">
              <strong>LEARN A POLICY</strong>
              <span>15 → 128 → 128 → 6</span>
            </div>
          </article>
        </section>
        <div className="research-observation">
          <span className="eyebrow">OBSERVATION /</span>
          <p>
            At the training scale explored so far, deep search wins most
            matchups. The gap is the experiment.{" "}
            <a
              href={`${REPO}#capturezero--deep-q-network`}
              target="_blank"
              rel="noreferrer"
            >
              Read the methods and limitations ↗
            </a>
          </p>
        </div>
      </main>
      <footer>
        <a className="brand footer-brand" href="#">
          <Mark />
          AlphaCapture.
        </a>
        <span>A game you can play. An experiment you can inspect.</span>
        <a href={REPO} target="_blank" rel="noreferrer">
          Built by Shea <Icon name="external" size={14} />
        </a>
      </footer>
      <dialog
        aria-labelledby="rules-title"
        ref={rulesRef}
        onCancel={() => setRulesOpen(false)}
        onClick={(e) => {
          if (e.target === e.currentTarget) setRulesOpen(false);
        }}
      >
        <div className="rules-content">
          <div className="section-heading">
            <span className="eyebrow">A ONE-MINUTE FIELD GUIDE</span>
            <button
              className="icon-button"
              onClick={() => setRulesOpen(false)}
              aria-label="Close rules"
            >
              <Icon name="close" />
            </button>
          </div>
          <h2 id="rules-title">
            Easy to learn.
            <br />
            <em>Hard to outthink.</em>
          </h2>
          <ol>
            <li>
              <strong>Pick a pit on your side.</strong>
              <p>
                Take all its stones and drop one into each following pit,
                counterclockwise. Include your store on the right; skip the
                opponent’s store.
              </p>
            </li>
            <li>
              <strong>Land in your store? Go again.</strong>
              <p>
                An extra turn can turn a small move into a big advantage. Try
                pit 3 on the opening board.
              </p>
            </li>
            <li>
              <strong>An empty pit can be a trap.</strong>
              <p>
                Land your last stone in an empty pit on your side to capture it
                and all the stones directly opposite.
              </p>
            </li>
            <li>
              <strong>Most stones wins.</strong>
              <p>
                When either side runs out, the remaining stones go into their
                owner’s store. There are 48 stones; more than 24 wins.
              </p>
            </li>
          </ol>
          <button
            className="primary-button"
            onClick={() => setRulesOpen(false)}
          >
            Let’s play <Icon name="arrow" />
          </button>
        </div>
      </dialog>
    </div>
  );
}
