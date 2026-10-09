import { actions, result, terminal } from "./alphacapture.js";

export const INITIAL_BOARD = [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0];
export const pitNumber = (index) => (index < 6 ? index + 1 : index - 6);
export const playerName = (player, mode) =>
  player === 0
    ? mode === "AvC"
      ? "AlphaCapture"
      : "You"
    : mode === "hvA"
      ? "AlphaCapture"
      : "CaptureZero";

// Keep sowing frames separate from the settled game state so animation never changes the rules.
export function resolveMove(state, action) {
  if (terminal(state) || !actions(state).includes(action))
    throw new Error("Illegal move");
  const [board, player] = state;
  const store = player === 0 ? 6 : 13;
  const frames = [];
  const sowed = board.slice();
  let stones = sowed[action];
  sowed[action] = 0;
  frames.push({ board: sowed.slice(), active: action });
  let landing = action;
  while (stones > 0) {
    landing = (landing + 1) % 14;
    if (landing === (player === 0 ? 13 : 6)) continue;
    sowed[landing]++;
    stones--;
    frames.push({ board: sowed.slice(), active: landing });
  }
  const captured =
    landing !== store &&
    (player === 0 ? landing < 6 : landing > 6 && landing < 13) &&
    sowed[landing] === 1 &&
    sowed[12 - landing] > 0
      ? sowed[12 - landing] + 1
      : 0;
  const next = result(state, action);
  const done = terminal(next);
  if (done) {
    for (let side = 0; side < 2; side++) {
      const start = side === 0 ? 0 : 7;
      const bank = side === 0 ? 6 : 13;
      for (let i = start; i < start + 6; i++) {
        next[0][bank] += next[0][i];
        next[0][i] = 0;
      }
    }
  }
  return {
    board: next[0],
    player: next[1],
    done,
    frames,
    captured,
    extraTurn: !done && next[1] === player,
    gain: next[0][store] - board[store],
  };
}
