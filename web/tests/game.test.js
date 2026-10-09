import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { INITIAL_BOARD, resolveMove } from "../src/logic/game.js";
import {
  actions,
  bestAction,
  result,
  terminal,
} from "../src/logic/alphacapture.js";
import {
  loadWeights,
  bestAction as neuralAction,
} from "../src/logic/capturezero.js";

const total = (board) => board.reduce((a, b) => a + b, 0);

test("an opening move into your store earns another turn", () => {
  const next = resolveMove([INITIAL_BOARD, 0], 2);
  assert.equal(next.player, 0);
  assert.equal(next.extraTurn, true);
  assert.equal(next.board[6], 1);
  assert.equal(next.frames.length, 5);
  assert.equal(total(next.board), 48);
  assert.equal(INITIAL_BOARD[2], 4);
});

test("landing in an empty own pit captures it and the opposite stones", () => {
  const board = [1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0];
  const next = resolveMove([board, 0], 0);
  assert.equal(next.captured, 6);
  assert.equal(next.board[6], 6);
  assert.equal(next.board[1], 0);
  assert.equal(next.board[11], 0);
  assert.equal(total(next.board), total(board));
});

test("sowing skips the opponent store, including on a full lap", () => {
  const board = [1, 1, 1, 1, 1, 15, 0, 1, 1, 1, 1, 1, 1, 0];
  const next = resolveMove([board, 0], 5);
  assert.equal(next.board[13], 0);
  assert.equal(next.board[6], 2);
  assert.equal(total(next.board), total(board));
});

test("the final move sweeps both sides before scoring", () => {
  const board = [0, 0, 0, 0, 0, 1, 20, 5, 5, 5, 4, 4, 4, 0];
  const next = resolveMove([board, 0], 5);
  assert.equal(next.done, true);
  assert.equal(next.extraTurn, false);
  assert.equal(next.board[6], 21);
  assert.equal(next.board[13], 27);
  assert.deepEqual(next.board.slice(7, 13), [0, 0, 0, 0, 0, 0]);
  assert.equal(total(next.board), 48);
});

test("empty, wrong-side, and terminal moves are rejected", () => {
  assert.throws(() => resolveMove([INITIAL_BOARD, 0], 7), /Illegal/);
  assert.throws(
    () => resolveMove([[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 0], 0),
    /Illegal/,
  );
  assert.throws(
    () => resolveMove([[0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0], 0], 0),
    /Illegal/,
  );
});

test("seeded games preserve every stone, match engine moves, and terminate", () => {
  let seed = 17;
  const random = (max) => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed % max;
  };
  for (let game = 0; game < 60; game++) {
    let state = [INITIAL_BOARD.slice(), game % 2];
    let plies = 0;
    while (!terminal(state) && plies < 500) {
      const legal = actions(state);
      const move = legal[random(legal.length)];
      const next = resolveMove(state, move);
      if (!next.done)
        assert.deepEqual([next.board, next.player], result(state, move));
      assert.equal(total(next.board), 48);
      next.frames.forEach((frame) =>
        assert.equal(
          total(frame.board),
          48 - state[0][move] + next.frames.indexOf(frame),
        ),
      );
      assert.ok(next.board.every((n) => Number.isInteger(n) && n >= 0));
      state = [next.board, next.player];
      plies++;
    }
    assert.ok(plies < 500);
    assert.equal(state[0][6] + state[0][13], 48);
  }
});

test("real CaptureZero weights and alpha-beta search complete a legal match", async () => {
  const weights = JSON.parse(
    await readFile(new URL("../public/dqn_weights.json", import.meta.url)),
  );
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => weights });
  try {
    await loadWeights();
  } finally {
    globalThis.fetch = originalFetch;
  }
  let state = [INITIAL_BOARD.slice(), 0];
  let plies = 0;
  while (!terminal(state) && plies < 300) {
    const move =
      state[1] === 0 ? bestAction(state, 0, 4).action : neuralAction(state, 1);
    assert.ok(actions(state).includes(move));
    const next = resolveMove(state, move);
    assert.equal(total(next.board), 48);
    state = [next.board, next.player];
    plies++;
  }
  assert.ok(plies < 300);
  assert.equal(state[0][6] + state[0][13], 48);
});
