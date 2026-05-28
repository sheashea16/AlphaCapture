// Minimax with alpha-beta pruning for Mancala.
// Board layout: indices 0-5 = P0 pits, 6 = P0 store, 7-12 = P1 pits, 13 = P1 store.

export function actions(state) {
  const [board, player] = state;
  if (player === 0) return [0,1,2,3,4,5].filter(i => board[i] > 0);
  return [7,8,9,10,11,12].filter(i => board[i] > 0);
}

export function result(state, action) {
  const [board, player] = state;
  const b = board.slice();
  let stones = b[action];
  b[action] = 0;
  let index = action;
  while (stones > 0) {
    index = (index + 1) % 14;
    if (player === 0 && index === 13) continue;
    if (player === 1 && index === 6) continue;
    b[index]++;
    stones--;
  }
  // capture
  const myPits = player === 0 ? [0,1,2,3,4,5] : [7,8,9,10,11,12];
  if (myPits.includes(index) && b[index] === 1) {
    const opposite = 12 - index;
    if (b[opposite] > 0) {
      const store = player === 0 ? 6 : 13;
      b[store] += b[opposite] + 1;
      b[opposite] = 0;
      b[index] = 0;
    }
  }
  // extra turn
  const mancala = player === 0 ? 6 : 13;
  const nextPlayer = index === mancala ? player : 1 - player;
  return [b, nextPlayer];
}

export function terminal(state) {
  const board = state[0];
  return board.slice(0,6).every(x => x === 0) || board.slice(7,13).every(x => x === 0);
}

function utility(state, maxPlayer) {
  const b = state[0].slice();
  if (b.slice(0,6).every(x => x === 0)) {
    b[13] += b.slice(7,13).reduce((a,x) => a+x, 0);
    for (let i = 7; i < 13; i++) b[i] = 0;
  } else {
    b[6] += b.slice(0,6).reduce((a,x) => a+x, 0);
    for (let i = 0; i < 6; i++) b[i] = 0;
  }
  return maxPlayer === 0 ? b[6] - b[13] : b[13] - b[6];
}

function heuristic(state, maxPlayer) {
  const board = state[0];
  const myStore  = maxPlayer === 0 ? board[6]  : board[13];
  const oppStore = maxPlayer === 0 ? board[13] : board[6];
  const myPits  = maxPlayer === 0 ? [0,1,2,3,4,5]   : [7,8,9,10,11,12];
  const oppPits = maxPlayer === 0 ? [7,8,9,10,11,12] : [0,1,2,3,4,5];
  const mySkip  = maxPlayer === 0 ? 13 : 6;
  const oppSkip = maxPlayer === 0 ? 6  : 13;

  let score = 3 * (myStore - oppStore);

  let captureBonus = 0;
  for (const pit of myPits) {
    let stones = board[pit];
    if (!stones) continue;
    let landing = pit;
    let s = stones;
    while (s > 0) {
      landing = (landing + 1) % 14;
      if (landing === mySkip) continue;
      s--;
    }
    if (myPits.includes(landing) && board[landing] === 0) {
      captureBonus += board[12 - landing];
    }
  }

  let capturePenalty = 0;
  for (const pit of oppPits) {
    let stones = board[pit];
    if (!stones) continue;
    let landing = pit;
    let s = stones;
    while (s > 0) {
      landing = (landing + 1) % 14;
      if (landing === oppSkip) continue;
      s--;
    }
    if (oppPits.includes(landing) && board[landing] === 0) {
      capturePenalty += board[12 - landing];
    }
  }

  return score + 2 * captureBonus - 2 * capturePenalty;
}

function minimax(state, depth, alpha, beta, maxPlayer, stats) {
  stats.nodes++;
  if (terminal(state)) return utility(state, maxPlayer);
  if (depth === 0) return heuristic(state, maxPlayer);

  const player = state[1];
  const moves = actions(state);

  if (player === maxPlayer) {
    let value = -Infinity;
    for (const action of moves) {
      value = Math.max(value, minimax(result(state, action), depth - 1, alpha, beta, maxPlayer, stats));
      alpha = Math.max(alpha, value);
      if (alpha >= beta) break;
    }
    return value;
  } else {
    let value = Infinity;
    for (const action of moves) {
      value = Math.min(value, minimax(result(state, action), depth - 1, alpha, beta, maxPlayer, stats));
      beta = Math.min(beta, value);
      if (beta <= alpha) break;
    }
    return value;
  }
}

// Returns { action, stats: { nodes, ms, value } }
export function bestAction(state, maxPlayer = 0, depth = 8) {
  const stats = { nodes: 0 };
  const t0 = performance.now();
  let bestValue = -Infinity;
  let bestAct = null;
  for (const action of actions(state)) {
    const value = minimax(result(state, action), depth - 1, -Infinity, Infinity, maxPlayer, stats);
    if (value > bestValue) { bestValue = value; bestAct = action; }
  }
  stats.ms = Math.round(performance.now() - t0);
  stats.value = bestValue;
  return { action: bestAct, stats };
}

// Chains extra-turn moves. Returns { sequence, stats }
export function bestSequence(state, maxPlayer = 0, depth = 8) {
  const sequence = [];
  let current = state;
  const totalStats = { nodes: 0, ms: 0, value: 0 };
  while (true) {
    const [, player] = current;
    if (terminal(current) || player !== maxPlayer) break;
    const { action, stats } = bestAction(current, maxPlayer, depth);
    if (action === null) break;
    totalStats.nodes += stats.nodes;
    totalStats.ms += stats.ms;
    totalStats.value = stats.value;
    sequence.push(action);
    const next = result(current, action);
    if (next[1] !== maxPlayer) break;
    current = next;
  }
  return { sequence, stats: totalStats };
}
