// CaptureZero: DQN inference in JS. Mirrors the Python DQN(15->128->128->6).

let W = null;

export async function loadWeights() {
  if (W) return;
  // Vite serves JSON as a static asset; fetch at runtime to avoid bundling 385KB inline.
  const res = await fetch(
    `${import.meta.env?.BASE_URL ?? "/"}dqn_weights.json`,
  );
  if (!res.ok) throw new Error(`Model request failed: ${res.status}`);
  const weights = await res.json();
  for (const [layer, input, output] of [
    [0, 15, 128],
    [2, 128, 128],
    [4, 128, 6],
  ]) {
    const matrix = weights[`net.${layer}.weight`];
    const bias = weights[`net.${layer}.bias`];
    if (
      !Array.isArray(matrix) ||
      matrix.length !== output ||
      !Array.isArray(bias) ||
      bias.length !== output ||
      !bias.every(Number.isFinite) ||
      matrix.some(
        (row) =>
          !Array.isArray(row) ||
          row.length !== input ||
          !row.every(Number.isFinite),
      )
    )
      throw new Error("Invalid model weights");
  }
  W = weights;
}

function relu(arr) {
  return arr.map((x) => (x > 0 ? x : 0));
}

function linear(x, weight, bias) {
  // weight: [out, in], x: [in]
  return weight.map(
    (row, i) => row.reduce((sum, w, j) => sum + w * x[j], 0) + bias[i],
  );
}

function forward(inputVec) {
  let h = linear(inputVec, W["net.0.weight"], W["net.0.bias"]);
  h = relu(h);
  h = linear(h, W["net.2.weight"], W["net.2.bias"]);
  h = relu(h);
  h = linear(h, W["net.4.weight"], W["net.4.bias"]);
  return h; // Q-values for actions 0-5 (mapped to actual board indices by caller)
}

function encodeState(state, dqnPlayer) {
  const [board, player] = state;
  const total = 48.0;
  const vec = board.map((b) => b / total);
  vec.push(player === dqnPlayer ? 1.0 : 0.0);
  return vec;
}

function legalActions(state, dqnPlayer) {
  const [board, player] = state;
  if (player !== dqnPlayer) return [];
  if (dqnPlayer === 0) return [0, 1, 2, 3, 4, 5].filter((i) => board[i] > 0);
  return [0, 1, 2, 3, 4, 5].filter((i) => board[7 + i] > 0);
}

function toPitIndex(action, dqnPlayer) {
  return dqnPlayer === 0 ? action : 7 + action;
}

export function analyze(state, dqnPlayer = 1) {
  if (!W) return null;
  const legal = legalActions(state, dqnPlayer);
  if (!legal.length) return null;
  const vec = encodeState(state, dqnPlayer);
  const q = forward(vec);
  let bestQ = -Infinity,
    bestAct = null;
  for (const a of legal) {
    if (q[a] > bestQ) {
      bestQ = q[a];
      bestAct = a;
    }
  }
  return {
    action: toPitIndex(bestAct, dqnPlayer),
    value: bestQ,
    choices: legal.length,
  };
}

export function bestAction(state, dqnPlayer = 1) {
  return analyze(state, dqnPlayer)?.action ?? null;
}
