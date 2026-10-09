import { bestAction } from "./alphacapture.js";

self.onmessage = ({ data }) => {
  try {
    const move = bestAction([data.board, data.player], data.player, data.depth);
    self.postMessage({ ...move, depth: data.depth });
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
