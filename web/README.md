# AlphaCapture interactive demo

A browser-based Mancala arena for exploring the difference between classical game-tree search and reinforcement learning. Built with React and Vite; both agents run locally in the visitor's browser, with no backend or account required.

## Run

```sh
npm ci
npm run dev
```

```sh
npm test        # rules, stone conservation, and a match using real DQN weights
npm run build  # production bundle, including the search worker
npm run preview
```

## Experience

- Play AlphaCapture or CaptureZero; watch the agents compete with pause/resume.
- Preview a move by hovering or focusing a playable pit. Keyboard keys 1–6 select your pits.
- Follow animated sowing, capture explanations, extra turns, scores, and a move journal.
- Undo your last move together with the opponent's reply.
- Adjust AlphaCapture's search depth: 4, 6, or 8 plies.
- Inspect measured search statistics or the neural agent's selected Q-value.
- Responsive layouts, native rule dialog, keyboard controls, and reduced-motion support.

## Implementation

`src/logic/alphacapture.js` contains the original minimax/alpha-beta engine. `search.worker.js` runs each search outside the UI thread; obsolete searches are terminated on resets, mode changes, undo, and unmount.

`src/logic/capturezero.js` loads and validates `public/dqn_weights.json`, then performs the trained 15→128→128→6 network's forward pass. Illegal actions are masked. Model load failures leave AlphaCapture available and expose a retry.

`src/logic/game.js` derives sowing frames without mutating the live position, explains captures and extra turns, and sweeps remaining stones before final scoring. Search and inference statistics reflect actual decisions, not pre-recorded numbers.

## Deployment

The existing Vercel project should use `web` as its root directory, the Vite framework preset, `npm run build`, and `dist` as its output directory. The worker and model JSON are included by the production build.

For a résumé, link the stable **public production domain**, rather than a protected deployment-specific preview URL. Verify the link in a signed-out browser before sharing it.
