# AlphaCapture interactive demo

Play Mancala against AlphaCapture or CaptureZero, or watch them play each other. Built with React and Vite. Both computer players run in the browser, with no backend or account required.

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

## Features

- Play AlphaCapture or CaptureZero; watch the agents compete with pause/resume.
- Preview a move by hovering or focusing a playable pit. Keyboard keys 1–6 select your pits.
- Follow animated sowing, capture explanations, extra turns, scores, and recent moves.
- Undo your last move together with the opponent's reply.
- Adjust AlphaCapture's search depth: 4, 6, or 8 plies.
- See boards checked and time taken. Open More stats for the search depth and move score.
- Responsive layouts, native rule dialog, keyboard controls, and reduced-motion support.

## Implementation

`src/logic/alphacapture.js` contains the original minimax/alpha-beta engine. `search.worker.js` runs each search outside the UI thread; obsolete searches are terminated on resets, mode changes, undo, and unmount.

`src/logic/capturezero.js` loads and validates `public/dqn_weights.json`, then performs the trained 15→128→128→6 network's forward pass. Illegal actions are masked. Model load failures leave AlphaCapture available and expose a retry.

`src/logic/game.js` derives sowing frames without mutating the live position, explains captures and extra turns, and sweeps remaining stones before final scoring. Search and inference statistics reflect actual decisions, not pre-recorded numbers.

## Deployment

The public demo is [sheashea16.github.io/AlphaCapture](https://sheashea16.github.io/AlphaCapture/).

`.github/workflows/demo.yml` tests and builds every web pull request. Pushes to `main` also deploy to GitHub Pages. Pages must use **GitHub Actions** as its publishing source. No deployment secrets are needed.

```sh
npm run build:pages  # builds with the /AlphaCapture/ base path
```

The model JSON, favicon, and worker assets resolve correctly under the project subdirectory. `npm run build` remains available for root-domain hosts such as Vercel.
