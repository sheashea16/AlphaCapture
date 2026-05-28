# AlphaCapture

**[Play it →](https://web-sandy-pi-ck1rovq00i.vercel.app)**

Three approaches to the same problem: make an AI that plays Mancala well. Classical search, reinforcement learning, and a computer vision attempt at reading a physical board.

---

## The thesis

On a small-state game with cheap simulation, **deep search dominates a tabula-rasa DQN** — at least at the training scales I ran. AlphaCapture (minimax, depth 8) consistently beats CaptureZero (DQN trained against it). The gap is the experiment.

---

## AlphaCapture — minimax with alpha-beta pruning

`agents/alphacapture.py`

A classical game-tree search that finds the best move (or move sequence, if extra turns chain) by searching up to depth 12. Alpha-beta pruning cuts the search tree aggressively — at depth 8 it typically evaluates ~50K positions in under 200ms in the browser.

**Heuristic** (for non-terminal nodes): weighted store difference + capture potential for both players. Three terms:
- `3 × (my_store − opp_store)` — primary objective
- `+2 × capture_bonus` — stones I can capture this turn
- `−2 × capture_penalty` — stones the opponent can capture

The live demo shows the search stats on every AlphaCapture move: depth searched, positions evaluated, time taken, and evaluation score.

---

## CaptureZero — deep Q-network

`agents/capturezero.py` · `notebooks/CaptureZero.ipynb`

A DQN (15→128→128→6) trained via self-play against AlphaCapture at varying depths. The network sees the full board state (14 values normalized by total stones, + a turn indicator) and outputs Q-values for each legal move.

**Training setup**
- Opponent: AlphaCapture at depth 6–8
- Reward: score margin delta per turn + win/loss bonus
- Replay buffer: 50K transitions, batch size 64
- Epsilon decay from 0.13 → 0.05 over 10K steps

**Results (so far):** AlphaCapture wins the majority of matchups at depth 8+. CaptureZero occasionally wins at lower depths. The likely cause: Mancala's branching factor is low enough that exhaustive search with a decent heuristic is hard to beat without substantially more training or self-play curriculum.

Full tournament results (win rates across depths, both first-mover positions) are in progress.

---

## Exploration: PitSight — live board reading

`exploration/pitsight_live.py` · `notebooks/pit_sight.ipynb`

A CNN (`Conv2d` → `MaxPool` × 2 → `AdaptiveAvgPool` → `Linear`) trained to count stones in each pit from a webcam feed. Press space, it reads the board, AlphaCapture computes the best move.

**What didn't work:** accuracy was inconsistent in practice — lighting variation and slight camera angle changes caused significant miscounts. The regression head (predicting stone count as a continuous value, then rounding) was the wrong framing; a classification head over 0–10 stones, or better data augmentation for lighting, would likely help. Kept here as a documented attempt rather than a working feature.

---

## Repo

```
agents/          alphacapture.py, capturezero.py, dqn_mancala.pt
web/             React + Vite app (deployed to Vercel)
notebooks/       CaptureZero.ipynb, pit_sight.ipynb
exploration/     PitSight webcam code
```

## Run locally

```bash
# Web app
cd web
npm install
npm run dev
```

The DQN weights (`public/dqn_weights.json`) are fetched at runtime. AlphaCapture and CaptureZero are ported to plain JS — no backend required.

```bash
# Python agents (requires torch, opencv-python, torchvision)
cd agents
python alphacapture.py    # runs a sample best_sequence call
python capturezero.py     # trains the DQN (eval_games=100 by default)
```
