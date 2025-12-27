# a deep q-network designed to play mancala against my alphacapture agent
import math
import os
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from alphacapture import AlphaCapture

Transition = namedtuple(
    "Transition", "state action reward next_state done next_legal"
)

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_dim=15, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MancalaEnv:
    def __init__(self, dqn_player=0, opponent_depth=6, random_start=False):
        self.dqn_player = dqn_player
        self.opponent = AlphaCapture(max_player=1 - dqn_player, depth=opponent_depth)
        self.random_start = random_start
        self.board = []
        self.player = 0

    def reset(self):
        self.board = [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]
        self.player = random.choice([0, 1]) if self.random_start else 0
        state = (self.board[:], self.player)
        if self.player != self.dqn_player:
            state = self._advance_opponent(state)
        self.board, self.player = state
        return state

    def legal_actions(self, state=None):
        if state is None:
            board, player = self.board, self.player
        else:
            board, player = state
        if player != self.dqn_player:
            return []
        if self.dqn_player == 0:
            return [i for i in range(6) if board[i] > 0]
        return [i for i in range(6) if board[7 + i] > 0]

    def _to_pit_index(self, action):
        return action if self.dqn_player == 0 else 7 + action

    def _score(self, board):
        return board[6] - board[13] if self.dqn_player == 0 else board[13] - board[6]

    def _advance_opponent(self, state):
        board, player = state
        while not self.opponent.terminal((board, player)) and player != self.dqn_player:
            action = self.opponent.best_action((board, player))
            if action is None:
                break
            board, player = self.opponent.result((board, player), action)
        return (board, player)

    def step(self, action):
        if self.player != self.dqn_player:
            raise ValueError("Not DQN player's turn.")
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError("Illegal action.")
        before_score = self._score(self.board)
        state = (self.board[:], self.player)
        pit_index = self._to_pit_index(action)
        next_state = self.opponent.result(state, pit_index)
        if not self.opponent.terminal(next_state) and next_state[1] != self.dqn_player:
            next_state = self._advance_opponent(next_state)
        board, player = next_state
        self.board, self.player = board[:], player
        done = self.opponent.terminal(next_state)
        after_score = self._score(board)
        reward = after_score - before_score
        if done:
            if after_score > 0:
                reward += 1.0
            elif after_score < 0:
                reward -= 1.0
        return (self.board[:], self.player), reward, done


def encode_state(state, dqn_player):
    board, player = state
    total_stones = 48.0
    vec = [b / total_stones for b in board]
    vec.append(1.0 if player == dqn_player else 0.0)
    return torch.tensor(vec, dtype=torch.float32)


def select_action(policy_net, state_vec, legal_actions, epsilon, device):
    if random.random() < epsilon:
        return random.choice(legal_actions)
    with torch.no_grad():
        q_values = policy_net(state_vec.to(device).unsqueeze(0)).squeeze(0)
        mask = torch.full_like(q_values, -1e9)
        mask[legal_actions] = 0.0
        q_values = q_values + mask
        return int(torch.argmax(q_values).item())


def optimize_model(
    policy_net,
    target_net,
    optimizer,
    batch,
    gamma,
    device,
):
    states = torch.stack(batch.state).to(device)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_states = torch.stack(batch.next_state).to(device)
    dones = torch.tensor(batch.done, dtype=torch.bool, device=device)

    q_values = policy_net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states)
        mask = torch.full_like(next_q, -1e9)
        for i, legal in enumerate(batch.next_legal):
            if legal:
                mask[i, legal] = 0.0
            else:
                mask[i, :] = 0.0
        next_q = next_q + mask
        next_max = next_q.max(1).values
        targets = rewards + gamma * next_max * (~dones)

    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(policy_net, env, games=100, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.eval()
    wins = 0
    losses = 0
    draws = 0
    total_diff = 0

    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            legal = env.legal_actions(state)
            if not legal:
                break
            state_vec = encode_state(state, env.dqn_player)
            action = select_action(policy_net, state_vec, legal, epsilon=0.0, device=device)
            state, _, done = env.step(action)
        diff = env._score(env.board)
        total_diff += diff
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
        else:
            draws += 1

    avg_diff = total_diff / games if games else 0.0
    print(
        f"Eval {games} games | wins {wins} | losses {losses} | draws {draws} | "
        f"avg store diff {avg_diff:.2f}"
    )
    policy_net.train()


def train(
    episodes=2000,
    max_steps=200,
    batch_size=64,
    gamma=0.99,
    lr=5e-4,
    replay_capacity=50000,
    start_training=500,
    target_update=1000,
    epsilon_start=0.35,
    epsilon_end=0.05,
    epsilon_decay=20000,
    opponent_depth=8,
    seed=0,
    save_path="dqn_mancala.pt",
    load_path=None,
    eval_games=0,
):
    random.seed(seed)
    torch.manual_seed(seed)

    env = MancalaEnv(dqn_player=0, opponent_depth=opponent_depth, random_start=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN().to(device)
    resume_path = load_path or save_path
    if resume_path and os.path.exists(resume_path):
        policy_net.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"Loaded model from {resume_path}")
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(replay_capacity)
    steps = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        done = False

        for _ in range(max_steps):
            legal = env.legal_actions(state)
            if not legal:
                break
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(
                -1.0 * steps / epsilon_decay
            )
            state_vec = encode_state(state, env.dqn_player)
            action = select_action(policy_net, state_vec, legal, epsilon, device)
            next_state, reward, done = env.step(action)
            next_state_vec = encode_state(next_state, env.dqn_player)
            next_legal = env.legal_actions(next_state) if not done else []

            buffer.push(
                state_vec,
                action,
                reward,
                next_state_vec,
                done,
                next_legal,
            )

            state = next_state
            episode_reward += reward
            steps += 1

            if len(buffer) >= max(batch_size, start_training):
                batch = Transition(*zip(*buffer.sample(batch_size)))
                optimize_model(
                    policy_net,
                    target_net,
                    optimizer,
                    batch,
                    gamma,
                    device,
                )

            if steps % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        if episode % 50 == 0:
            print(
                f"Episode {episode} | reward {episode_reward:.2f} | "
                f"epsilon {epsilon:.3f}"
            )

    torch.save(policy_net.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    if eval_games > 0:
        evaluate(policy_net, env, eval_games, device)


if __name__ == "__main__":
    train(eval_games=500)
