# program uses minimax algorithm with alpha-beta pruning to play Mancala
# AlphaCapture will determine the best move seqeuence as the "maximizing" player

# creates AlphaCapture class
# sets max player and depth for search
class AlphaCapture:
    def __init__(self, max_player=0, depth=12):
        self.MAX_PLAYER = max_player
        self.depth = depth

    # possible actions in a state
    def actions(self, state):
        board, player = state
        if player == 0:
            return [i for i in range(0, 6) if board[i] > 0]
        else:
            return [i for i in range(7, 13) if board[i] > 0]
    
    # result of applying an action to a state
    def result(self, state, action):
        board, player = state
        new_board = board[:]
        stones = new_board[action]
        new_board[action] = 0
        index = action
        while stones > 0:
            index = (index + 1) % 14
            if (player == 0 and index == 13) or (player == 1 and index == 6):
                continue
            new_board[index] += 1
            stones -= 1
        # Check for capture
        player_pits = range(0, 6) if player == 0 else range(7, 13)
        if index in player_pits and new_board[index] == 1:
            opposite = 12 - index
            if new_board[opposite] > 0:
                mancala = 6 if player == 0 else 13
                new_board[mancala] += new_board[opposite] + 1
                new_board[opposite] = 0
                new_board[index] = 0
        # Check for extra turn
        mancala_index = 6 if player == 0 else 13
        extra_turn = (index == mancala_index)
        new_player = player if extra_turn else 1 - player
        return (new_board, new_player)
    
    # terminal test for a state
    def terminal(self, state):
        board = state[0]
        return sum(board[0:6]) == 0 or sum(board[7:13]) == 0

    # utility of a terminal state
    def utility(self, state):
        board = state[0][:]
        if sum(board[0:6]) == 0:
            board[13] += sum(board[7:13])
            for i in range(7, 13): board[i] = 0
        elif sum(board[7:13]) == 0:
            board[6] += sum(board[0:6])
            for i in range(0, 6): board[i] = 0
        
        if self.MAX_PLAYER == 0:
            return board[6] - board[13]
        else:
            return board[13] - board[6]

    # heuristic evaluation for non-terminal states
    def heuristic(self, state):
        board = state[0]

        # determine MAX (AI) vs MIN (opponent)
        if self.MAX_PLAYER == 0:
            my_store, opp_store = board[6], board[13]
            my_pits = range(0, 6)
            opp_pits = range(7, 13)
            my_skip = 13   # skip opponent store
            opp_skip = 6
        else:
            my_store, opp_store = board[13], board[6]
            my_pits = range(7, 13)
            opp_pits = range(0, 6)
            my_skip = 6
            opp_skip = 13

        # store difference (most important term)
        score = 3 * (my_store - opp_store)

        # capture potential for MAX (AI)
        capture_bonus = 0
        for pit in my_pits:
            stones = board[pit]
            if stones == 0:
                continue
            landing = pit
            s = stones
            while s > 0:
                landing = (landing + 1) % 14
                if landing == my_skip:
                    continue
                s -= 1
            if landing in my_pits and board[landing] == 0:
                capture_bonus += board[12 - landing]

        # capture potential for MIN (opponent)
        capture_penalty = 0
        for pit in opp_pits:
            stones = board[pit]
            if stones == 0:
                continue
            landing = pit
            s = stones
            while s > 0:
                landing = (landing + 1) % 14
                if landing == opp_skip:
                    continue
                s -= 1
            if landing in opp_pits and board[landing] == 0:
                capture_penalty += board[12 - landing]

        return score + 2 * capture_bonus - 2 * capture_penalty

    # minimax algorithm with depth limit and alpha-beta pruning
    def minimax(self, state, depth, alpha, beta):
        if self.terminal(state):
            return self.utility(state)
        if depth == 0:
            return self.heuristic(state)

        player = state[1]

        if player == self.MAX_PLAYER:  # MAX node
            value = float('-inf')
            for action in self.actions(state):
                value = max(
                    value, self.minimax(self.result(state, action), depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        else:  # MIN node
            value = float('inf')
            for action in self.actions(state):
                value = min(
                    value, self.minimax(self.result(state, action), depth - 1, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    # choose the best action using minimax
    def best_action(self, state):
        best_value = float('-inf')
        best_action = None

        for action in self.actions(state):
            value = self.minimax(self.result(state, action), self.depth - 1, float('-inf'), float('inf'))
            if value > best_value:
                best_value = value
                best_action = action

        return best_action
    
    # choose the best sequence of actions using minimax (if possible extra turns)
    def best_sequence(self, state):
    
        sequence = []
        current_state = state

        while True:
            _, player = current_state
            # stop if game is over or it's not MAX's turn
            if self.terminal(current_state) or player != self.MAX_PLAYER:
                break
            best_value = float('-inf')
            best_action = None
            for action in self.actions(current_state):
                value = self.minimax(self.result(current_state, action), self.depth - 1, float('-inf'), float('inf'))
                if value > best_value:
                    best_value = value
                    best_action = action

            # apply the best action
            sequence.append(best_action)
            next_state = self.result(current_state, best_action)

            # stop chaining if turn switches to opponent
            if next_state[1] != self.MAX_PLAYER:
                break

            current_state = next_state

        return sequence

# Example usage:
if __name__ == "__main__":
    initial_board = [4, 4, 4, 4, 4, 4, 0,
                     4, 4, 4, 4, 4, 4, 0]
    initial_state = (initial_board, 0)  # Player 0 starts

    bot = AlphaCapture(max_player=0, depth=8)
    best_seq = bot.best_sequence(initial_state)
    print("Best move sequence for Player 0:", best_seq)

