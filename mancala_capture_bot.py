# program uses minimax algorithm with alpha-beta pruning to play Mancala
# AI can be either player 0 or 1, and plays against a human player

# global vars
MAX_PLAYER = None
state = ([4,4,4,4,4,4,0,4,4,4,4,4,4,0], 0)  # initial state: (board, current_player)

# possible actions in a state
def actions(state):
    board, player = state
    if player == 0:
        return [i for i in range(0, 6) if board[i] > 0]
    else:
        return [i for i in range(7, 13) if board[i] > 0]
    
# add move ordering
def ordered_actions(state):
    possible_actions = actions(state)
    # Order actions by heuristic of resulting state in descending order
    return sorted(possible_actions, key=lambda x: heuristic(result(state, x)), reverse=True)

# result of applying an action to a state
def result(state, action):
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
def terminal(state):
    board = state[0]
    return sum(board[0:6]) == 0 or sum(board[7:13]) == 0

# utility of a terminal state
def utility(state):
    board = state[0][:]
    if sum(board[0:6]) == 0:
        board[13] += sum(board[7:13])
        for i in range(7, 13): board[i] = 0
    elif sum(board[7:13]) == 0:
        board[6] += sum(board[0:6])
        for i in range(0, 6): board[i] = 0
    
    if MAX_PLAYER == 0:
        return board[6] - board[13]
    else:
        return board[13] - board[6]
    
# print the current board state
def print_board(state):
    board = state[0]
    print("Player 2:")
    print("  Side:   ", board[12:6:-1])
    print("  Mancala:", board[13])
    print()
    print("Player 1:")
    print("  Mancala:", board[6])
    print("  Side:   ", board[0:6])
    print()

# heuristic evaluation for non-terminal states
def heuristic(state):
    board = state[0]

    # determine MAX (AI) vs MIN (opponent)
    if MAX_PLAYER == 0:
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
def minimax(state, depth, alpha=float('-inf'), beta=float('inf')):
    if terminal(state) or depth == 0:
        return utility(state) if terminal(state) else heuristic(state)

    player = state[1]

    if player == MAX_PLAYER:  # MAX
        value = float('-inf')
        for action in ordered_actions(state):
            value = max(value, minimax(result(state, action), depth - 1, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:  # MIN
        value = float('inf')
        for action in ordered_actions(state):
            value = min(value, minimax(result(state, action), depth - 1, alpha, beta))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

# choose the best action using minimax
def best_action(state, depth=12):
    best_val = float('-inf')
    best_act = None

    for action in ordered_actions(state):
        val = minimax(result(state, action), depth - 1)
        if val > best_val:
            best_val = val
            best_act = action

    return best_act
    
# game loop
def play_game():
    global state, MAX_PLAYER
    state = ([4,4,4,4,4,4,0,4,4,4,4,4,4,0], 0)
    human_player = int(input("Choose your player: 0 for first, 1 for second: "))
    ai_player = 1 - human_player
    MAX_PLAYER = ai_player
    print(f"You are Player {human_player + 1}, AI is Player {ai_player + 1}")
    print("\n" + "="*40)
    ai_moves = []
    
    while not terminal(state):
        print_board(state)
        print(f"Player {state[1] + 1}'s turn")
        possible_actions = actions(state)
        if state[1] == human_player:
            print(f"Your possible moves: {possible_actions}")
            try:
                action = int(input("Enter your move: "))
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue
        else:
            print(f"AI's possible moves: {possible_actions}")
            print("AI is thinking...")
            action = best_action(state)
            print(f"AI chooses: {action}")
            ai_moves.append(action)
        if action in possible_actions:
            current_player = state[1]
            state = result(state, action)
            if state[1] == current_player:
                print("Extra turn!")
            else:
                if current_player == ai_player:
                    print(f"AI's turn moves: {ai_moves}")
                ai_moves = []
            print("\n" + "="*40)
        else:
            print("Invalid move. Try again.")
    print_board(state)
    print("Game over!")
    util = utility(state)
    if util > 0:
        print("Player 1 wins!")
    elif util < 0:
        print("Player 2 wins!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    play_game()
