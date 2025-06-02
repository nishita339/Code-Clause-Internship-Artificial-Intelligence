import numpy as np # type: ignore

def get_best_move(board):
    """
    Use the Minimax algorithm to determine the best move for the AI.
    
    Args:
        board: The current game board state (3x3 list)
    
    Returns:
        Tuple (row, col) representing the best move
    """
    # Find all available moves
    available_moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == '':
                available_moves.append((i, j))
    
    # If there are no available moves, return None
    if not available_moves:
        return None
    
    # If this is the first move, just pick a random corner or center
    # This optimization speeds up the first move which would otherwise take a long time
    if len(available_moves) == 9:
        corners_and_center = [(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)]
        return corners_and_center[np.random.randint(0, len(corners_and_center))]
    
    # Find the move with the best score
    best_score = float('-inf')
    best_move = None
    
    for move in available_moves:
        row, col = move
        # Try this move
        board[row][col] = 'O'
        # Calculate the score for this move
        score = minimax(board, 0, False)
        # Undo the move
        board[row][col] = ''
        
        # Update best move if this move has a better score
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move

def minimax(board, depth, is_maximizing):
    """
    The Minimax algorithm implementation.
    
    Args:
        board: The current game board state
        depth: How deep in the game tree we are
        is_maximizing: True if it's the AI's turn (maximizing player), False if it's the human's turn (minimizing player)
    
    Returns:
        The score of the best move
    """
    # Define scores for terminal states
    scores = {
        'X': -10,  # Human wins
        'O': 10,   # AI wins
        'Draw': 0  # Draw
    }
    
    # Check if the game is over
    winner = check_winner_for_minimax(board)
    if winner:
        return scores[winner]
    
    # Check for a draw
    if is_board_full(board):
        return scores['Draw']
    
    # Maximizing player (AI)
    if is_maximizing:
        best_score = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ''
                    best_score = max(score, best_score)
        return best_score
    
    # Minimizing player (Human)
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ''
                    best_score = min(score, best_score)
        return best_score

def check_winner_for_minimax(board):
    """
    Check if there's a winner on the board.
    
    Args:
        board: The game board
    
    Returns:
        'X' if player won, 'O' if AI won, None if no winner
    """
    # Check rows
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] and board[row][0] != '':
            return board[row][0]
    
    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != '':
            return board[0][col]
    
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != '':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != '':
        return board[0][2]
    
    # No winner
    return None

def is_board_full(board):
    """
    Check if the board is full.
    
    Args:
        board: The game board
    
    Returns:
        True if the board is full, False otherwise
    """
    for row in range(3):
        for col in range(3):
            if board[row][col] == '':
                return False
    return True