def get_empty_board():
    """
    Create and return an empty 3x3 Tic-Tac-Toe board.
    
    Returns:
        A 3x3 list representing an empty board
    """
    return [['', '', ''] for _ in range(3)]

def check_winner(board, player):
    """
    Check if the specified player has won the game.
    
    Args:
        board: The game board
        player: The player to check for ('X' or 'O')
    
    Returns:
        True if the player has won, False otherwise
    """
    # Check rows
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] == player:
            return True
    
    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] == player:
            return True
    
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] == player:
        return True
    if board[0][2] == board[1][1] == board[2][0] == player:
        return True
    
    # No win found
    return False

def is_board_full(board):
    """
    Check if the board is full (draw condition).
    
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

def print_board(board):
    """
    Print the board to the console (for debugging purposes).
    
    Args:
        board: The game board
    """
    for row in board:
        print(' | '.join([cell if cell != '' else ' ' for cell in row]))
        print('-' * 9)