import streamlit as st # type: ignore
import numpy as np # type: ignore
from ai import get_best_move # type: ignore
from utils import check_winner, is_board_full, get_empty_board # type: ignore

# Set page config
st.set_page_config(
    page_title="Tic-Tac-Toe AI",
    page_icon="🎮",
)

# Initialize session state
if 'board' not in st.session_state:
    st.session_state.board = get_empty_board()
    
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
    
if 'winner' not in st.session_state:
    st.session_state.winner = None
    
if 'player_turn' not in st.session_state:
    st.session_state.player_turn = True  # True for player (X), False for AI (O)
    
if 'player_score' not in st.session_state:
    st.session_state.player_score = 0
    
if 'ai_score' not in st.session_state:
    st.session_state.ai_score = 0
    
if 'draws' not in st.session_state:
    st.session_state.draws = 0

# Function to handle player move
def handle_click(row, col):
    # If the cell is already filled or the game is over, do nothing
    if st.session_state.board[row][col] != '' or st.session_state.game_over:
        return
    
    # Update board with player's move
    st.session_state.board[row][col] = 'X'
    
    # Check if player won
    if check_winner(st.session_state.board, 'X'):
        st.session_state.game_over = True
        st.session_state.winner = 'X'
        st.session_state.player_score += 1
        return
    
    # Check if board is full (draw)
    if is_board_full(st.session_state.board):
        st.session_state.game_over = True
        st.session_state.winner = 'Draw'
        st.session_state.draws += 1
        return
    
    # AI's turn
    st.session_state.player_turn = False
    
    # Get AI move
    ai_row, ai_col = get_best_move(st.session_state.board)
    st.session_state.board[ai_row][ai_col] = 'O'
    
    # Check if AI won
    if check_winner(st.session_state.board, 'O'):
        st.session_state.game_over = True
        st.session_state.winner = 'O'
        st.session_state.ai_score += 1
        return
    
    # Check if board is full (draw)
    if is_board_full(st.session_state.board):
        st.session_state.game_over = True
        st.session_state.winner = 'Draw'
        st.session_state.draws += 1
        return
    
    # Back to player's turn
    st.session_state.player_turn = True

# Function to reset the game
def reset_game():
    st.session_state.board = get_empty_board()
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.player_turn = True

# Main UI
st.title("🎮 Tic-Tac-Toe with AI")

# Display scoreboard
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Player (X)", st.session_state.player_score)
with col2:
    st.metric("Draws", st.session_state.draws)
with col3:
    st.metric("AI (O)", st.session_state.ai_score)

# Display game status
if st.session_state.game_over:
    if st.session_state.winner == 'X':
        st.success("🎉 You won! Congratulations!")
    elif st.session_state.winner == 'O':
        st.error("❌ AI won! Better luck next time.")
    else:
        st.info("🤝 It's a draw!")
else:
    if st.session_state.player_turn:
        st.info("Your turn (X). Click on an empty cell.")
    else:
        st.warning("AI is thinking...")
        # This is just for display - AI actually moves instantly

# Game board UI
st.write("## Game Board")
board = st.session_state.board

# Create a grid of buttons for the game board
for i in range(3):
    cols = st.columns([1, 1, 1])
    for j in range(3):
        with cols[j]:
            # Define button style and text
            if board[i][j] == 'X':
                button_text = '❌'
                button_color = 'red'
            elif board[i][j] == 'O':
                button_text = '⭕'
                button_color = 'blue'
            else:
                button_text = ' '
                button_color = 'gray'
            
            # Create button with appropriate styling
            if st.button(button_text, key=f"cell_{i}_{j}", 
                        use_container_width=True,
                        help=f"Row {i+1}, Column {j+1}"):
                handle_click(i, j)
                st.rerun()

# Game controls
st.write("## Game Controls")
if st.button("Restart Game", use_container_width=True):
    reset_game()
    st.rerun()

# Game instructions
with st.expander("How to Play"):
    st.write("""
    1. You are X, the AI is O.
    2. Players take turns placing their marks on the board.
    3. The first player to get 3 of their marks in a row (horizontally, vertically, or diagonally) wins.
    4. If all 9 squares are filled and no player has 3 marks in a row, the game is a draw.
    
    The AI uses the Minimax algorithm to make optimal moves, making it essentially unbeatable. At best, you can hope for a draw!
    """)

# About the AI
with st.expander("About the AI"):
    st.write("""
    This game uses the Minimax algorithm, a decision-making algorithm commonly used in two-player games.
    
    How it works:
    - The AI evaluates all possible future moves and their outcomes.
    - It assumes you'll make the best possible move at each step.
    - The algorithm then chooses the move that maximizes its chances of winning.
    
    This makes the AI unbeatable - the best you can achieve against it is a draw!
    """)




