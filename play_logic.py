import numpy as np
import config
from logic_agent import GomokuLogicAgent 
from game import GomokuGame

class AIPlayer:
    def __init__(self): 
        # No device or model_path needed for pure logic
        self.agent = GomokuLogicAgent()

    def get_move(self, board, ai_color):
        """
        board: 2D numpy array from GomokuGame
        ai_color: 1 (Black) or -1 (White)
        """
        # The logic agent returns (row, col) directly
        move = self.agent.get_move(board, ai_color)
        return move

def main():
    print("Welcome to Gomoku (15x15) - Logic Engine Edition")
    ai = AIPlayer() 

    while True:
        choice = input("Choose your color (B/W): ").lower()
        if choice in ['b', 'w']: break
    
    human_color = 1 if choice == 'b' else -1
    ai_color = -human_color
    
    game = GomokuGame()
    print(f"\nYou are {'Black (X)' if human_color == 1 else 'White (O)'}")
    game.print_board()

    while not game.game_over:
        if game.current_player == human_color:
            print(">> Your Move (row col):")
            try:
                u_in = input("Input (1-15): ").strip().replace(',', ' ').split()
                if len(u_in) != 2: continue
                r, c = int(u_in[0]), int(u_in[1])
                if not (1 <= r <= config.BOARD_SIZE and 1 <= c <= config.BOARD_SIZE):
                    print(f"Must be 1-{config.BOARD_SIZE}")
                    continue
                if not game.make_move(r-1, c-1):
                    print("Occupied!")
                    continue
            except ValueError: continue
        else:
            print(">> AI is thinking (Logic Heuristics)...")
            move = ai.get_move(game.board, ai_color)
            if move:
                r, c = move
                game.make_move(r, c)
                print(f"AI chose: {r+1}, {c+1}")
            else:
                print("AI passes (No moves left).")

        game.print_board()

        if game.check_win(game.current_player):
            print(f"{'Black' if game.current_player == 1 else 'White'} Wins!")
            break
        if game.is_full():
            print("Draw!")
            break
        game.switch_turn()

if __name__ == "__main__":
    main()