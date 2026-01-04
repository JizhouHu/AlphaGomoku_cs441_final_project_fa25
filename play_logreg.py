import torch
import numpy as np
import config
from model import GomokuLogisticRegression
from game import GomokuGame

class AIPlayer:
    def __init__(self, model_path=config.LOGREG_PATH):
        self.device = config.DEVICE
        self.model = GomokuLogisticRegression().to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"AI Model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model ({e}). AI will play randomly.")
        
        self.model.eval()

    def get_move(self, board, ai_color):
        input_board = board * ai_color 
        input_tensor = torch.tensor(input_board.flatten()).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs = probs.cpu().numpy()[0]

        flat_board = board.flatten()
        probs[flat_board != 0] = -1 # Mask taken spots

        best_move = np.argmax(probs)
        return best_move // config.BOARD_SIZE, best_move % config.BOARD_SIZE

def main():
    print("Welcome to Gomoku (15x15)")
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
            print(">> AI is thinking...")
            r, c = ai.get_move(game.board, ai_color)
            game.make_move(r, c)
            print(f"AI chose: {r+1}, {c+1}")

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