import torch
import numpy as np
import time
import config
from model import GomokuResNet
from logic_agent import GomokuLogicAgent
from game import GomokuGame

# --- AI Wrapper Class ---
class VisualAI:
    def __init__(self, model_class, model_path, name, is_logic=False):
        self.device = config.DEVICE
        self.name = name
        self.is_logic = is_logic
        
        if is_logic:
            self.model = GomokuLogicAgent()
            print(f"[{name}] Initialized (Pure Logic).")
        else:
            self.model = model_class().to(self.device)
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"[{name}] Loaded from {model_path}")
            except Exception as e:
                print(f"ERROR: Could not load {name}. Error: {e}")
                exit(1)

    def get_move(self, board, ai_color):
        # Logic Agent Path
        if self.is_logic:
            return self.model.get_move(board, ai_color)
        
        # ResNet Path (Needs 2-Channel Input fix)
        # Create 2 channels: [AI_stones, Opponent_stones]
        ai_stones = (board == ai_color).astype(np.float32)
        opp_stones = (board == -ai_color).astype(np.float32)
        
        input_board = np.stack([ai_stones, opp_stones]) # Shape (2, 15, 15)
        input_tensor = torch.tensor(input_board, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs = probs.cpu().numpy()[0]

        # Mask occupied spots
        flat_board = board.flatten()
        probs[flat_board != 0] = -1 
        
        best_move = np.argmax(probs)
        return best_move // config.BOARD_SIZE, best_move % config.BOARD_SIZE

# --- Game Runner ---
def play_visual_game(black_player, white_player):
    game = GomokuGame()
    print(f"\n=== NEW GAME: {black_player.name} (Black/X) vs {white_player.name} (White/O) ===")
    game.print_board()
    time.sleep(1)

    while not game.game_over:
        # Determine current player
        if game.current_player == 1:
            player = black_player
            symbol = "X"
            color_name = "Black"
        else:
            player = white_player
            symbol = "O"
            color_name = "White"

        # Get Move
        print(f"{player.name} ({color_name}) is thinking...")

        # time.sleep(5) # Uncomment to slow down the replay

        r, c = player.get_move(game.board, game.current_player)

        # Make Move
        if r is None: # Logic agent might return None if board is full
             print("No moves left!")
             break

        if not game.make_move(r, c):
            print(f"CRITICAL ERROR: {player.name} tried occupied spot ({r},{c}). Random fallback.")
            # Fallback to random empty spot
            empty = np.argwhere(game.board == 0)
            if len(empty) > 0:
                r, c = empty[np.random.randint(len(empty))]
                game.make_move(r, c)
            else:
                break

        # Show Board
        print(f"Move: {r+1}, {c+1}")
        game.print_board()

        # Check Result
        if game.check_win(game.current_player):
            print(f"WINNER: {player.name} ({color_name})!")
            return
        
        if game.is_full():
            print("DRAW!")
            return

        game.switch_turn()

def main():
    print("Initializing Agents...")
    
    # SETUP: Ensure path matches your saved model
    resnet_path = config.RESNET_PATH 
    
    resnet = VisualAI(GomokuResNet, resnet_path, "ResNet_AI")
    logic = VisualAI(None, None, "Logic_Agent", is_logic=True)

    # GAME 1: Logic starts (Black)
    print("\n" + "="*40)
    print("GAME 1: Logic Agent goes first")
    print("="*40)
    play_visual_game(black_player=logic, white_player=resnet)

    # GAME 2: ResNet starts (Black)
    print("\n" + "="*40)
    print("GAME 2: ResNet goes first")
    print("="*40)
    play_visual_game(black_player=resnet, white_player=logic)

if __name__ == "__main__":
    main()