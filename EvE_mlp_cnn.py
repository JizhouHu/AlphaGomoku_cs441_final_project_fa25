import torch
import numpy as np
import config
from model import GomokuMLP, GomokuCNN
from game import GomokuGame
from tqdm import tqdm

# Define output filename
OUTPUT_FILE = "EvE_mlp_cnn_output.txt"

class SimAI:
    def __init__(self, model_class, model_path, name, is_cnn=True):
        self.device = config.DEVICE
        self.name = name
        self.is_cnn = is_cnn  # Add a flag to distinguish input types
        
        self.model = model_class().to(self.device)
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Loaded {name} successfully.")
        except Exception as e:
            print(f"ERROR: Could not load {name} from {model_path}. Error: {e}")
            exit(1)

    # def get_move(self, board, ai_color):
    #     # 1. Prepare Input based on model type
    #     if self.is_cnn:
    #         # --- CNN: Expects (Batch, 2, 15, 15) ---
    #         my_pieces = (board == ai_color).astype(np.float32)
    #         opp_pieces = (board == -ai_color).astype(np.float32)
    #         input_stack = np.stack([my_pieces, opp_pieces])
    #         input_tensor = torch.tensor(input_stack).unsqueeze(0).to(self.device)
    #     else:
    #         # --- MLP: Expects (Batch, 225) ---
    #         # NOTE: Assuming MLP takes raw board (1 for self, -1 for opp, 0 empty)
    #         # We orient the board so 'ai_color' is always positive (perspective)
    #         input_board = (board * ai_color).flatten().astype(np.float32)
    #         input_tensor = torch.tensor(input_board).unsqueeze(0).to(self.device)

    #     # 2. Inference
    #     with torch.no_grad():
    #         logits = self.model(input_tensor)
    #         probs = torch.nn.functional.softmax(logits, dim=1)
    #         probs = probs.cpu().numpy()[0]

    #     # 3. Mask occupied spots
    #     flat_board = board.flatten()
    #     probs[flat_board != 0] = -1 

    #     best_move = np.argmax(probs)
    #     return best_move // config.BOARD_SIZE, best_move % config.BOARD_SIZE

    def get_move(self, board, ai_color):
        if self.is_cnn:
            # --- CNN: Use Absolute Coordinates (Black/White) ---
            # Fixes the "random play" issue by aligning with Renju/Gomoku datasets
            # Shape: (Batch, 2, 15, 15)
            black_pieces = (board == 1).astype(np.float32)
            white_pieces = (board == -1).astype(np.float32)
            
            # Stack Channel 0 = Black, Channel 1 = White
            input_stack = np.stack([black_pieces, white_pieces])
            input_tensor = torch.tensor(input_stack).unsqueeze(0).to(self.device)
            
        else:
            # --- MLP: Use Relative Coordinates (Self/Opponent) ---
            # Must be flattened to match the Linear layer input size (225)
            # Shape: (Batch, 225)
            input_board = (board * ai_color).flatten().astype(np.float32)
            input_tensor = torch.tensor(input_board).unsqueeze(0).to(self.device)

        # --- Shared Inference Logic ---
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs = probs.cpu().numpy()[0]
            
            # Optional: Print confidence only for CNN to debug
            if self.is_cnn:
                best_idx = np.argmax(probs)
                # print(f"CNN Confidence: {probs[best_idx]:.4f}")

        flat_board = board.flatten()
        probs[flat_board != 0] = -1 # Mask taken spots

        best_move = np.argmax(probs)
        return best_move // config.BOARD_SIZE, best_move % config.BOARD_SIZE

# def play_one_game(player_black, player_white):
#     """
#     Returns:
#          1 if Black wins
#         -1 if White wins
#          0 if Draw
#     """
#     game = GomokuGame()
    
#     while not game.game_over:
#         # Determine who is moving
#         if game.current_player == 1:
#             player = player_black
#             color = 1
#         else:
#             player = player_white
#             color = -1
            
#         # Get move
#         r, c = player.get_move(game.board, color)
        
#         # Make move (Warning: If AI is broken and picks occupied spot, this loops forever.
#         # Ideally, SimAI logic prevents this via masking.)
#         if not game.make_move(r, c):
#             # Fallback for safety: pick first empty spot
#             empty_spots = np.argwhere(game.board == 0)
#             if len(empty_spots) > 0:
#                 r, c = empty_spots[0]
#                 game.make_move(r, c)
#             else:
#                 return 0 # Draw

#         # Check Win
#         if game.check_win(game.current_player):
#             return game.current_player # 1 or -1
            
#         # Check Draw
#         if game.is_full():
#             return 0
            
#         game.switch_turn()
#     return 0

def play_one_game(player_black, player_white, verbose=False):
    game = GomokuGame()
    move_count = 0
    
    while not game.game_over:
        if game.current_player == 1:
            player = player_black
            color = 1
            p_name = "Black"
        else:
            player = player_white
            color = -1
            p_name = "White"
            
        r, c = player.get_move(game.board, color)
        
        # Debug printing for the first few moves
        if verbose and move_count < 6:
            print(f"Turn {move_count} ({p_name}): AI suggests ({r}, {c})")

        # Attempt move
        if not game.make_move(r, c):
            if verbose: 
                print(f"!!! INVALID MOVE by {p_name} at ({r}, {c}). Triggering Fallback.")
            
            # Fallback
            empty_spots = np.argwhere(game.board == 0)
            if len(empty_spots) > 0:
                r, c = empty_spots[0]
                game.make_move(r, c)
                if verbose: print(f" -> Fallback moved to ({r}, {c})")
            else:
                return 0 # Draw

        if game.check_win(game.current_player):
            return game.current_player
            
        if game.is_full():
            return 0
            
        game.switch_turn()
        move_count += 1
        
    return 0

def run_matchup(p1, p1_color_name, p2, p2_color_name, num_games):
    """
    Runs a batch of games and returns stats.
    p1 is always BLACK (1), p2 is always WHITE (-1) in this function scope.
    """
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    desc = f"{p1.name} ({p1_color_name}) vs {p2.name} ({p2_color_name})"
    
    for _ in tqdm(range(num_games), desc=desc):
        result = play_one_game(player_black=p1, player_white=p2)
        
        if result == 1:
            p1_wins += 1
        elif result == -1:
            p2_wins += 1
        else:
            draws += 1
            
    return p1_wins, p2_wins, draws

def main():
    print(f"Loading Models on {config.DEVICE}...")
    
    # 1. Initialize Agents
    # Ensure config.LOGREG_PATH and config.MLP_PATH are set in config.py
    cnn_agent = SimAI(GomokuCNN, config.CNN_PATH, "CNN", is_cnn=True)
    mlp_agent = SimAI(GomokuMLP, config.MLP_PATH, "MLP", is_cnn=False)
    with open(OUTPUT_FILE, "w") as f:
        f.write("=== GOMOKU AI BATTLE RESULTS ===\n")
        f.write(f"Board Size: {config.BOARD_SIZE}x{config.BOARD_SIZE}\n\n")

    print("\nStarting Simulation...")

    # --- ROUND 1: LogReg (Black) vs MLP (White) ---
    p1_wins, p2_wins, draws = run_matchup(cnn_agent, "Black", mlp_agent, "White", 100)
    
    r1_log = (f"ROUND 1: CNN (Black) vs MLP (White) | "
              f"CNN Wins: {p1_wins} | MLP Wins: {p2_wins} | Draws: {draws}")
    print(r1_log)
    
    with open(OUTPUT_FILE, "a") as f:
        f.write(r1_log + "\n")

    # --- ROUND 2: MLP (Black) vs LogReg (White) ---
    # Note: run_matchup always treats first arg as Black
    p1_wins_r2, p2_wins_r2, draws_r2 = run_matchup(mlp_agent, "Black", cnn_agent, "White", 100)
    
    r2_log = (f"ROUND 2: MLP (Black) vs CNN (White) | "
              f"MLP Wins: {p1_wins_r2} | CNN Wins: {p2_wins_r2} | Draws: {draws_r2}")
    print(r2_log)
    
    with open(OUTPUT_FILE, "a") as f:
        f.write(r2_log + "\n")
        
        
    # --- FINAL CALCULATIONS ---
    total_logreg_wins = p1_wins + p2_wins_r2
    total_mlp_wins = p2_wins + p1_wins_r2
    total_draws = draws + draws_r2
    total_games = 200
    
    summary = (
        f"\n=== FINAL SUMMARY (200 Games) ===\n"
        f"CNN Total Wins: {total_logreg_wins} ({total_logreg_wins/total_games*100:.1f}%)\n"
        f"MLP Total Wins:    {total_mlp_wins} ({total_mlp_wins/total_games*100:.1f}%)\n"
        f"Draws:             {total_draws}\n"
    )
    
    print(summary)
    with open(OUTPUT_FILE, "a") as f:
        f.write(summary)
        
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()