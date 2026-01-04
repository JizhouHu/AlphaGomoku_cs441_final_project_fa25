import torch
import numpy as np
import config
from model import GomokuMLP, GomokuCNN, GomokuResNet
from game import GomokuGame
from tqdm import tqdm

# Define output filename
OUTPUT_FILE = "EvE_resnet_cnn_output.txt"

class SimAI:
    def __init__(self, model_class, model_path, name):
        self.device = config.DEVICE
        self.name = name
        
        # Initialize architecture
        self.model = model_class().to(self.device)
        
        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            print(f"ERROR: Could not load {name} from {model_path}. Error: {e}")
            exit(1)

    def get_move(self, board, ai_color):
        # 1. Create two separate 15x15 grids (Channels)
        # Channel 0: "My" stones (where board matches ai_color)
        # Channel 1: "Opponent" stones (where board matches -ai_color)
        
        mask_self = (board == ai_color).astype(np.float32)
        mask_opp  = (board == -ai_color).astype(np.float32)
        
        # 2. Stack them to create shape [2, 15, 15]
        input_stack = np.stack((mask_self, mask_opp))
        
        # 3. Add Batch Dimension -> [1, 2, 15, 15]
        input_tensor = torch.tensor(input_stack).unsqueeze(0).to(self.device)
        
        # 4. Predict
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs = probs.cpu().numpy()[0]

        # 5. Mask occupied spots (set prob to -1 so they are never picked)
        flat_board = board.flatten()
        probs[flat_board != 0] = -1 

        # 6. Select best move
        best_move = np.argmax(probs)
        return best_move // config.BOARD_SIZE, best_move % config.BOARD_SIZE

def play_one_game(player_black, player_white):
    """
    Returns:
         1 if Black wins
        -1 if White wins
         0 if Draw
    """
    game = GomokuGame()
    
    while not game.game_over:
        # Determine who is moving
        if game.current_player == 1:
            player = player_black
            color = 1
        else:
            player = player_white
            color = -1
            
        # Get move
        if np.sum(game.board == 0) == config.BOARD_SIZE**2:
            
            empty = np.argwhere(game.board == 0)
            r, c = empty[np.random.randint(len(empty))]
        else:
            r, c = player.get_move(game.board, color)
        
        # Make move (Warning: If AI is broken and picks occupied spot, this loops forever.
        # Ideally, SimAI logic prevents this via masking.)
        if not game.make_move(r, c):
            # Fallback for safety: pick first empty spot
            empty_spots = np.argwhere(game.board == 0)
            if len(empty_spots) > 0:
                r, c = empty_spots[0]
                game.make_move(r, c)
            else:
                return 0 # Draw

        # Check Win
        if game.check_win(game.current_player):
            return game.current_player # 1 or -1
            
        # Check Draw
        if game.is_full():
            return 0
            
        game.switch_turn()
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
    cnn_agent = SimAI(GomokuCNN, config.CNN_PATH, "CNN")
    mlp_agent = SimAI(GomokuResNet, config.RESNET_PATH, "RESNET")

    with open(OUTPUT_FILE, "w") as f:
        f.write("=== GOMOKU AI BATTLE RESULTS ===\n")
        f.write(f"Board Size: {config.BOARD_SIZE}x{config.BOARD_SIZE}\n\n")

    print("\nStarting Simulation...")

    # --- ROUND 1: LogReg (Black) vs MLP (White) ---
    p1_wins, p2_wins, draws = run_matchup(cnn_agent, "Black", mlp_agent, "White", 100)
    
    r1_log = (f"ROUND 1: CNN (Black) vs RESNET (White) | "
              f"CNN Wins: {p1_wins} | RESNET Wins: {p2_wins} | Draws: {draws}")
    print(r1_log)
    
    with open(OUTPUT_FILE, "a") as f:
        f.write(r1_log + "\n")

    # --- ROUND 2: MLP (Black) vs LogReg (White) ---
    # Note: run_matchup always treats first arg as Black
    p1_wins_r2, p2_wins_r2, draws_r2 = run_matchup(mlp_agent, "Black", cnn_agent, "White", 100)
    
    r2_log = (f"ROUND 2: RESNET (Black) vs CNN (White) | "
              f"RESNET Wins: {p1_wins_r2} | CNN Wins: {p2_wins_r2} | Draws: {draws_r2}")
    print(r2_log)
    
    with open(OUTPUT_FILE, "a") as f:
        f.write(r2_log + "\n")
        
        
    # --- FINAL CALCULATIONS ---
    total_logreg_wins = p1_wins + p2_wins_r2
    total_resnet_wins = p2_wins + p1_wins_r2
    total_draws = draws + draws_r2
    total_games = 200
    
    summary = (
        f"\n=== FINAL SUMMARY (200 Games) ===\n"
        f"CNN Total Wins: {total_logreg_wins} ({total_logreg_wins/total_games*100:.1f}%)\n"
        f"RESNET Total Wins:    {total_resnet_wins} ({total_resnet_wins/total_games*100:.1f}%)\n"
        f"Draws:             {total_draws}\n"
    )
    
    print(summary)
    with open(OUTPUT_FILE, "a") as f:
        f.write(summary)
        
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()