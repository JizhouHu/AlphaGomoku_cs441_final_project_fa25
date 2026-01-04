import torch
import numpy as np
import config
from model import GomokuResNet # Changed to ResNet
from logic_agent import GomokuLogicAgent # Import your logic agent
from game import GomokuGame
from tqdm import tqdm

# Define output filename
OUTPUT_FILE = "EvE_resnet_logic_output.txt"

class SimAI:
    def __init__(self, model_class, model_path, name, is_logic=False):
        self.device = config.DEVICE
        self.name = name
        self.is_logic = is_logic
        
        if is_logic:
            self.model = GomokuLogicAgent() # Pure logic agent
        else:
            self.model = model_class().to(self.device)
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
            except Exception as e:
                print(f"ERROR: Could not load {name} from {model_path}. Error: {e}")
                exit(1)

    def get_move(self, board, ai_color):
            if self.is_logic:
                # Logic agent takes raw numpy board and returns (r, c)
                return self.model.get_move(board, ai_color)
            
            # --- FIX STARTS HERE ---
            # Create 2 channels: [AI_stones, Opponent_stones]
            ai_stones = (board == ai_color).astype(np.float32)
            opp_stones = (board == -ai_color).astype(np.float32)
            
            # Stack into shape (2, 15, 15)
            input_board = np.stack([ai_stones, opp_stones]) 
            
            # Tensorize: result shape is (1, 2, 15, 15)
            input_tensor = torch.tensor(input_board, dtype=torch.float32).unsqueeze(0).to(self.device)
            # --- FIX ENDS HERE ---
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                probs = probs.cpu().numpy()[0]

            flat_board = board.flatten()
            probs[flat_board != 0] = -1 
            best_move = np.argmax(probs)
            return best_move // config.BOARD_SIZE, best_move % config.BOARD_SIZE

def play_one_game(player_black, player_white):
    game = GomokuGame()
    while not game.game_over:
        player = player_black if game.current_player == 1 else player_white
        color = game.current_player
            
        if np.sum(game.board == 0) == config.BOARD_SIZE**2:
            empty = np.argwhere(game.board == 0)
            r, c = empty[np.random.randint(len(empty))]
        else:
            r, c = player.get_move(game.board, color)
        
        if not game.make_move(r, c):
            empty_spots = np.argwhere(game.board == 0)
            if len(empty_spots) > 0:
                r, c = empty_spots[0]
                game.make_move(r, c)
            else:
                return 0 

        if game.check_win(game.current_player):
            return game.current_player 
            
        if game.is_full():
            return 0
            
        game.switch_turn()
    return 0

def run_matchup(p1, p1_color_name, p2, p2_color_name, num_games):
    p1_wins, p2_wins, draws = 0, 0, 0
    desc = f"{p1.name} ({p1_color_name}) vs {p2.name} ({p2_color_name})"
    for _ in tqdm(range(num_games), desc=desc):
        result = play_one_game(player_black=p1, player_white=p2)
        if result == 1: p1_wins += 1
        elif result == -1: p2_wins += 1
        else: draws += 1
    return p1_wins, p2_wins, draws

def main():
    print(f"Loading Models on {config.DEVICE}...")
    
    # Initialize ResNet and Logic Agent
    resnet_agent = SimAI(GomokuResNet, config.RESNET_PATH, "ResNet")
    logic_agent = SimAI(None, None, "LogicAgent", is_logic=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write("=== GOMOKU AI BATTLE RESULTS ===\n")
        f.write(f"Board Size: {config.BOARD_SIZE}x{config.BOARD_SIZE}\n\n")

    # --- ROUND 1: ResNet (Black) vs Logic (White) ---
    p1_wins, p2_wins, draws = run_matchup(resnet_agent, "Black", logic_agent, "White", 100)
    r1_log = f"ROUND 1: ResNet (Black) vs Logic (White) | ResNet Wins: {p1_wins} | Logic Wins: {p2_wins} | Draws: {draws}"
    print(r1_log)
    
    # --- ROUND 2: Logic (Black) vs ResNet (White) ---
    p1_wins_r2, p2_wins_r2, draws_r2 = run_matchup(logic_agent, "Black", resnet_agent, "White", 100)
    r2_log = f"ROUND 2: Logic (Black) vs ResNet (White) | Logic Wins: {p1_wins_r2} | ResNet Wins: {p2_wins_r2} | Draws: {draws_r2}"
    print(r2_log)
    
    total_resnet_wins = p1_wins + p2_wins_r2
    total_logic_wins = p2_wins + p1_wins_r2
    total_draws = draws + draws_r2
    
    summary = (
        f"\n=== FINAL SUMMARY (200 Games) ===\n"
        f"ResNet Total Wins: {total_resnet_wins} ({total_resnet_wins/200*100:.1f}%)\n"
        f"Logic Total Wins:  {total_logic_wins} ({total_logic_wins/200*100:.1f}%)\n"
        f"Draws:             {total_draws}\n"
    )
    
    print(summary)
    with open(OUTPUT_FILE, "a") as f:
        f.write(r1_log + "\n" + r2_log + "\n" + summary)

if __name__ == "__main__":
    main()