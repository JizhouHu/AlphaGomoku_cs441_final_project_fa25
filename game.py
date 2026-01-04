import numpy as np
import config

class GomokuGame:
    def __init__(self):
        self.board = np.zeros((config.BOARD_SIZE, config.BOARD_SIZE), dtype=int)
        self.current_player = 1 # 1 = Black, -1 = White
        self.game_over = False

    def print_board(self):
        # 1. Print Column Headers
        header_str = "   " 
        for i in range(1, config.BOARD_SIZE + 1):
            header_str += f"{i:>3}" 
        print("\n" + header_str)
        
        # 2. Print Divider
        print("   " + "-" * (config.BOARD_SIZE * 3))
        
        # 3. Print Rows
        for r in range(config.BOARD_SIZE):
            row_str = f"{r + 1:>2}|"
            for c in range(config.BOARD_SIZE):
                val = self.board[r, c]
                if val == 1:
                    token = " X "
                elif val == -1:
                    token = " O "
                else:
                    token = " . "
                row_str += token
            print(row_str)
        print("\n")

    def check_win(self, player):
        b = self.board
        BS = config.BOARD_SIZE
        for r in range(BS):
            for c in range(BS):
                if b[r, c] != player: continue
                
                if c + 4 < BS and all(b[r, c+k] == player for k in range(5)): return True
                if r + 4 < BS and all(b[r+k, c] == player for k in range(5)): return True
                if r + 4 < BS and c + 4 < BS and all(b[r+k, c+k] == player for k in range(5)): return True
                if r - 4 >= 0 and c + 4 < BS and all(b[r-k, c+k] == player for k in range(5)): return True
        return False

    def is_full(self):
        return np.all(self.board != 0)

    def make_move(self, row_idx, col_idx):
        if self.board[row_idx, col_idx] != 0:
            return False
        self.board[row_idx, col_idx] = self.current_player
        return True

    def switch_turn(self):
        self.current_player *= -1