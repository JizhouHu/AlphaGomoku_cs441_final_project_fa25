import numpy as np
import config

class GomokuLogicAgent:
    """
    A pure-logic heuristic agent for Gomoku.
    This does not use PyTorch and acts as a traditional rule-based AI.
    """
    def __init__(self):
        self.size = config.BOARD_SIZE
        self.directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        # Heuristic scores for patterns
        self.threat_scores = {
            'WIN': 1000000,   # 5-in-a-row
            'OPEN_4': 50000,  # _XXXX_
            'CLOSED_4': 5000, # OXXXX_
            'OPEN_3': 2500,   # _XXX_
            'CLOSED_3': 500,  # OXXX_
            'OPEN_2': 100,    # _XX_
            'CLOSED_2': 10,   # OXX_
        }

    def get_move(self, board, ai_color):
        """
        Calculates the best move by scoring every empty spot.
        board: 2D numpy array (1, -1, or 0)
        ai_color: the color the AI is playing (1 or -1)
        """
        best_move = None
        max_score = -float('inf')
        
        # Get all empty positions
        empty_cells = list(zip(*np.where(board == 0)))
        
        if not empty_cells:
            return None

        # Optimization: Only check cells near existing stones
        possible_moves = self._filter_active_area(board, empty_cells)

        for r, c in possible_moves:
            # 1. Offensive Score: How good is this for the AI?
            ai_score = self._calculate_threats_at(board, r, c, ai_color)
            
            # 2. Defensive Score: How much does the AI need to block the human?
            opponent_color = -ai_color
            opp_score = self._calculate_threats_at(board, r, c, opponent_color)

            # Combined score: Defensive blocking is weighted slightly higher (1.1x)
            # to prevent the human from completing a 4-in-a-row.
            total_score = ai_score + (opp_score * 1.1)

            if total_score > max_score:
                max_score = total_score
                best_move = (r, c)

        return best_move

    def _filter_active_area(self, board, empty_cells):
        """Returns empty cells that are within 1-2 steps of an existing stone."""
        if np.all(board == 0):
            return [(self.size // 2, self.size // 2)]
        
        active_moves = []
        for r, c in empty_cells:
            found_neighbor = False
            # Check a 3x3 or 5x5 area around the spot
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if board[nr][nc] != 0:
                            active_moves.append((r, c))
                            found_neighbor = True
                            break
                if found_neighbor: break
        return active_moves if active_moves else empty_cells

    def _calculate_threats_at(self, board, r, c, color):
        """Scores a single point based on the lines it creates."""
        total_score = 0
        
        for dr, dc in self.directions:
            count = 1
            open_ends = 0
            
            # Check positive direction
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if not (0 <= nr < self.size and 0 <= nc < self.size): break
                if board[nr][nc] == color: 
                    count += 1
                elif board[nr][nc] == 0:
                    open_ends += 1
                    break
                else: break # Blocked by opponent

            # Check negative direction
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if not (0 <= nr < self.size and 0 <= nc < self.size): break
                if board[nr][nc] == color: 
                    count += 1
                elif board[nr][nc] == 0:
                    open_ends += 1
                    break
                else: break

            # Map counts and open ends to heuristic scores
            if count >= 5: total_score += self.threat_scores['WIN']
            elif count == 4:
                total_score += self.threat_scores['OPEN_4'] if open_ends == 2 else self.threat_scores['CLOSED_4']
            elif count == 3:
                total_score += self.threat_scores['OPEN_3'] if open_ends == 2 else self.threat_scores['CLOSED_3']
            elif count == 2:
                total_score += self.threat_scores['OPEN_2'] if open_ends == 2 else self.threat_scores['CLOSED_2']
                
        return total_score