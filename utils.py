import os
import zipfile
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import config
from tqdm import tqdm
import random

# Verifies that the data directory exists
def setup_data():
    if os.path.exists(config.DATA_DIR):
        print(f"Data directory found: '{config.DATA_DIR}'")
        
        # Check data dir empty
        if not os.listdir(config.DATA_DIR):
             print(f"WARNING: '{config.DATA_DIR}' exists but appears to be empty!")
    else:
        print(f"ERROR: Data directory '{config.DATA_DIR}' not found!")
        print(f"Please make sure you uploaded your unzipped folder and named it '{config.DATA_DIR}'.")

# Parses lines from a .psq file
def parse_psq_content(lines):
    moves = []
    for line in lines:
        line = line.strip()
        if not line: continue

        parts = line.split(',')
        if len(parts) >= 2:
            # Filter headers/footers
            if parts[0].strip().lstrip('-').isdigit() and parts[1].strip().lstrip('-').isdigit():
                try:
                    x = int(parts[0]) - 1 # Convert 1-based to 0-based
                    y = int(parts[1]) - 1
                    if 0 <= x < config.BOARD_SIZE and 0 <= y < config.BOARD_SIZE:
                        moves.append((x, y))
                except ValueError:
                    continue
    return moves

def process_data():
    setup_data()
    
    # Find all files recursively
    files = glob.glob(os.path.join(config.DATA_DIR, "**", "*.psq"), recursive=True)
    print(f"Found {len(files)} files.")

    if len(files) < 10:
        print("WARNING: Very few files found. Check your zip file content.")

    # Split train/val set
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    # Added 'desc_text' parameter for the progress bar label
    def process_file_list(file_list, desc_text):
        X_temp = []
        y_temp = []
        
        # Wrap the file_list with tqdm to show a progress bar
        for filepath in tqdm(file_list, desc=desc_text, unit="file"):
            with open(filepath, 'r', errors='replace') as f:
                moves = parse_psq_content(f.readlines())
            
            if not moves: continue

            board = np.zeros((config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.float32)
            current_player = 1 
            
            for x, y in moves:
                # Canonical form: AI is always 1
                state = (board * current_player).flatten()
                
                X_temp.append(state)
                y_temp.append(x * config.BOARD_SIZE + y)
                
                board[x, y] = current_player
                current_player *= -1
        return np.array(X_temp), np.array(y_temp)

    X_train, y_train = process_file_list(train_files, desc_text="Processing Train")
    X_val, y_val = process_file_list(val_files, desc_text="Processing Val")
    
    return X_train, y_train, X_val, y_val

def process_data_augmented():
    setup_data()
    
    files = glob.glob(os.path.join(config.DATA_DIR, "**", "*.psq"), recursive=True)
    print(f"Found {len(files)} files.")

    if len(files) < 10:
        print("WARNING: Very few files found.")

    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    # Define augmentation helper inside
    def get_augments(board, target_move):
        """Returns list of (board_state, target_index) tuples"""
        x, y = target_move
        data_points = []
        
        # Helper functions
        transforms = [
            (lambda b: b, lambda r, c: (r, c)), # Identity
            (lambda b: np.rot90(b, 1), lambda r, c: (14-c, r)), # Rot90
            (lambda b: np.rot90(b, 2), lambda r, c: (14-r, 14-c)), # Rot180
            (lambda b: np.rot90(b, 3), lambda r, c: (c, 14-r)), # Rot270
            (lambda b: np.flipud(b),   lambda r, c: (14-r, c)), # Flip Horizontal
            (lambda b: np.fliplr(b),   lambda r, c: (r, 14-c)), # Flip Vertical
            (lambda b: np.transpose(b), lambda r, c: (c, r)),   # Transpose (Main Diag)
            (lambda b: np.rot90(np.transpose(b), 2), lambda r, c: (14-c, 14-r)) # Anti-Diag
        ]

        for t_func, c_func in transforms:
            # Transform Board
            aug_b = t_func(board)
            
            # Transform Move Coordinate
            tx, ty = c_func(x, y)
            t_idx = tx * config.BOARD_SIZE + ty
            
            layer_self = (aug_b == 1).astype(np.float32)
            layer_opp  = (aug_b == -1).astype(np.float32)
            
            stacked = np.stack([layer_self, layer_opp], axis=0) # Shape: [2, 15, 15]
            data_points.append((stacked, t_idx))
            
        return data_points

    def process_file_list(file_list, desc_text):
        X_temp = []
        y_temp = []
        
        for filepath in tqdm(file_list, desc=desc_text, unit="file"):
            with open(filepath, 'r', errors='replace') as f:
                moves = parse_psq_content(f.readlines())
            
            if not moves: continue

            board = np.zeros((config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.float32)
            current_player = 1 
            
            for x, y in moves:
                aug_data = get_augments(board * current_player, (x, y))
                
                for input_tensor, target_idx in aug_data:
                    X_temp.append(input_tensor)
                    y_temp.append(target_idx)
                
                board[x, y] = current_player
                current_player *= -1
                
        return np.array(X_temp, dtype=np.float32), np.array(y_temp, dtype=np.int64)

    X_train, y_train = process_file_list(train_files, desc_text="Processing Train")
    X_val, y_val = process_file_list(val_files, desc_text="Processing Val")
    
    return X_train, y_train, X_val, y_val