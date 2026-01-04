import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import config
from model import GomokuLogisticRegression
import utils

def main():
    print(f"Device: {config.DEVICE}")
    
    output_filename = "logreg_output.txt"
    
    # 1. Load Data
    X_train_np, y_train_np, X_val_np, y_val_np = utils.process_data()
    print(f"Train Samples: {len(X_train_np)} | Val Samples: {len(X_val_np)}")

    X_train = torch.tensor(X_train_np).float().to(config.DEVICE)
    y_train = torch.tensor(y_train_np).long().to(config.DEVICE)
    X_val = torch.tensor(X_val_np).float().to(config.DEVICE)
    y_val = torch.tensor(y_val_np).long().to(config.DEVICE)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. Init Model
    model = GomokuLogisticRegression().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Initialize log file (Clear previous content)
    with open(output_filename, "w") as f:
        f.write(f"Training Log - Device: {config.DEVICE}\n")
        f.write("-" * 60 + "\n")

    # 3. Train Loop
    print("\nStarting Training...")
    
    # Wrap the range with tqdm for the progress bar
    pbar = tqdm(range(config.EPOCHS), unit="epoch")
    
    # Store the last epoch result to print
    final_log_msg = "" 

    for epoch in pbar:
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                logits = model(inputs)
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate stats
        avg_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total

        # Prepare the full string for the file
        log_msg = (f"Epoch {epoch+1}/{config.EPOCHS} | "
                   f"Loss: {avg_loss:.4f} | "
                   f"Train Acc: {train_acc:.2f}% | "
                   f"Val Acc: {val_acc:.2f}%")
        
        # store the last epoch result
        final_log_msg = log_msg
        
        # File Output
        with open(output_filename, "a") as f:
            f.write(log_msg + "\n")

        # Progress Bar
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Val Acc": f"{val_acc:.2f}%"})

    pbar.close() 
    print("\nFinal Result:")
    print(final_log_msg)

    # 4. Save model
    torch.save(model.state_dict(), config.LOGREG_PATH)
    print(f"Model saved as '{config.LOGREG_PATH}'")

if __name__ == "__main__":
    main()