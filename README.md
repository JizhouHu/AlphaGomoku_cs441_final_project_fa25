# AlphaGomoku (CS441 Final Project)

This project implements various AI agents to play Gomoku (Five-in-a-Row) on a 15x15 board. The agents range from simple logic-based heuristics to deep learning models (CNN, ResNet).

## Project Structure

### Core Files
- `config.py`: Central configuration file containing hyperparameters (Batch Size, Learning Rate), file paths, and board settings.
- `game.py`: Implements the `GomokuGame` class, handling game rules, move validation, win detection, and board state.
- `model.py`: Defines the neural network architectures (`GomokuCNN`, `GomokuResNet`, `GomokuMLP`, `LogisticRegression`).
- `utils.py`: Utility functions for data processing, parsing `.psq` files (Renju dataset), and handling data augmentation.
- `logic_agent.py`: A pure heuristic-based agent that plays using traditional Gomoku strategies (blocking 3s and 4s) without neural networks.

### Training Scripts
These scripts are used to train the different AI models.
- `train_logreg.py`: Trains a Logistic Regression model.
- `train_mlp.py`: Trains a Multi-Layer Perceptron (MLP) model.
- `train_cnn.py`: Trains a Convolutional Neural Network (CNN).
- `train_resnet.py`: Trains a ResNet-based model.

### Evaluation Scripts (EvE - Environment vs Environment)
These scripts simulate matches between different AI agents to evaluate their performance.
- `EvE_mlp_logreg.py`: Simulates matches between MLP and Logistic Regression agents.
- `EvE_mlp_cnn.py`: Simulates matches between MLP and CNN agents.
- `EvE_cnn_resnet.py`: Simulates matches between CNN and ResNet agents.
- `EvE_logic_resnet.py`: Simulates matches between the Logic Agent and ResNet.

### Gameplay Scripts
These scripts allow humans to play against AI or watch AI vs AI matches.
- `play_logreg.py`: Play against the Logistic Regression agent.
- `play_mlp.py`: Play against the MLP agent.
- `play_cnn.py`: Play against the CNN agent.
- `play_resnet.py`: Play against the ResNet agent.
- `play_logic.py`: Play against the Logic Agent.
- `play_onboard_logic_resnet.py`: Watch a visual match between the Logic Agent and the ResNet Agent.

### Data and Models
- `game_data/`: Directory containing the raw game records (Renju dataset) used for training. More training data can be added to this directory.
- `*.pth`: Saved model weights (e.g., `gomoku_cnn.pth`, `gomoku_resnet.pth`).
- `*.txt`: Output logs from training and evaluation scripts (e.g., `cnn_output.txt`, `EvE_mlp_cnn_output.txt`).

### Data Source
The training data for this project was sourced from [Gomocup](https://gomocup.org/), specifically the results from the **2024 Freestyle 15x15** tournament.
- **Source URL**: [Gomocup 2024 Results](https://gomocup.org/results/gomocup-result-2024/)
- **Dataset**: Freestyle 15x15 games.

## Setup Instructions (Google Colab)

1.  **Mount Google Drive**:
    Run the following code in the first cell of your notebook to mount your Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2.  **Navigate to Project Directory**:
    Change the directory to where your project files are located. **Update the path below** to match your actual folder structure:
    ```python
    %cd /content/drive/MyDrive/CS441/Final_Project
    ```

## Usage Guide

### 1. Train the Models
Run the training scripts in order or as needed to generate the model weights (`.pth` files).

```python
!python train_logreg.py
!python train_mlp.py
!python train_cnn.py
!python train_resnet.py
```

### 2. Evaluate Models (AI vs AI)
Run these scripts to see how different models perform against each other. The results will be printed and saved to text files.

```python
!python EvE_mlp_cnn.py
!python EvE_cnn_resnet.py
!python EvE_logic_resnet.py
```

### 3. Play the Game
To play a game against the CNN agent (or others by changing the script name):
```python
!python play_cnn.py
```
*Follow the on-screen prompts to choose your color (Black/White) and enter your moves.*

### 4. Watch AI vs AI
To watch a visual demonstration of the Logic Agent playing against the ResNet Agent:
```python
! python play_onboard_logic_resnet.py
```

## Requirements
- Python 3.x
- PyTorch
- NumPy
- tqdm
