import torch
from tokenization import load_tokenizer, tokenize_sequences
from model import load_model, setup_optimizer
from train import train_model
from evaluation import evaluate_model
from hyperparameter_tuning import hyperparameter_grid
from torch.nn import MultiheadAttention as MHA
import pandas as pd
import torch.optim as optim
import wandb

# Initialize WandB project
wandb.init(project='Evo_Finetuning')

def main():
    model_name = 'togethercomputer/evo-1-131k-base'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the training and test data
    try:
        train_sequences = pd.read_csv('train_sequences.csv')['train_seq'].tolist()
        test_sequences = pd.read_csv('test_sequences.csv')['test_seq'].tolist()
        train_percentages = pd.read_csv('train_percentages.csv')['train_perc'].tolist()
        test_percentages = pd.read_csv('test_percentages.csv')['test_perc'].tolist()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e.filename}. Ensure that the CSV files are in the correct directory.")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

    # Check that the sequences and percentages are correctly loaded
    if not train_sequences or not train_percentages:
        raise ValueError("Training sequences or percentages are empty.")
    if not test_sequences or not test_percentages:
        raise ValueError("Test sequences or percentages are empty.")
    if len(train_sequences) != len(train_percentages):
        raise ValueError("Mismatch between the number of training sequences and percentages.")
    if len(test_sequences) != len(test_percentages):
        raise ValueError("Mismatch between the number of test sequences and percentages.")

    # Tokenize sequences and convert to tensors
    try:
        tokenizer = load_tokenizer(model_name)
        train_encodings = tokenize_sequences(tokenizer, train_sequences)
        test_encodings = tokenize_sequences(tokenizer, test_sequences)

        # Extract input_ids from the encodings
        train_sequences = train_encodings['input_ids'].to(device)
        test_sequences = test_encodings['input_ids'].to(device)

        # Convert percentages to tensors
        train_percentages = torch.tensor(train_percentages, dtype=torch.float32).unsqueeze(1).to(device)
        test_percentages = torch.tensor(test_percentages, dtype=torch.float32).unsqueeze(1).to(device)
        
    except Exception as e:
        raise RuntimeError(f"Error during tokenization or tensor conversion: {e}")

    # Hyperparameter tuning loop
    for idx, params in enumerate(hyperparameter_grid()):
        print(f"Training with params: {params}")
        try:
            print('Model Name:', model_name)
            model = load_model(model_name, device=device)
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        except Exception as e:
            raise RuntimeError(f"Error initializing model or optimizer: {e}")

        try:
            train_model(
                model, optimizer, train_sequences, train_percentages,
                num_epochs=params['num_epochs'],
                batch_size=params['batch_size'],
                device=device
            )
        except Exception as e:
            raise RuntimeError(f"Error during training: {e}")

        try:
            predicted_y = evaluate_model(model, test_sequences, device=device)
            print(f"Predictions: {predicted_y}")
        except Exception as e:
            raise RuntimeError(f"Error during evaluation: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
