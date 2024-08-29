import torch
from tokenization import load_tokenizer, tokenize_sequences, convert_to_tensor
from model import load_model, setup_optimizer
from train import train_model
from evaluation import evaluate_model
from hyperparameter_tuning import hyperparameter_grid
import pandas as pd
import torch.optim as optim
import wandb
#

wandb.init(project='Evo_Finetuning')

def main():
    model_name = 'togethercomputer/evo-1-131k-base'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_sequences = pd.read_csv('train_sequences.csv').values.tolist()
    train_percentages = pd.read_csv('train_percentages.csv').values.tolist()

    test_sequences = pd.read_csv('test_sequences.csv').values.tolist()
    test_percentages = pd.read_csv('test_percentages.csv').values.tolist()

    # Tokenize sequences and convert to tensors
    tokenizer = load_tokenizer(model_name)
    train_sequences = convert_to_tensor(tokenize_sequences(tokenizer, train_sequences), device)
    test_sequences = convert_to_tensor(tokenize_sequences(tokenizer, test_sequences), device)
    train_percentages = torch.tensor(train_percentages, device=device)
    test_percentages = torch.tensor(test_percentages, device=device)


    for idx, params in enumerate(hyperparameter_grid()):
        print(f"Training with params: {params}")
        model = load_model(model_name, device=device)

        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        train_model(
            model, optimizer, train_sequences, train_percentages,
            num_epochs=params['num_epochs'],
            batch_size=params['batch_size'],
            device=device
        )

        predicted_y = evaluate_model(model, test_sequences, device=device)
        print(f"Predictions: {predicted_y}")


if __name__ == "__main__":
    main()