import torch
from tokenization import load_tokenizer, tokenize_sequences, convert_to_tensor
from model import load_model, setup_optimizer
from train import train_model
from evaluation import evaluate_model
from hyperparameter_tuning import hyperparameter_grid, get_optimizer
import pandas as pd

def main():
    model_name = 'togethercomputer/evo-1-131k-base'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your data
    train_sequences = pd.read_csv('train_sequences.csv').values.tolist()
    train_percentages = pd.read_csv('train_percentages.csv').values.tolist()

    test_sequences = pd.read_csv('test_sequences.csv').values.tolist()
    test_percentages = pd.read_csv('test_percentages.csv').values.tolist()

    # Hyperparameter tuning
    for params in hyperparameter_grid():
        print(f"Training with params: {params}")
        model = load_model(model_name, device=device)
        optimizer = get_optimizer(params['optimizer'], model, learning_rate=params['learning_rate'])

        # Training
        train_model(
            model, optimizer, train_sequences, train_percentages,
            num_epochs=params['num_epochs'],
            batch_size=params['batch_size'],
            accumulation_steps=params['accumulation_steps'],
            device=device
        )

        # Evaluation
        predicted_y = evaluate_model(model, test_sequences, device=device)
        print(f"Predictions: {predicted_y}")

if __name__ == "__main__":
    main()
