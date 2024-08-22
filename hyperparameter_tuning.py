from itertools import product
import torch.optim as optim

def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate)
    # Add more optimizers as needed
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")

def hyperparameter_grid():
    param_grid = {
        'optimizer': ['AdamW'],  # Example optimizers
        'num_epochs': [10],
        'batch_size': [16],
        'accumulation_steps': [2],
        'learning_rate': [0.001]
    }
    '''
    param_grid = {
        'optimizer': ['AdamW', 'SGD'],  # Example optimizers
        'num_epochs': [10, 20, 30],
        'batch_size': [16, 32, 64],
        'accumulation_steps': [2, 4, 8],
        'learning_rate': [0.001, 0.0005, 0.0001]
    }
    '''
    for params in product(*param_grid.values()):
        yield dict(zip(param_grid.keys(), params))