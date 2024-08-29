from itertools import product
import torch.optim as optim


def hyperparameter_grid():
    param_grid = {
        #'optimizer': ['AdamW'],  # Example optimizers 
        'num_epochs': [2],
        'batch_size': [16], # maybe size=8 
        'accumulation_steps': [2],
        'learning_rate': [0.001]
    }
    '''
    param_grid = {
        'num_epochs': [10, 20, 30],
        'batch_size': [16, 32, 64],
        'accumulation_steps': [2, 4, 8],
        'learning_rate': [0.001, 0.0005, 0.0001]
    }
    '''
    for params in product(*param_grid.values()):
        yield dict(zip(param_grid.keys(), params))

        