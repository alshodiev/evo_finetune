from itertools import product
from flash_attn.modules.mha import MHA

def hyperparameter_grid():
    param_grid = {
        'num_epochs': [1],  
        'batch_size': [16], 
        'learning_rate': [0.001] 
    }
    '''
    # Example of a more extensive grid for exploration
    param_grid = {
        'num_epochs': [1, 5, 10],  # Fewer epochs can be used for fine-tuning
        'batch_size': [4, 8, 16],  # Varying batch sizes can affect training stability and speed
        'learning_rate': [0.001, 0.0005, 0.0001]  # Try different learning rates to find the optimal value
    }
    '''

    for params in product(*param_grid.values()):
        yield dict(zip(param_grid.keys(), params))
