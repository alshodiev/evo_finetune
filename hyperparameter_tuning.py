from itertools import product

def hyperparameter_grid():
    param_grid = {
        'num_epochs': [1],  
        'batch_size': [16], 
        'learning_rate': [0.001] 
    }

    for params in product(*param_grid.values()):
        yield dict(zip(param_grid.keys(), params))
