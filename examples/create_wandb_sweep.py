import wandb
wandb.login()

sweep_config = {
    'method': 'grid'
    }
metric = {
    'name': 'val_loss',
    'goal': 'minimize'
    }

sweep_config['metric'] = metric
parameters_dict = {
    'num_layers':{
        'value': 2
    },
    'layer_dims':{
        'value': 250
    },
    'batch_size': {
        'values':  [1000, 4000, 10000]
    },
    'lr': {
        'values': [1e-5, 1e-4, 1e-3]
    },
    'nw':{
        'value': 0
    },
    'kl':{
        'values': [0, 1e-4, 1e-2, 1]
    },
    'latent': {
        'value': 2
    },
    'time':{
        'value': False
    },
    'emg': {
        'value': True
    },
    'save_best':{
        'value': False
    }
    }

sweep_config['parameters'] = parameters_dict
parameters_dict.update({
    'epochs': {
        'value': 1000},
    'data_artifact': {
        'value': 'juliahwang/braivest_tutorial/training_set:v0'
    },
    })
sweep_id = wandb.sweep(sweep_config, project="YOUR PROJECT HERE")
