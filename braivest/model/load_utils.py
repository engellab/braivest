import yaml
import torch
from emgVAE import emgVAE

def load_model(model_path, config_path, input_dim, device='cpu'):
    """
    Load a PyTorch emgVAE model from saved weights and config.
    Inputs:
    - model_path (str): Path to the saved model weights (.pt or .pth)
    - config_path (str): Path to the YAML config file
    - input_dim (int): Input dimension of the model
    - device (str): Device to load the model on ('cpu' or 'cuda')
    Returns:
    - model (emgVAE): Loaded emgVAE model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # If config['layer_dims'] is a list, use it directly; otherwise, build from num_layers
    if isinstance(config['layer_dims'], list):
        layers = config['layer_dims']
    else:
        layers = [config['layer_dims'] for _ in range(config['num_layers'])]
    model = emgVAE(input_dim, config['latent'], layers, config['kl'], emg=config.get('emg', True))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model