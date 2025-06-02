import wandb
from braivest.model.emgVAE import emgVAE
import yaml

def load_wandb_model(run, input_dim, epoch=None):
    """
    Load a model from a wandb run
    Inputs:
    -run (dtype:str): The id of the run, specified by <entity>/<project>/<run id>
    -input_dim (dtype:int): The input dimension of the model
    -epoch (dtype: int): The epoch of model to load. If None, then try to load model.h5 or model_best.h5
    """
    api = wandb.Api()
    run = api.run(run)
    layers = [run.config['layer_dims'] for layer in range(run.config['num_layers'])]
    if 'emg' not in run.config.keys():
        run.config['emg'] = True
    model = emgVAE(input_dim, run.config['latent'], layers, run.config['kl'], emg = run.config['emg'])
    model.build((None, input_dim))
    if epoch:
        try:
            run.file("model_{}.h5".format(epoch)).download(root = './temp', replace=True)
            model.load_weights('./temp/model_{}.h5'.format(epoch))
        except:
            print("No model for that epoch found")
            raise

    else:
        try:
            run.file("model.h5").download(root = './temp', replace=True)
            model.load_weights('./temp/model.h5')
        except:
            try:
                run.file("model-best.h5").download(root = './temp', replace=True)
                model.load_weights('./temp/model-best.h5')
            except:
                print("No model found")
                raise       
    return model

def load_model(model_path, config_path, input_dim):
    """
    Load a model from a path
    Inputs:
    -model_path (dtype:str): The path to the model
    -config_path (dtype:str): The path to the config file
    -input_dim (dtype:int): The input dimension of the model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    layers = [config['layer_dims'] for layer in range(config['num_layers'])]
    model = emgVAE(input_dim, config['latent'], layers, config['kl'], emg = config['emg'])
    model.build((None, input_dim))
    model.load_weights(model_path)
    return model