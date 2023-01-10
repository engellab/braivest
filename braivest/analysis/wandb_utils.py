import wandb
from braivest.model.emgVAE import emgVAE

def load_wandb_model(run, input_dim, epoch=None):
    api = wandb.Api()
# run is specified by <entity>/<project>/<run id>

    run = api.run(run)
    layers = [run.config.layer_dims for layer in range(run.config.num_layers)]
    model = emgVAE(input_dim, run.config.latent, layers, run.config['kl'], emg = run.config.emg)
    model.build((None, input_dim))
    try:
        run.file("model.h5".format(epoch)).download(root = './temp', replace=True)
        model.load_weights('./temp/model.h5'.format(epoch))
    except:
        run.file("model-best.h5").download(root = './temp', replace=True)
        model.load_weights('./temp/model-best.h5')
    return model

