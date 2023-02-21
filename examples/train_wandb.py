from braivest.train.Trainer import Trainer
import wandb
from braivest.utils import load_data
import os

def train_config(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config

        artifact = run.use_artifact(config.data_artifact, type='dataset')
        artifact_dir = artifact.download()

        train_X = load_data(artifact_dir, 'train.npy')
        input_dim = train_X.shape[1]
        trainer= Trainer(config, input_dim)

        trainer.load_dataset(artifact_dir)

        history = trainer.train(wandb=True)
        #save last model just in case
        trainer.model.save_weights(os.path.join(wandb.run.dir, "model.h5"))

def main():
	#sweep_id = SET SWEEP ID HERE
	wandb.login()

	wandb.agent(sweep_id, train_config, count=30)
if __name__ == '__main__':
	main()