import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from braivest.model.emgVAE import emgVAE_Lightning

train, train_Y, test, test_Y, hypnos, ss = None, None, None, None, None, None  # Replace with actual data loading logic
train_loader = DataLoader(TensorDataset(torch.tensor(train, dtype=torch.float32), torch.tensor(train_Y, dtype=torch.float32)), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(test, dtype=torch.float32), torch.tensor(test_Y, dtype=torch.float32)), batch_size=64, shuffle=False)

# Model hyperparameters
# Can also use a for loop to iterate over different configurations
input_dim = 31 # this is based on the number of wavelets and the average EMG power.

latent_dim = 2 
hidden_states = [250,250]
kl_weight = 0.1
emg = True  # Set to False if you don't want to explicitly set the EMG channel in the latent space. 
#The EMG channel will still be used in the model as long as it's in your data.
# If true, the EMG channel should be the last channel in your data.

model = emgVAE_Lightning(input_dim, latent_dim, hidden_states, kl_weight)

# Add ModelCheckpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(
    monitor='val/neighbor_loss',
    dirpath='checkpoints',
    filename='emgvae-{epoch:02d}-{val_total_loss:.2f}',
    save_top_k=1,
    mode='min'
)

trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)

# Save the inner PyTorch model (not the LightningModule)
torch.save(model.model.state_dict(), "emgvae_final_model.pth")