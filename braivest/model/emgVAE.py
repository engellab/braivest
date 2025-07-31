import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import lightning as L

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_states):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_states:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.hidden = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.hidden(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_states):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h in hidden_states:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class emgVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_states, kl_weight, emg=True):
        super().__init__()
        self.input_dim = input_dim
        self.emg = emg
        if emg:
            self.latent_dim = latent_dim - 1
        else:
            self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, self.latent_dim, hidden_states)
        self.decoder = Decoder(latent_dim, input_dim, hidden_states)
        self.kl_weight = kl_weight

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x, numpy=False):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        if self.emg:
            z = torch.cat([z, x[:, -1:].detach()], dim=1)
        if numpy:
            return z.detach().cpu().numpy()
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        if self.emg:
            z = torch.cat([z, x[:, -1:].detach()], dim=1)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def compute_loss(self, x, y):
        x_recon, mu, logvar, z_in = self.forward(x)
        _, _, _, z_out = self.forward(y)
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, y, reduction='mean')
        # KL divergence
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        # Neighbor loss
        neighbor_loss = torch.mean(torch.norm(z_in - z_out, dim=1))
        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss, neighbor_loss


class emgVAE_Lightning(L.LightningModule):
    def __init__(self, input_dim, latent_dim, hidden_states, kl_weight, emg=True, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = emgVAE(input_dim, latent_dim, hidden_states, kl_weight, emg)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        total_loss, recon_loss, kl_loss, neighbor_loss = self.model.compute_loss(x, y)
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/recon_loss', recon_loss)
        self.log('train/kl_loss', kl_loss)
        self.log('train/neighbor_loss', neighbor_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        total_loss, recon_loss, kl_loss, neighbor_loss = self.model.compute_loss(x, y)
        self.log('val/total_loss', total_loss, prog_bar=True)
        self.log('val/recon_loss', recon_loss)
        self.log('val/kl_loss', kl_loss)
        self.log('val/neighbor_loss', neighbor_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        total_loss, recon_loss, kl_loss, neighbor_loss = self.model.compute_loss(x, y)
        self.log('test/total_loss', total_loss, prog_bar=True)
        self.log('test/recon_loss', recon_loss)
        self.log('test/kl_loss', kl_loss)
        self.log('test/neighbor_loss', neighbor_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)