import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import lightning as L

class stitchedVAE(nn.Module):
    def __init__(self, input_dim, pretrained_model):
        super(stitchedVAE, self).__init__()
        self.input_dim = input_dim
        self.pretrained_model = pretrained_model # emgVAE instance
        self.intermediate_dim = pretrained_model.encoder.hidden[2].in_features
        self.stitch = nn.Linear(self.input_dim, self.intermediate_dim)

    def encode(self, x, numpy=False):
        # pass through stitching layer
        x_int = self.stitch(x)
        # pass through the pretrained model's encoder
        h = self.pretrained_model.encoder.hidden[2:](x_int)
        mu = self.pretrained_model.encoder.fc_mu(h)
        logvar = self.pretrained_model.encoder.fc_logvar(h)
        z = self.pretrained_model.reparameterize(mu, logvar)
        if self.pretrained_model.emg:
            z = torch.cat((z, x[:, -1:]), dim=1)
        if numpy:
            return z.cpu().detach().numpy()
        return z
        
    def forward(self, x):
        # pass through stitching layer
        x_int = self.stitch(x)
        # pass through the pretrained model's encoder
        h = self.pretrained_model.encoder.hidden[2:](x_int)
        mu = self.pretrained_model.encoder.fc_mu(h)
        logvar = self.pretrained_model.encoder.fc_logvar(h)
        z = self.pretrained_model.reparameterize(mu, logvar)
        if self.pretrained_model.emg:
            z = torch.cat((z, x[:, -1:]), dim=1)
        x_recon = self.pretrained_model.decoder(z)
        return x_recon, mu, logvar, z
    
    def compute_loss(self, x, y, loss_dims=None):
        # pass through stitching layer
        x_recon, mu, logvar, z = self.forward(x)
        _, _, _, z_out = self.forward(y)
        # Reconstruction loss
        if loss_dims is None:
            loss_dims = slice(None)

        assert x_recon[:, loss_dims].shape == y.shape, "Reconstructed shape does not match input shape. Reconstructed shape: {}, Input shape: {}".format(x_recon[loss_dims].shape, y.shape)
        recon_loss = F.mse_loss(x_recon[:, loss_dims], y, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # Neighbor loss
        neighbor_loss = torch.mean(torch.norm(z - z_out, dim=1)) / torch.mean(torch.norm(z, dim=1))

        total_loss = recon_loss + self.pretrained_model.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss, neighbor_loss
    
class stitchedVAE_Lightning(L.LightningModule):
    def __init__(self, input_dim, pretrained_model, loss_dims=None, lr=1e-3, kl_weight=1.0):
        super(stitchedVAE_Lightning, self).__init__()
        self.save_hyperparameters()
        pretrained_model.kl_weight = kl_weight
        self.model = stitchedVAE(input_dim, pretrained_model)
        self.loss_dims = loss_dims
        self.lr = lr
        self.kl_weight = kl_weight

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, recon_loss, kl_loss, neighbor_loss = self.model.compute_loss(x, y, self.loss_dims)
        self.log('train/total_loss', loss)
        self.log('train/recon_loss', recon_loss)
        self.log('train/kl_loss', kl_loss)
        self.log('train/neighbor_loss', neighbor_loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, recon_loss, kl_loss, neighbor_loss = self.model.compute_loss(x, y, self.loss_dims)
        self.log('val/total_loss', loss)
        self.log('val/recon_loss', recon_loss)
        self.log('val/kl_loss', kl_loss)
        self.log('val/neighbor_loss', neighbor_loss)
    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, recon_loss, kl_loss, neighbor_loss = self.model.compute_loss(x, y, self.loss_dims)
        self.log('test/total_loss', loss)
        self.log('test/recon_loss', recon_loss)
        self.log('test/kl_loss', kl_loss)
        self.log('test/neighbor_loss', neighbor_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

