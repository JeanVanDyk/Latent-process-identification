import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import lightning as L
import logging

logging.getLogger("ligthning.pytorch").setLevel(logging.INFO)

# configure logging on module level, redirect to file
logger = logging.getLogger("lightning.pytorch.core")
logger.addHandler(logging.StreamHandler())

###
# This model follows the nn structure introduce by El Esawey et al. (2015). It is a classic denoising auto encoder,
# in which we've added a recurrent connection to detect time dependecies, between the encoder and the decoder.
# In this paper, their model used hyperparameters found using hyperopt, which means that they are particular to their dataset. 
#
# TODO : Add pre-training.
###

class Encoder(nn.Module):
    def __init__(self, time_length, nb_joints, latent_dim):
        super().__init__()

        # I've added normalisation, which is supposed to improve accuracy and training efficiency according to Ioffe, Szegedy (2015). 
        # According to Pytorch forums, LayerNorm keeps time dependencies unlike BatchNorm who normalise accross the batch. 

        self.ln0 = nn.LayerNorm(nb_joints*3)
        self.ln1 = nn.LayerNorm(nb_joints*2)
        self.ln2 = nn.LayerNorm(nb_joints)

        self.conv1 = nn.Conv2d(in_channels=time_length,
                              out_channels=time_length,
                              kernel_size=(5,3),
                              padding=(2,1))
        self.conv2 = nn.Conv2d(in_channels=time_length,
                              out_channels=time_length,
                              kernel_size=(5,2),
                              padding=(2,0))
        self.conv3 = nn.Conv2d(in_channels=time_length,
                              out_channels=time_length * 2,
                              kernel_size=(5,2),
                              padding=(2,0))
        
        self.output = nn.Linear(2*nb_joints, latent_dim)   

    def forward(self, x):
        batchsize, time_length, nb_joints, dim = x.shape

        # Layer of convolution for detecting spatial dependencies
        x = self.ln0(x.reshape(batchsize, time_length, nb_joints * 3))
        x = x.reshape(batchsize, time_length, nb_joints, 3)
        x = F.relu(self.conv1(x))

        # Layer of convolution for reducing dimensions from N x T x L x 3 to N x T x L x 2
        x = self.ln0(x.reshape(batchsize, time_length, nb_joints * 3))
        x = x.reshape(batchsize, time_length, nb_joints, 3)
        x = F.relu(self.conv2(x))

        # Layer of convolution for reducing dimensions from N x T x L x 2 to N x 2(T x L)
        x = self.ln1(x.reshape(batchsize, time_length, nb_joints * 2))
        x = x.reshape(batchsize, time_length, nb_joints, 2)
        x = F.relu(self.conv3(x))

        # We're reshaping it this way to keep temporal dependencies for 
        res = torch.zeros((batchsize, time_length, 2*nb_joints))
        x = x.reshape(batchsize, 2*time_length, nb_joints)
        for i in range(time_length):
            res[:,i,:] = torch.concat((x[:,i,:], x[:,i+time_length,:]), dim=1)

        # Output
        res = self.output(self.ln1(res))
        return res

class Decoder(nn.Module):
    def __init__(self, latent_dim, time_length, nb_joints):
        super().__init__()

        # We've added normalisation, which is supposed to improve accuracy and training efficiency according to Ioffe, Szegedy (2015). 
        # According to Pytorch forums, LayerNorm keeps time dependencies unlike BatchNorm who normalise accross the batch. 
        self.time_length = time_length
        self.nb_joints = nb_joints
        

        self.ln0 = nn.LayerNorm(latent_dim)
        self.ln1 = nn.LayerNorm(nb_joints)
        self.ln2 = nn.LayerNorm(nb_joints*2)
        self.ln3 = nn.LayerNorm(nb_joints*3)

        self.conv1 = nn.Conv2d(in_channels=time_length,
                              out_channels=time_length,
                              kernel_size=(5,2),
                              padding=(2,1))
        
        self.conv2 = nn.Conv2d(in_channels=time_length,
                              out_channels=time_length,
                              kernel_size=(5,2),
                              padding=(2,1))
        
        self.conv3 = nn.Conv2d(in_channels=time_length,
                              out_channels=time_length,
                              kernel_size=(5,1),
                              padding=(2,0))
        
        self.lin1 = nn.Linear(latent_dim, nb_joints)   

    def forward(self, x):

        # Symmetric of the output layer of the decoder
        x = self.lin1(self.ln0(x))

        # Layer of convolution for going from dimensions N x T x L to N x T x L x 2
        x = self.ln1(x)
        x = x.reshape(*x.shape[0:2], self.nb_joints, 1)
        x = F.relu(self.conv1(x))

        # Layer of convolution for going from dimensions N x T x L x 2 to N x T x L x 3
        x = self.ln2(x.reshape(*x.shape[0:2], self.nb_joints * 2))
        x = x.reshape(*x.shape[0:2], self.nb_joints, 2)
        x = F.relu(self.conv2(x))

        # Layer of convolution for spatial accurcy
        x = self.ln3(x.reshape(*x.shape[0:2], self.nb_joints * 3))
        x = x.reshape(*x.shape[0:2], self.nb_joints, 3)
        return self.conv3(x)


class LitDAE(L.LightningModule):
    def __init__(self, time_length, nb_joints, latent_dim, missing_ratio, lr, batchsize, epochs):
        super().__init__()
        self.save_hyperparameters()
        self.time_length = time_length
        self.nb_joints = nb_joints
        self.latent_dim = latent_dim
        self.missing_ratio = missing_ratio
        self.lr = lr
        self.batchsize = batchsize
        self.epochs = epochs
        self.encoder = Encoder(time_length, nb_joints, latent_dim)
        self.decoder = Decoder(latent_dim, time_length, nb_joints)

        # I've currently chosen GRU over RNN and LSTM because it resolves the vanishing 
        # gradient issue from RNN and is easier to train than the LSTM

        self.ln = nn.LayerNorm(latent_dim)
        self.gru = nn.GRU(latent_dim, latent_dim, batch_first=True)
    
    def training_step(self, batch, batch_idx):
        x = batch
        x = x.float()

        # Randomly set missing_ratio of the time steps to zero
        x_masked = x.clone()
        nb_missing = int(self.missing_ratio * self.time_length)

        for i in range(len(x_masked)):
            random_timesteps = torch.randint(0, self.time_length, (nb_missing,))
            x_masked[i, random_timesteps, :,:] = 0.

        z = self.encoder(x_masked)
        z = self.ln(z)
        h_states, _ = self.gru(z)
        x_hat = self.decoder(h_states)

        loss = F.mse_loss(x_hat, x)
        self.log("train_reconstruction_loss", loss)
        return loss

    def on_train_epoch_end(self):
        # Log the learning rate at the end of each epoch
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', current_lr)
    
    def validation_step(self, batch, batch_idx):

        x = batch
        x = x.float()

        # Randomly set missing_ratio of the time steps to zero
        x_masked = x.clone()
        nb_missing = int(self.missing_ratio * self.time_length)

        for i in range(len(x_masked)):
            random_timesteps = torch.randint(0, self.time_length, (nb_missing,))
            x_masked[i, random_timesteps, :,:] = 0.

        z = self.encoder(x_masked)
        z = self.ln(z)
        h_states, _ = self.gru(z)
        x_hat = self.decoder(h_states)

        loss = F.mse_loss(x_hat, x)
        self.log("validation_reconstruction_loss", loss)
    
    def test_step(self, batch, batch_idx):

        x = batch
        x = x.float()

        # Randomly set missing_ratio of the time steps to zero
        x_masked = x.clone()
        nb_missing = int(self.missing_ratio * self.time_length)

        for i in range(self.batchsize):
            random_timesteps = torch.randint(0, self.time_length, (nb_missing,))
            x_masked[i, random_timesteps, :,:] = 0.

        z = self.encoder(x_masked)
        z = self.ln(z)
        h_states, _ = self.gru(z)
        x_hat = self.decoder(h_states)

        loss = F.mse_loss(x_hat, x)
        self.log("test_reconstruction_loss", loss)

    def forward(self, x):

        # Here we assume that the data already has missing values

        z = self.encoder(x)
        h_states, _ = self.gru(z)
        x_hat = self.decoder(h_states)

        return x_hat

    def configure_optimizers(self):

        # This is the most used optimizer in recent work. AdamW came as a correction of Adam 
        # on the way they implemented weight decay according to Loshchilov & Hutter (2019)

        # This learning scheduler permit to achieve training 2 to 4 time faster accordig 
        # to Loshchilov & Hutter (2017)

        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, self.epochs)
        return {"optimizer": optimizer,
                "lr_scheduler":{
                    "scheduler": scheduler,
                    "monitor": "epoch"
                }}