import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

import numpy as np
import torch
from tqdm import tqdm

import random
import torch
import torch.nn as nn
import math
import torch
import random
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from itertools import islice

#Define parameters
from arguments import get_args
args = get_args()

num_epochs = args.num_epochs
batch_size = args.batch_size
learn_rate = args.learn_rate

latent_dim = args.latent_dim
belief_dim = args.belief_dim
num_sequences = args.num_sequences
sequence_length = args.len_sequence
speed = args.mnist_speed
digit = args.mnist_digit

load_chkpt_path = args.load_chkpt_path
assert os.path.exists(load_chkpt_path), "Please provide a valid checkpoint path!"

save_dir = f"{args.save_dir}/rollout"
os.makedirs(save_dir, exist_ok=True)

loss_type = args.loss
assert loss_type in ["bce", "git_bce"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Useful Funcs
def ax_standard(ax):

    ax.grid(True, alpha=0.5)
    ax.set_xlabel("Epoch")

def plot_results(history):

    line_color = "#f07167"

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.tight_layout(pad=5)

    ax[0].plot(history["loss"], color = line_color)
    ax_standard(ax[0])
    ax[0].set_ylabel("Total Loss")


    ax[1].plot(history["kl_loss"], color = line_color)
    ax_standard(ax[1])
    ax[1].set_ylabel("KL Loss")


    ax[2].plot(history["reconstruction_loss"], color = line_color)
    ax_standard(ax[2])
    ax[2].set_ylabel("Reconstruction Loss")

def plot_training_vs_validation_loss(history):
    """
    Plots the total loss, reconstruction loss, and KL divergence loss
    for both training and validation.

    Parameters:
    history: Keras history object containing training and validation loss values per epoch.
    """
    plt.figure(figsize=(10, 5))

    # Plot total loss
   # plt.plot(history.history["loss"], label="Train Loss", color="#545f66")
   # plt.plot(history.history["val_loss"], label="Validation Loss", color="#829399", linestyle="dashed")

    # Plot reconstruction loss
    plt.plot(history.history["reconstruction_loss"], label="Train Reconstruction Loss", color="#8BE4CB")
    #plt.plot(history.history["val_reconstruction_loss"], label="Validation Reconstruction Loss", color="#DAFA9E", linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("Training vs. Validation Reconstruction Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Plot KL loss
    plt.plot(history.history["loss"], label="Total Loss", color="#b1cc74")
    #plt.plot(history.history["val_kl_loss"], label="Validation KL Loss", color="#DAFA9E", linestyle="dashed")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("Training vs. Validation Kl_loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(history.history["kl_loss"], label="KL Loss", color="#b1cc74")
    #plt.plot(history.history["val_kl_loss"], label="Validation KL Loss", color="#DAFA9E", linestyle="dashed")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("Training vs. Validation Kl_loss")
    plt.legend()
    plt.grid(True)
    plt.show()


## Paper Implementation

#Generating Data
class MovingMNIST(torch.utils.data.Dataset):
    def __init__(self, num_sequences=10000, sequence_length=20, image_size=28, digit_size=28, speed=2, digit=99):
        self.mnist = MNIST(root='/mnt/TD-VAE/data', train=True, download=True, transform=transforms.ToTensor())
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.digit_size = digit_size
        self.speed = speed  # pixels/frame
        self.digit = digit

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        canvas_size = self.image_size #32x32 looks similiar as in paper increase if required
        frames = np.zeros((self.sequence_length, canvas_size, canvas_size), dtype=np.float32)   #Shape: (20, 32, 32)

        digit_img, _ = self.mnist[idx]       #Shape: (1, 28, 28)
        digit_img = digit_img[0]             #Shape: (28, 28)

        if self.speed not in range(0,5):
            self.speed = random.randint(1,4)

        # Random direction: -1 (left) or +1 (right)
        direction = random.choice([-1, 1]) #direction controlled by the training
        dx = direction * self.speed  # no of pixel left/right

        for t in range(self.sequence_length):
            frames[t] = np.roll(digit_img, shift=t * dx, axis=1)

        frames = torch.tensor(frames).unsqueeze(1)  # shape: (T, 1, H, W)
        return frames

def show_sequence(frames, title="Wrapped Digit Movement"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(frames), figsize=(len(frames), 1.5))
    for i, ax in enumerate(axes):
        ax.imshow(frames[i, 0], cmap="gray")
        ax.axis("off")
    #plt.suptitle(title)
    plt.show()

def time_interval():
    """
    Choose a random t1 in [1,19]
    Choose a random dt between 1 and 4
    Choose random direction +/-1
    Compute t2 = t1 + dt * direction
    Clip t2 to stay within [0,20]
    """
    t1 = random.randint(1, 18)
    dt = random.randint(1, 4)
    direction = random.choice([-1, 1])
    direction = 1  # TODo: Try for both left and right
    t2 = t1 + dt * direction

    # Border check: clip to [0, 20]
    if t2 < 0:
        t2 = 0
    elif t2 > 19:
        t2 = 19

    return t1, t2, direction


## Architecture
def sample_gaussian(mu, logvar):
  std = torch.exp(0.5 * logvar)
  #std = torch.exp(logvar)  #TODO CHANGED
  eps = torch.randn_like(std)
  return mu + std * eps

def log_normal_pdf(x, mean, logvar, eps=1e-6):
    log_two_pi = torch.log(torch.tensor(2. * math.pi, device=x.device, dtype=x.dtype))
    return -0.5 * torch.sum(
        log_two_pi + logvar + ((x - mean) ** 2) / (torch.exp(logvar) + eps),
        dim=-1
    )

def kl_divergence_old(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q) #getting back the variance from the log of variance
    var_p = torch.exp(logvar_p)
    return 0.5 * torch.sum(  #kl divergence of twogaussian distributions
        logvar_p - logvar_q +
        (var_q + (mu_q - mu_p)**2) / var_p - 1,
        dim=-1
    )

def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q) #getting back the variance from the log of variance
    var_p = torch.exp(logvar_p)
    return 0.5 * torch.sum(((mu_p - mu_q)/(var_p))**2, -1) + torch.sum(logvar_p, -1) - torch.sum(logvar_q, -1)


## The belief state network/agrregator is an LSTM.
# we will use the lstm to generate belief states

class BeliefLSTM(nn.Module):
    def __init__(self, input_dim=1, belief_dim = 50):
        super(BeliefLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=belief_dim, 
                            batch_first=True)

    def forward(self, x):
        b, _= self.lstm(x)
        return b #(batch_size, sequence_length, belief_dim) = (1000, 200, 50)

class PreProcess(nn.Module):
  """ The pre-process layer for MNIST image

  """
  def __init__(self, input_size=1024, processed_x_size=1024):
    super(PreProcess, self).__init__()
    self.input_size = input_size
    self.fc1 = nn.Linear(input_size, processed_x_size)
    self.fc2 = nn.Linear(processed_x_size, processed_x_size)

  def forward(self, input):
    t = torch.relu(self.fc1(input))
    t = torch.relu(self.fc2(t))
    return t

## From the paper we can see that DBlock can be MLP or RNN.
##For now we go with MLP
class DBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim = 50, latent_dim = 8):
        """
        input_dim: dimensionality of the input context (e.g., b_t, z_t2, etc.)
        hidden_dim: dimensionality of the hidden layer
        output_dim: dimensionality of the output (i.e., z size)
        """
        super(DBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # W1 and B1
        self.fc2 = nn.Linear(input_dim, hidden_dim)  # W2 and B2
        #self.fc3 = nn.Linear(hidden_dim, 2 * latent_dim)  # W3 and B3 (outputs both mu and log sigma)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """
        x: context input (batch_size, input_dim)
        returns: mu, log_sigma of shape (batch_size, output_dim)
        """
        t1 = torch.tanh(self.fc1(x))        # W1x + B1 → tanh
        t2 = torch.sigmoid(self.fc2(x))     # W2x + B2 → sigmoid
        t = t1 * t2                         # element-wise product
        #out = self.fc3(t)                   # W3·(t) + B3 → outputs both mu and log sigma
        #mu, log_sigma = torch.chunk(out, 2, dim=-1)
        mu = self.fc_mu(t)
        log_sigma = self.fc_logsigma(t)

        return mu, log_sigma

class Decoder_LSTM(nn.Module):
    """
    LSTM-based decoder for single frame reconstruction
    Input:  (B, latent_dim)
    Output: (B, 1, 32, 32)
    """
    def __init__(self, latent_dim, lstm_hidden=256, num_layers=2, output_size=32*32):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(lstm_hidden, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # Add sequence dimension: (B, 1, latent_dim)
        z = z.unsqueeze(1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(z)   # (B, 1, hidden)
        x = lstm_out[:, -1, :]       # Take last step

        # Map to image space
        x = self.fc(x)               # (B, 1024)
        x = self.sigmoid(x)
        x = x.view(-1, 1, 32, 32)    # (B, 1, 32, 32)
        return x

class Decoder(nn.Module):
    """ The decoder layer converting state to observation.
    Because the observation is MNIST image whose elements are values
    between 0 and 1, the output of this layer are probabilities of
    elements being 1.
    """
    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        logits = (self.fc3(t))
        return logits

class Decoder_Git(nn.Module):
    """ The decoder layer converting state to observation.
    Because the observation is MNIST image whose elements are values
    between 0 and 1, the output of this layer are probabilities of
    elements being 1.
    """
    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder_Git, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p

class TDVAE_hierachical(nn.Module):
    def __init__(self, dblock, BeliefLSTM, preprocess, decoder, input_size, processed_x_size, belief_dim=50, latent_dim_1=8, latent_dim_2=8):
        super().__init__()
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2

        self.preprocess = preprocess(input_size, processed_x_size)

        self.beliefs = BeliefLSTM(processed_x_size) #be careful with input dimension after preprocessing

        self.belief_layer2 = dblock(belief_dim, 50, latent_dim_2)
        self.belief_layer1 = dblock(belief_dim + latent_dim_2, 50, latent_dim_1)

        self.smoothing_layer2 = dblock(belief_dim + latent_dim_1 + latent_dim_2, 50, latent_dim_1)
        self.smoothing_layer1 = dblock(belief_dim + latent_dim_2 + latent_dim_1 + latent_dim_2, 50, latent_dim_1)

        self.transition_layer2 = dblock(latent_dim_2 + latent_dim_1, 50, latent_dim_1)
        self.transition_layer1 = dblock(latent_dim_2 + latent_dim_1 +latent_dim_2, 50, latent_dim_1)

        self.decoder = decoder(latent_dim_1 + latent_dim_2, 200,input_size)

        self.total_loss = 0.0
        self.reconstruction_loss = 0.0
        self.kl_loss = 0.0

    def reset_loss_trackers(self):
        self.total_loss = 0.0
        self.reconstruction_loss = 0.0
        self.kl_loss = 0.0

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data, optimizer, device, beta, loss_type="bce"):
        data = data.to(device)
        t1, t2, drxn= time_interval()

        #preprocess
        B, T, C, H, W = data.shape
        data = data.view(B, T, -1)
        original_data = data.clone()
        preprocessed_data = self.preprocess(data)

        #encoder
        bt = self.beliefs(preprocessed_data)            
        mu2, logvar2 = self.belief_layer2(bt[:, t2, :])  # shape: [batch, latent_dim_2]
        zt2_layer2 = sample_gaussian(mu2, logvar2) #term 1

        mu1, logvar1 = self.belief_layer1(torch.cat((bt[:, t2, :], zt2_layer2), dim=-1))
        zt2_layer1 = sample_gaussian(mu1, logvar1) #term 2

        zt2 = torch.cat((zt2_layer1, zt2_layer2), dim =-1) #term3

        mut1_layer2, logvart1_layer2 = self.belief_layer2(bt[:, t1, :])  # shape: [batch, latent_dim_2] #term 4


        #smoothing
        dt = torch.full((preprocessed_data.size(0), 1), t2 - t1, dtype=bt.dtype, device=preprocessed_data.device)
        # mu_smooth_layer2, logvar_smooth_layer2 = self.smoothing_layer2(torch.cat((bt[:, t1, :],zt2,dt], dim=-1))
        mu_smooth_layer2, logvar_smooth_layer2 = self.smoothing_layer2(torch.cat((bt[:, t1, :],zt2), dim=-1))
        zt1_layer2_smooth = sample_gaussian(mu_smooth_layer2, logvar_smooth_layer2) #term5

        mut1_layer1, logvart1_layer1 = self.belief_layer1(torch.cat((bt[:, t1, :], zt1_layer2_smooth), dim=-1)) #term 6

        # mu_smooth_layer1, logvar_smooth_layer1 = self.smoothing_layer1(torch.cat((bt[:, t1, :],zt2, zt1_layer2_smooth, dt], dim=-1))
        mu_smooth_layer1, logvar_smooth_layer1 = self.smoothing_layer1(torch.cat((bt[:, t1, :],zt2, zt1_layer2_smooth), dim=-1))
        zt1_layer1_smooth = sample_gaussian(mu_smooth_layer1, logvar_smooth_layer1) #term7

        zt1 = torch.cat((zt1_layer1_smooth, zt1_layer2_smooth ), dim = -1) #term 8

        #transition
        # mu_trans_layer2 , logvar_trans_layer2 = self.transition_layer2(torch.cat((zt1,dt], dim = -1)) #term 9

        # mu_trans_layer1 , logvar_trans_layer1 = self.transition_layer1(torch.cat((zt1,zt2_layer2, dt], dim = -1)) #term 10

        mu_trans_layer2 , logvar_trans_layer2 = self.transition_layer2(zt1) #term 9
        mu_trans_layer1 , logvar_trans_layer1 = self.transition_layer1(torch.cat((zt1,zt2_layer2), dim = -1)) #term 10

        #decoder
        #reconstruction = self.decoder(zt2_layer1[:,:1]) #IMP!! This is only for harmonic oscillator. dont do this for MNIST remember to change
        reconstruction = self.decoder(zt2)
        target = original_data[:, t2, :]          # shape [B, 1024]
        
        #BCE Implementation
        if loss_type == "bce":
            bce = nn.BCEWithLogitsLoss(reduction='none')
            Lx = bce(reconstruction, target).sum(dim=1)  # sum over input dimensions    #TODO CHECK

        #Git Implementation
        else:
            Lx = -torch.sum(target*torch.log(reconstruction) + (1-target)*torch.log(1-reconstruction), -1)


        #calculating losses now
        # L1 = kl_divergence(mu_smooth_layer2, logvar_smooth_layer2, mut1_layer2, logvart1_layer2)
        # L2 = kl_divergence(mu_smooth_layer1, logvar_smooth_layer1, mut1_layer1, logvart1_layer1)
        L1 = kl_divergence(zt1_layer2_smooth, logvar_smooth_layer2, mut1_layer2, logvart1_layer2)
        L2 = kl_divergence(zt1_layer1_smooth, logvar_smooth_layer1, mut1_layer1, logvart1_layer1)
        L3 = log_normal_pdf(zt2_layer2, mu2, logvar2) - log_normal_pdf(zt2_layer2, mu_trans_layer2, logvar_trans_layer2)
        L4 = log_normal_pdf(zt2_layer1, mu1, logvar1) - log_normal_pdf(zt2_layer1, mu_trans_layer1, logvar_trans_layer1)


        #Lx = F.mse_loss(reconstruction, data[:,t2,:])#.sum(dim=-1)
        #Lx = -log_normal_pdf(data[:,t2,:],reconstruction,torch.tensor([0.0]))

        total_loss = (Lx + beta*(L1 + L2 + L3 + L4)).mean()
        reconstruction_loss = Lx.mean()
        kl_loss = (L1 + L2).mean()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # === Return metrics ===
        return {
            'loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item()
        }


#Rollout Function
def tdvae_rollout(vae, x_sample, t1=10, t_end=15, loss_type="bce", device='cuda'):
    """
    Performs autoregressive rollout from t1 to t_end for a single batch item.

    Args:
        vae: Trained TD-VAE model
        x: Input tensor of shape [B, T, 1, 32, 32]
        t1: Starting timestep for rollout
        t_end: Ending timestep for rollout
        batch_idx: Which index from the batch to visualize
        device: 'cuda' or 'cpu'

    Returns:
        List of predicted frames [ (1, 32, 32), ... ] for the selected batch_idx
    """
    vae.eval()
    with torch.no_grad():

        T, C, H, W = x_sample.shape
        data = x_sample.view(T, -1).unsqueeze(0)  # shape: [1, T, 784]

        # Preprocess
        processor = PreProcess(784, 784).to(device)
        preprocessed_data = processor(data)  # shape: [1, T, 784]

        bt = vae.beliefs(preprocessed_data)

        # Get z_t1
        mu2, logvar2 = vae.belief_layer2(bt[:, t1 - 1, :])
        zt2_layer2 = sample_gaussian(mu2, logvar2)
        mu1, logvar1 = vae.belief_layer1(torch.cat((bt[:, t1 - 1, :], zt2_layer2), dim=-1))
        zt2_layer1 = sample_gaussian(mu1, logvar1)
        zt1 = torch.cat((zt2_layer1, zt2_layer2), dim=-1)

        xt2_list = []

        for dt in range(1, t_end - t1 + 1):
            dt_tensor = torch.full((zt1.size(0), 1), float(1), device=zt1.device)

            #   mu_trans2, logvar_trans2 = vae.transition_layer2(torch.cat((zt1, dt_tensor], dim=-1))
            mu_trans2, logvar_trans2 = vae.transition_layer2(zt1)
            zt2_layer2 = sample_gaussian(mu_trans2, logvar_trans2)

            #   mu_trans1, logvar_trans1 = vae.transition_layer1(torch.cat((zt1, zt2_layer2, dt_tensor], dim=-1))
            mu_trans1, logvar_trans1 = vae.transition_layer1(torch.cat((zt1, zt2_layer2), dim=-1))
            zt2_layer1 = sample_gaussian(mu_trans1, logvar_trans1)

            zt2 = torch.cat((zt2_layer1, zt2_layer2), dim=-1)
            logits = vae.decoder(zt2)

            if loss_type == "bce":
                probs = torch.sigmoid(logits)
                xt2 = probs.view(1, 28, 28)
            else:
                xt2 = logits.view(1, 28, 28) 

            # Append only the selected index
            xt2_list.append(xt2)  # shape [1, 28, 28]

            zt1 = zt2

        return xt2_list


def to_numpy(img_tensor):
    return img_tensor.detach().cpu().numpy()

def plot_predictions_with_gt(x_batch, t1, xt2_list, save_path):

    total_cols = 20
    fig, axes = plt.subplots(2, total_cols, figsize=(2 * total_cols, 4))

    # Row 1: Ground Truth
    for i in range(total_cols):

        axes[0, i].imshow(to_numpy(x_batch[i, 0]), cmap='gray')

        #Title according to time stamp
        if i < t1: axes[0, i].set_title(f"GT {i}")
        else:  axes[0, i].set_title(f"GT t+{i - t1 + 1}")

        axes[0, i].axis('off')

    # Row 2: Predictions
    for j, pred_frame in enumerate(xt2_list):
        col_idx = t1 + j
        axes[1, col_idx].imshow(to_numpy(pred_frame[0]), cmap='gray')
        axes[1, col_idx].set_title(f"Pred t+{j+1}")
        axes[1, col_idx].axis('off')


    for i in range(t1):
        axes[1, i].axis('off')

    for i in range(t1+len(xt2_list), total_cols):
        axes[1, i].axis('off')

    plt.tight_layout()

    fig.savefig(save_path)


# Generate data
X_train = MovingMNIST(num_sequences=num_sequences,
                      sequence_length=sequence_length, 
                      speed=speed, 
                      digit=digit)

# # Create data loaders
train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

#Initializing TD-VAE Model
vae = TDVAE_hierachical(DBlock, 
                        BeliefLSTM,
                        PreProcess, 
                        Decoder if loss_type == "bce" else Decoder_Git,        #TODO CHANGED 
                        784, 784, 
                        belief_dim=belief_dim, latent_dim_1=latent_dim, latent_dim_2=latent_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=learn_rate)

#Load Checkpoint Weights
chkpt = torch.load(load_chkpt_path)
chkpt_epoch = chkpt["epoch"]
vae.load_state_dict(chkpt["model_state_dict"])
optimizer.load_state_dict(chkpt["optimizer_state_dict"])

vae.to(device)
vae.eval()

for batch in train_loader:
    x = batch.to(device)
    break

#Run rollout
for batch_idx in np.arange(5):
    print(f"Rollout for Batch {batch_idx}")

    # x_batch = X_train[batch_idx].to(device)
    x_batch = x[batch_idx]

    preds = tdvae_rollout(vae, x_batch, 
                          t1 = 10, t_end = 15,
                          loss_type = loss_type,
                          device = device)
    
    plot_predictions_with_gt(x_batch,  
                             t1 = 10, 
                             xt2_list = preds,
                             save_path = f"{save_dir}/epoch_{chkpt_epoch}_batch_{batch_idx}.png")