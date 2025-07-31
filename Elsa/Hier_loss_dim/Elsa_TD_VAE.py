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

# ROOT_DIR = "/mnt/TD-VAE/Elsa/Hier_var_speed"
# os.makedirs(ROOT_DIR, exist_ok = True)

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
if os.path.exists(load_chkpt_path): load_chkpt = True
else: load_chkpt = False

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

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
    def __init__(self, num_sequences=10000, sequence_length=20, image_size=32, digit_size=28, speed=2, digit=99):
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
        frames = np.zeros((self.sequence_length, canvas_size, canvas_size), dtype=np.float32)

        # Select a random MNIST digit
        #digit_img, _ = self.mnist[random.randint(0, len(self.mnist) - 1)] #Random
        digit_img, label = self.mnist[random.randint(0, len(self.mnist) - 1)]

        if self.digit in range(0,10):
        # Uncomment the following line to select only a specific digit (e.g., 3) for testing
            while label != self.digit: digit_img, label = self.mnist[random.randint(0, len(self.mnist) - 1)]
            digit_img = digit_img.squeeze(0).numpy()

        else:
            digit_img = digit_img.squeeze(0).numpy()

        if self.speed in range(0,5):
            self.speed = self.speed

        else:
            self.speed = random.randint(1,4)



        # Random direction: -1 (left) or +1 (right)
        #direction = random.choice([-1, 1]) direction controlled by the training


        direction = 1
        direction = random.choice([-1, 1]) #direction controlled by the training

        dx = direction * self.speed  # no of pixel left/right

        # Random start
        x = random.randint(0, canvas_size - self.digit_size)
        y = random.randint(0, canvas_size - self.digit_size)
        #y = (canvas_size - self.digit_size) // 2  # center vertically if needed

        for t in range(self.sequence_length):
            x_int = int(np.floor(x)) % canvas_size  # wrap x using modulo

            # Handle wrap-around drawing
            if x_int + self.digit_size <= canvas_size:
                # Fully inside canvas
                frames[t, y:y+self.digit_size, x_int:x_int + self.digit_size] += digit_img
            else:
                # Split across right and left border
                right_part = canvas_size - x_int
                left_part = self.digit_size - right_part

                # Right slice
                frames[t, y:y+self.digit_size, x_int:] += digit_img[:, :right_part]
                # Left slice
                frames[t, y:y+self.digit_size, :left_part] += digit_img[:, right_part:]

            x += dx  # move with wrapping

        frames = np.clip(frames, 0, 1)
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
  eps = torch.randn_like(std)
  return mu + eps * std

def log_normal_pdf(x, mean, logvar, eps=1e-6):
    log_two_pi = torch.log(torch.tensor(2. * math.pi, device=x.device, dtype=x.dtype))
    return -0.5 * torch.sum(
        log_two_pi + logvar + ((x - mean) ** 2) / (torch.exp(logvar) + eps),
        dim=-1
    )

def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q) #getting back the variance from the log of variance
    var_p = torch.exp(logvar_p)
    return 0.5 * torch.sum(  #kl divergence of twogaussian distributions
        logvar_p - logvar_q +
        (var_q + (mu_q - mu_p)**2) / var_p - 1,
        dim=-1
    )


## The belief state network/agrregator is an LSTM.
# we will use the lstm to generate belief states

class BeliefLSTM(nn.Module):
    def __init__(self, input_dim=1, belief_dim = 50):
        super(BeliefLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=belief_dim, batch_first=True)

    def forward(self, x):
        b, _= self.lstm(x)
        return b #(batch_size, sequence_length, belief_dim) = (1000, 200, 50)

        #preprocess

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
    def __init__(self, hidden_dim = 50, latent_dim = 8):
        """
        input_dim: dimensionality of the input context (e.g., b_t, z_t2, etc.)
        hidden_dim: dimensionality of the hidden layer
        output_dim: dimensionality of the output (i.e., z size)
        """
        super(DBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # W1 and B1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # W2 and B2
        self.fc3 = nn.Linear(hidden_dim, 2 * latent_dim)  # W3 and B3 (outputs both mu and log sigma)

    def forward(self, x):
        """
        x: context input (batch_size, input_dim)
        returns: mu, log_sigma of shape (batch_size, output_dim)
        """
        t1 = torch.tanh(self.fc1(x))        # W1x + B1 → tanh
        t2 = torch.sigmoid(self.fc2(x))     # W2x + B2 → sigmoid
        t = t1 * t2                         # element-wise product
        out = self.fc3(t)                   # W3·(t) + B3 → outputs both mu and log sigma
        mu, log_sigma = torch.chunk(out, 2, dim=-1)
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
        x = torch.tanh(self.fc1(z))
        x = torch.tanh(self.fc2(x))
        #x = torch.sigmoid(self.fc3(x))  #TODO CHANGED. Added sigmoid
        x = (self.fc3(x))
        return x

class TDVAE_hierachical(nn.Module):
    def __init__(self, dblock, BeliefLSTM, preprocess, decoder, input_size, processed_x_size, belief_dim=50, latent_dim_1=8, latent_dim_2=8):
        super().__init__()
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2

        self.preprocess = preprocess(input_size, processed_x_size)

        self.beliefs = BeliefLSTM(processed_x_size) #be careful with input dimension after preprocessing

        self.belief_layer2 = dblock(belief_dim, latent_dim_2)
        self.belief_layer1 = dblock(belief_dim + latent_dim_2, latent_dim_1)

        self.smoothing_layer2 = dblock(belief_dim + latent_dim_1 + latent_dim_2 + 1, latent_dim_1)
        self.smoothing_layer1 = dblock(belief_dim + latent_dim_2 + latent_dim_1 + latent_dim_2 +1, latent_dim_1)

        self.transition_layer2 = dblock(latent_dim_2 + latent_dim_1 + 1, latent_dim_1)
        self.transition_layer1 = dblock(latent_dim_2 + latent_dim_1 +latent_dim_2+ 1,latent_dim_1)

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

    def train_step(self, data, optimizer, device, beta):
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

        mu1, logvar1 = self.belief_layer1(torch.cat([bt[:, t2, :], zt2_layer2], dim=-1))
        zt2_layer1 = sample_gaussian(mu1, logvar1) #term 2

        zt2 = torch.cat([zt2_layer1, zt2_layer2], dim =-1) #term3

        mut1_layer2, logvart1_layer2 = self.belief_layer2(bt[:, t1, :])  # shape: [batch, latent_dim_2] #term 4


        #smoothing
        dt = torch.full((preprocessed_data.size(0), 1), t2 - t1, dtype=bt.dtype, device=preprocessed_data.device)
        mu_smooth_layer2, logvar_smooth_layer2 = self.smoothing_layer2(torch.cat([bt[:, t1, :],zt2,dt], dim=-1))
        zt1_layer2_smooth = sample_gaussian(mu_smooth_layer2, logvar_smooth_layer2) #term5

        mut1_layer1, logvart1_layer1 = self.belief_layer1(torch.cat([bt[:, t1, :], zt1_layer2_smooth], dim=-1)) #term 6

        mu_smooth_layer1, logvar_smooth_layer1 = self.smoothing_layer1(torch.cat([bt[:, t1, :],zt2, zt1_layer2_smooth, dt], dim=-1))
        zt1_layer1_smooth = sample_gaussian(mu_smooth_layer1, logvar_smooth_layer1) #term7

        zt1 = torch.cat([zt1_layer1_smooth, zt1_layer2_smooth ], dim = -1) #term 8

        #transition
        mu_trans_layer2 , logvar_trans_layer2 = self.transition_layer2(torch.cat([zt1,dt], dim = -1)) #term 9

        mu_trans_layer1 , logvar_trans_layer1 = self.transition_layer1(torch.cat([zt1,zt2_layer2, dt], dim = -1)) #term 10

        #decoder
        #reconstruction = self.decoder(zt2_layer1[:,:1]) #IMP!! This is only for harmonic oscillator. dont do this for MNIST remember to change
        reconstruction = self.decoder(zt2)
        bce = nn.BCEWithLogitsLoss(reduction='none')
        mse = torch.nn.MSELoss()
        target = original_data[:, t2, :]          # shape [B, 1024]

        Lx = mse(reconstruction, target)

        #Lx = bce(reconstruction, target).sum(dim=1)    TODO CHANGED. Commented
        #Lx = -torch.sum(target*torch.log(reconstruction) + (1-target)*torch.log(1-reconstruction), -1)
        #the decoder is also different for MNIST

        #calculating losses now
        L1 = kl_divergence(mu_smooth_layer2, logvar_smooth_layer2, mut1_layer2, logvart1_layer2)
        L2 = kl_divergence(mu_smooth_layer1, logvar_smooth_layer1, mut1_layer1, logvart1_layer1)
        L3 = log_normal_pdf(zt2_layer2, mu2, logvar2) - log_normal_pdf(zt2_layer2, mu_trans_layer2, logvar_trans_layer2)
        L4 = log_normal_pdf(zt2_layer1, mu1, logvar1) - log_normal_pdf(zt2_layer1, mu_trans_layer1, logvar_trans_layer1)


        #Lx = F.mse_loss(reconstruction, data[:,t2,:])#.sum(dim=-1)
        #Lx = -log_normal_pdf(data[:,t2,:],reconstruction,torch.tensor([0.0]))

        total_loss = (Lx + beta*(L1 + L2 + L3 + L4)).mean()
        reconstruction_loss = Lx.mean()
        kl_loss = (L1 + L2).mean()


        ##NEW LOSS
        # L_t1_2 = kl_divergence(mu_smooth_layer2, logvar_smooth_layer2, mut1_layer2, logvart1_layer2)
        # L_t1_1 = kl_divergence(mu_smooth_layer1, logvar_smooth_layer1, mut1_layer1, logvart1_layer1)

        # L_t2_2 = kl_divergence(mu2, logvar2, mu_trans_layer2, logvar_trans_layer2)
        # L_t2_1 = kl_divergence(mu1, logvar1, mu_trans_layer1, logvar_trans_layer1)

        # Lx = F.mse_loss(reconstruction, data[:,t2,:], reduction='none').mean(dim=-1)
        # mse = torch.nn.MSELoss()
        # Lx = mse(reconstruction,data[:,t2,:])

        # total_loss = (Lx + beta*(L_t1_2 + L_t1_1 + L_t2_2 + L_t2_1)).mean()
        # reconstruction_loss = Lx.mean()
        # kl_loss = (L_t1_2 + L_t1_1 + L_t2_2 + L_t2_1).mean()


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # === Return metrics ===
        return {
            'loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item()
        }


#Beta Scheduling

def linear_beta_schedule(epoch, max_epochs, max_beta=1.0, min_beta=0.0):
    return min_beta + (max_beta - min_beta) * (epoch / max_epochs)

def cyclic_beta_schedule(step, warmup_steps, beta_max = 1.0):
    factor = step // warmup_steps + 1
    if factor % 2 == 1: # odd means ramping up
       current_max = warmup_steps * factor
       normalized_step = 1 - (current_max - step) / warmup_steps
       beta = beta_max * normalized_step
    else:
        beta = beta_max

    return beta


# Generate data
X_train = MovingMNIST(num_sequences=num_sequences,
                      sequence_length=sequence_length, 
                      speed=speed, 
                      digit=digit)

# Create data loaders
train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

#Initializing TD-VAE Model
vae = TDVAE_hierachical(DBlock, BeliefLSTM,PreProcess, Decoder, 1024, 1024, 
                        belief_dim=belief_dim, latent_dim_1=latent_dim, latent_dim_2=latent_dim)       # construct the model
optimizer = torch.optim.Adam(vae.parameters(), lr=learn_rate)

if load_chkpt:
    chkpt = torch.load(load_chkpt_path)
    vae.load_state_dict(chkpt["model_state_dict"])
    optimizer.load_state_dict(chkpt["optimizer_state_dict"])
    start_epoch = chkpt["epoch"] + 1
else:
    start_epoch = 0


vae.to(device)        # move to GPU/CPU
vae.train()           # <- do not remove or redefine this!
vae.reset_loss_trackers()


# scheduler = torch.optim.lr_scheduler.StepLR(  #TODO Added
#     optimizer,
#     step_size=10,   # every 10 epochs
#     gamma=0.1       # multiply LR by 0.1
# )

#TODO COMMENTED 
# print(f"Running the Fist Epoch...\n")
# for batch in train_loader:
#     x = batch.to(device)
#     print(x.shape)
#     metrics = vae.train_step(x, optimizer, device, beta = 1)
#     print(metrics)


#Training loop
loss_history = []
reconstruction_history = []
kl_history = []

#Save file
save_loss_path = f"{save_dir}/loss_per_epoch.txt"
save_temp_dict_path = f"{save_dir}/checkpoint.pt"

for epoch in range(start_epoch, num_epochs):
    total_loss = 0.0
    total_rec = 0.0
    total_kl = 0.0
    num_batches = 0

    for batch in tqdm(train_loader):
        x = batch.to(device)
        #beta = linear_beta_schedule(epoch, num_epochs)
        #beta = cyclic_beta_schedule(epoch, warmup_steps = 20)
        metrics = vae.train_step(x, optimizer, device, beta=1)

        total_loss += metrics['loss']
        total_rec += metrics['reconstruction_loss']
        total_kl += metrics['kl_loss']
        num_batches += 1

    #scheduler.step()
    #print(f"Epoch {epoch}: lr = {scheduler.get_last_lr()[0]}")

    avg_loss = total_loss / num_batches
    avg_rec  = total_rec / num_batches
    avg_kl   = total_kl / num_batches

    loss_history.append(avg_loss)
    reconstruction_history.append(avg_rec)
    kl_history.append(avg_kl)

    #Save the loss terms
    with open(save_loss_path, "a") as f:
        f.write(f"{epoch}, {avg_loss}, {avg_rec}, {avg_kl}\n")

    #Save temporary checkpoint for state dict
    torch.save({
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_temp_dict_path)

    #Save checkpoint with state dict
    if (epoch % 50 == 0) and (epoch > 0):
        save_state_dict_path = f"{save_dir}/checkpoint_{epoch}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_state_dict_path)


    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Recon: {avg_rec:.4f} | KL: {avg_kl:.4f}")
