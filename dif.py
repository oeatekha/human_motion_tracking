"""
We are creating a simple Diffusion model for Trajectory Generation
A Trajectory we hope to predict X_t+1 given X_t, S_t, and O_t

O_t is the occupancy of other agents
S_t is the Semantic Class of the Object
X_t is the current position of the object

The Diffusion Models Goal is to Predict:

p(X_{n-1} |X_{n}, S_n, O_{n})


Data for the model is given in the following format 

N is the number of agents
T is the number of time steps

X = (N, T, 4) ~ (x, y, theta, phi)
S = (N, T, 1) ~ Semantic Class
O = (N, T, N-1, 3) 3 is the x, y, confidence of the other agents

O Must Be Placed into a latent space that can encode a given step

trans: (N, T, 3) translation vector (x, y, z) - constant bc camera is fixed
root_orient: (N, T, 3) root orientation in world coordinates 
"""

# Place Occupancy, X, and S, into a data structure that can be used for training

# X = (N, T, 4) ~ (x, y, theta, phi)
# S = (N, T, 1) ~ Semantic Class
# O = (N, T, N-1, 3) 3 is the x, y, confidence of the other agents

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def getOccupancy(trans, heading, pedestrians_trans):
    # FOV Definitions
    paraFOV = {'angle_range': 8, 'resolution': 1.0}  # Half angle for easier calculations
    biFOV = {'angle_range': 60, 'resolution': 0.85}
    perFOV = {'angle_range': 120, 'resolution': 0.7}
    max_distance = 2  # Maximum distance to consider

    # Convert inputs to numpy arrays
    c_pose = np.array(trans[:2])
    pedestrians_pose = np.array(pedestrians_trans[:, :2])  # Ensure it's an array for operations
    pedestrian_vector = c_pose - pedestrians_pose
    heading_rad = -(np.pi - np.arctan2(heading[1], heading[0]))

    # Heading vector
    heading_vector = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    norms = np.linalg.norm(pedestrian_vector, axis=1)
    unitVectors = pedestrian_vector / norms[:, np.newaxis]
    
    angles_rad = np.arccos(np.clip(np.dot(unitVectors, heading_vector), -1, 1))
    angles_deg = np.abs(np.rad2deg(angles_rad))  # Convert angle to degrees

    # Initialize O_n with zeros
    O_n = np.zeros((len(pedestrians_pose), 3))
    
    for i, angl in enumerate(angles_deg):

        if  paraFOV['angle_range'] < angl and angl <= paraFOV['angle_range']:
            resolution = paraFOV['resolution']
        elif perFOV['angle_range'] < angl and angl <= biFOV['angle_range']:
            resolution = biFOV['resolution']
        elif biFOV['angle_range'] < angl and angl <= perFOV['angle_range']:
            resolution = perFOV['resolution']
        else:
            resolution = 0  # Keep as zero if not in FOV

        # Update O_n with [x, y, resolution]
        # print(i)
        # print(O_n.shape)
        O_n[i] = np.array([pedestrians_pose[i, 0], pedestrians_pose[i, 1], resolution])

    # print the angles of the last frame
    # print(angles_deg)

    return O_n

def getOccupancyFrames(trans, heading):
    N, T, _ = trans.shape
    # Initialize the occupancy array with zeros for each pedestrian, for each time step
    O_r = np.zeros((N, T, N-1, 3))

    for n in range(N):
        for t in range(T):
            cur_trans = trans[n, t]
            cur_heading = heading[n, t]
            pedestrian_trans = np.delete(trans, n, axis=0)[:, t, :]  # Exclude current pedestrian
            O_n = getOccupancy(cur_trans, cur_heading, pedestrian_trans)
            # Insert O_n into O_r, adjusting for the exclusion of the current pedestrian
            O_r[n, t, :, :] = O_n

    return O_r

def createData(transformations, root_orient):
    occupancy  = getOccupancyFrames(transformations, root_orient)
    N, T, _ = transformations.shape

    X = np.zeros((N, T, 4))
    S = np.zeros((N, T, 1))
    O = np.zeros((N, T, N-1, 3))

    for n in range(N):
        for t in range(T):
            X[n, t, 0] = transformations[n, t, 0]
            X[n, t, 1] = transformations[n, t, 1]
            X[n, t, 2] = root_orient[n, t, 0]
            X[n, t, 3] = root_orient[n, t, 1]
            S[n, t, 0] = 0
            if n == 9:
                S[n, t, 0] = 1
            O[n, t, :, :] = occupancy[n, t, :, :]

    return X, S, O


class TrajectoryDataset(Dataset):
    def __init__(self, X, S, O):
        N, T, _ = X.shape
        # Flatten the data across time steps, resulting in (N*T, ...) shape
        self.X = X.reshape(-1, 4)  # Flatten X to have shape [(N*T), 4]
        self.S = S.reshape(-1, 1)  # Flatten S to have shape [(N*T), 1]
        # O needs special handling to adjust shape and include correct occupancy information
        self.O = O.reshape(N*T, -1, 3)  # Flatten O and adjust shape accordingly
        
        # Prepare target positions (X_t+1)
        self.Y = np.roll(X, -1, axis=1).reshape(-1, 4)[:-N]  # Roll and reshape X to align X_t+1, then remove the last N elements to match shapes
        
        # Remove the last timestep for each agent from X, S, and O to match Y's shape
        self.X = self.X[:-N]
        self.S = self.S[:-N]
        self.O = self.O[:-N]

    def __len__(self):
        return len(self.X)  # Updated length after flattening and adjusting

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor(self.S[idx], dtype=torch.float), torch.tensor(self.O[idx], dtype=torch.float), torch.tensor(self.Y[idx], dtype=torch.float)

# Assuming your createData function returns correctly shaped X, S, O
X, S, O = createData(transformations, root_orient)

# Create the Dataset
dataset = TrajectoryDataset(X, S, O)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)


# Define the Occupancy Encoder

"""
We are creating a simple Diffusion model for Trajectory Generation
A Trajectory we hope to predict X_t+1 given X_t, S_t, and O_t

O_t is the occupancy of other agents
S_t is the Semantic Class of the Object
X_t is the current position of the object

The Diffusion Models Goal is to Predict:

p(X_{n-1} |X_{n}, S_n, O_{n})


Data for the model is given in the following format 

N is the number of agents
T is the number of time steps

X = (N, T, 4) ~ (x, y, theta, phi)
S = (N, T, 1) ~ Semantic Class
O = (N, T, N-1, 3) 3 is the x, y, confidence of the other agents

O Must Be Placed into a latent space that can encode a given step

trans: (N, T, 3) translation vector (x, y, z) - constant bc camera is fixed
root_orient: (N, T, 3) root orientation in world coordinates 
"""

# Place Occupancy, X, and S, into a data structure that can be used for training

# X = (N, T, 4) ~ (x, y, theta, phi)
# S = (N, T, 1) ~ Semantic Class
# O = (N, T, N-1, 3) 3 is the x, y, confidence of the other agents

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def createData(transformations, root_orient):
    occupancy  = getOccupancyFrames(transformations, root_orient)
    N, T, _ = transformations.shape

    X = np.zeros((N, T, 4))
    S = np.zeros((N, T, 1))
    O = np.zeros((N, T, N-1, 3))

    for n in range(N):
        for t in range(T):
            X[n, t, 0] = transformations[n, t, 0]
            X[n, t, 1] = transformations[n, t, 1]
            X[n, t, 2] = root_orient[n, t, 0]
            X[n, t, 3] = root_orient[n, t, 1]
            S[n, t, 0] = 0
            if n == 9:
                S[n, t, 0] = 1
            O[n, t, :, :] = occupancy[n, t, :, :]

    return X, S, O


import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryHistoryDataset(Dataset):
    def __init__(self, X, S, O, max_history_len=10):
        self.max_history_len = max_history_len
        N, T, _ = X.shape

        # Preparing the inputs (X_history) and outputs (Y_next)
        # X_history will be a list of tensors, each tensor representing the history of positions up to time t
        self.X_history = []
        self.S_t = []  # Semantic class at time t
        self.O_t = []  # Occupancy at time t
        self.Y_next = []  # Position at time t+1

        # Iterate over each agent and time step, collecting the necessary data
        for n in range(N):
            for t in range(1, T):  # Start from 1 since we need at least one previous step
                history_len = min(t, max_history_len)
                prev_positions = X[n, t-history_len:t, :].reshape(-1, 4)
                self.X_history.append(prev_positions)
                
                self.S_t.append(S[n, t, :])
                self.O_t.append(O[n, t, :].reshape(-1, 3))  # Adjust O_t shape if needed
                
                if t < T-1:  # Ensure there is a next step to predict
                    self.Y_next.append(X[n, t+1, :])
        
        # Convert lists to tensors for PyTorch compatibility
        self.X_history = [torch.tensor(pos, dtype=torch.float) for pos in self.X_history]
        self.S_t = torch.tensor(self.S_t, dtype=torch.float)
        self.O_t = [torch.tensor(occ, dtype=torch.float) for occ in self.O_t]
        self.Y_next = torch.tensor(self.Y_next, dtype=torch.float)

    def __len__(self):
        return len(self.Y_next)

    def __getitem__(self, idx):
        # For sequences, directly return the tensors without converting, as they are already tensors
        return self.X_history[idx], self.S_t[idx], self.O_t[idx], self.Y_next[idx]

# Assuming your createData function returns correctly shaped X, S, O
X, S, O = createData(transformations, root_orient)

# Create the Dataset with historical data
dataset = TrajectoryHistoryDataset(X, S, O)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size of 1 due to variable sequence lengths


class OccupancyEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OccupancyEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class TrajectoryAndOccupancyEncoder(nn.Module):
    def __init__(self, occupancy_embed_size, num_random_predictions=5):
        super().__init__()
        channel_in = 6  # Assuming 4 channels for X and 2 for encoded O
        channel_out = 32
        dim_kernel = 3
        self.dim_embedding_key = 256
        self.num_random_predictions = num_random_predictions
        
        # Spatial convolution operates on the X input
        self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        
        # Adjusting the input size for GRU to include occupancy embedding
        self.temporal_encoder = nn.GRU(channel_out + occupancy_embed_size, self.dim_embedding_key, 1, batch_first=True)
        
        # This linear layer maps the GRU output state to k random predictions
        self.prediction_head = nn.Linear(self.dim_embedding_key, num_random_predictions * 2)  # Assuming each prediction is 2D (x, y)
        
        self.relu = nn.ReLU()
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.zeros_(self.spatial_conv.bias)
        # Reset parameters for GRU and prediction head if necessary
        for name, param in self.temporal_encoder.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.kaiming_normal_(self.prediction_head.weight)
        nn.init.zeros_(self.prediction_head.bias)

    def forward(self, X, encoded_O, X_lengths):
        # Assume X is padded to the max length in the batch
        # encoded_O: b, occupancy_embed_size
        # X_lengths: Original lengths of X sequences before padding

        X_t = torch.transpose(X, 1, 2)  # (batch, channels, T)
        X_after_spatial = self.relu(self.spatial_conv(X_t))
        
        # Expand encoded_O to match the temporal dimension of X
        encoded_O_expanded = encoded_O.unsqueeze(1).expand(-1, X.size(1), -1)
        X_combined = torch.cat([X_after_spatial, encoded_O_expanded.transpose(1, 2)], dim=1)
        X_combined = torch.transpose(X_combined, 1, 2)  # (batch, T, channels)
        
        # Pack the sequences
        packed_X = pack_padded_sequence(X_combined, X_lengths, batch_first=True, enforce_sorted=False)
        
        # Temporal encoding with GRU
        packed_output, state_x = self.temporal_encoder(packed_X)
        output_x, _ = pad_packed_sequence(packed_output, batch_first=True)  # Optionally use output_x
        
        state_x = state_x.squeeze(0)  # Assuming single layer GRU
        predictions = self.prediction_head(state_x)
        predictions = predictions.reshape(-1, self.num_random_predictions, 2)  # (batch, k, 2)
        
        return predictions



# Make A Small Training Loop to Train the Model, use a MSE Loss Function

class TrajectoryAndOccupancyEncoderTrainingLoop:
    def __init__(self, model, occupancy_encoder, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.o_encoder = occupancy_encoder
        

    def train(self, dataloader, num_epochs):
        self.model.train()
        self.model.to(self.device)

        for epoch in range(num_epochs):
            for batch in dataloader:
                X, S, O, Y = [b.to(self.device) for b in batch]

                # Assuming X is a tensor of shape (batch, T, 4)
                # Assuming S is a tensor of shape (batch, T, 1)
                # Assuming O is a list of tensors, each tensor of shape (batch, N-1, 3)
                # Assuming Y is a tensor of shape (batch, 4)

                # Assuming X_lengths is a tensor of shape (batch,) containing the original lengths of X sequences
                X_lengths = torch.sum(torch.sum(X, dim=-1) != 0, dim=-1)

                # Encode O
                encoded_O = []
                for o in O:
                    encoded_O.append(self.o_encoder(o))
                encoded_O = torch.stack(encoded_O, dim=1)  # Combine the list of tensors into a single tensor

                # Predict
                predictions = self.model(X, encoded_O, X_lengths)

                # Calculate loss
                loss = self.loss_fn(predictions, Y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


