import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import gymnasium as gym
from collections import deque
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first

# custom feature extractor class used in trainig
class CarlaSequentialFeatureExtractor(BaseFeaturesExtractor):
    """
    Extracts features from Dict obs = { "image": (84x84x3), "state": (11,) }
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
        Multiple Inputs and Dictionary Observations
    """
    def __init__(
            self, 
            observation_space: gym.spaces.Dict, 
            features_dim: int = 512, 
            state_hidden: int = 64, 
            seq_len: int = 8 
            ):
        super().__init__(observation_space, features_dim)
        self.seq_len = seq_len
        self.state_buffer = deque(maxlen=self.seq_len)
        image_space = observation_space["image"]
        # sanity checks
        print("Is image space -- from feature extractor:", is_image_space(image_space))
        print("Is image space First -- from feature extractor:", is_image_space_channels_first(image_space))

        n_input_channels = observation_space["image"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample_img = th.as_tensor(observation_space["image"].sample()[None]).float()
            # print("Image --feature extractor [BEFORE] forward() mean/std:", sample_img.mean(), sample_img.std())            
            cnn_out_dim = self.cnn(sample_img).shape[1]
           
        # State LSTM 
        state_dim = observation_space["state"].shape[1]
        # print("State shape --feature extractor:", state_dim.shape)
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=state_hidden, batch_first=True)

        # Combine CNN + LSTM outputs
        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim + state_hidden, features_dim),
            nn.ReLU()
        )
    def forward(self, observations):
        """
        observations: dict with keys:
            - 'image': (B, C, H, W) uint8 tensor
            - 'state': (B, seq_len, state_dim) float tensor
        returns:
            features: (B, features_dim) float tensor
        """
        # --- Image branch ---
        img = observations["image"]  # (B, C, H, W)
        if img.ndim == 3:
            img = img.unsqueeze(0)  # add batch dim if missing

        img = img.float() / 255.0  # normalize to 0..1
        cnn_out = self.cnn(img)    # (B, cnn_out_dim)

        # --- State sequence branch ---
        state_seq = observations["state"]  # (B, seq_len, state_dim)
        
        # Add batch & sequence dimension if missing
        # was suspicious about the image data due to learning was not good
        # tried to make sure the data is in correct shape and type
        if state_seq.ndim == 2:  # only current step
            B, state_dim = state_seq.shape
            pad = state_seq.new_zeros((B, self.seq_len - 1, state_dim))
            state_seq = th.cat([pad, state_seq.unsqueeze(1)], dim=1)
        
        elif state_seq.ndim == 3:
            B, S, state_dim = state_seq.shape
            if S < self.seq_len:
                pad = state_seq.new_zeros((B, self.seq_len - S, state_dim))
                state_seq = th.cat([pad, state_seq], dim=1)  # pad at beginning

        # LSTM forward
        lstm_out, _ = self.lstm(state_seq)  # (B, seq_len, hidden)
        lstm_feat = lstm_out[:, -1, :]      # last time step

        # --- Normalize LSTM features per batch ---
        lstm_mean = lstm_feat.mean(dim=1, keepdim=True)
        lstm_std = lstm_feat.std(dim=1, keepdim=True) + 1e-6
        lstm_feat = (lstm_feat - lstm_mean) / lstm_std

        # --- Concatenate CNN + LSTM ---
        concat = th.cat([cnn_out, lstm_feat], dim=1)

        # --- Final linear layer ---
        return self.linear(concat)  # (B, features_dim)