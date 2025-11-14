import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import gymnasium as gym
from collections import deque
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first




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
        print("Is image space -- from feature extractor:", is_image_space(image_space))
        print("Is image space First -- from feature extractor:", is_image_space_channels_first(image_space))

        
    # def __init__(self, observation_space, features_dim: int = 512):
    #     super().__init__(observation_space, features_dim)
        # Image CNN
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
        state_seq = observations["state"]  # (B, seq_len, state_dim) or (B, state_dim)
        
        # Add batch & sequence dimension if missing
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

    # def forward(self, observations):
    #     # --- Image branch ---
    #     img = observations["image"]
    #     if img.ndim == 3:  # add batch dim
    #         img = img.unsqueeze(0)
    #     img = img.float() / 255.0
    #     # print(f"[DEBUG] CNN input shape feature exctractor [IN] forward(): {img.shape}")
    #     # print("Image --feature extractor [IN] forward() mean/std:", img.mean(), img.std())
    #     cnn_out = self.cnn(img)  # shape: (B, cnn_feat)

    #     # --- State sequence branch ---
    #     state_seq = observations["state"]  # expected shape: (B, seq_len, state_dim)
    #     if state_seq.ndim == 2:
    #         # Only current step provided: pad with zeros for previous steps
    #         B, state_dim = state_seq.shape
    #         pad = state_seq.new_zeros((B, self.seq_len - 1, state_dim))
    #         state_seq = th.cat([pad, state_seq.unsqueeze(1)], dim=1)  # shape: (B, seq_len, state_dim)

    #     elif state_seq.ndim == 3:
    #         B, S, state_dim = state_seq.shape
    #         if S < self.seq_len:
    #             pad = state_seq.new_zeros((B, self.seq_len - S, state_dim))
    #             state_seq = th.cat([pad, state_seq], dim=1)

    #     lstm_out, _ = self.lstm(state_seq)       # shape: (B, seq_len, hidden)
    #     lstm_feat = lstm_out[:, -1, :]           # take last time step

    #     # --- Concatenate and final linear ---
    #     concat = th.cat([cnn_out, lstm_feat], dim=1)
    #     return self.linear(concat)


    # def forward(self, observations):
    #     img = observations["image"]
    #     # state_seq = observations["state"]
    #     state = observations["state"]  # shape: (batch, state_dim)
    #     if len(self.state_buffer) < self.seq_len:
    #         # initialize with repeated first state
    #         for _ in range(self.seq_len):
    #             self.state_buffer.append(state)
    #     else:
    #         self.state_buffer.append(state)


    #     # Ensure batch dimension
    #     if img.ndim == 3:
    #         img = img.unsqueeze(0)
    #     if state_seq.ndim == 2:
    #         state_seq = state_seq.unsqueeze(0)

    #     img = img.float() / 255.0
    #     cnn_out = self.cnn(img)

    #     lstm_out, _ = self.lstm(state_seq)  # lstm_out shape: (B, seq_len, hidden)
    #     lstm_feat = lstm_out[:, -1, :]      # last step

    #     concat = th.cat([cnn_out, lstm_feat], dim=1)
    #     return self.linear(concat)

        # # CNN for image input
        # # n_input_channels = observation_space["image"].shape[2]
        # n_input_channels = observation_space["image"].shape[0]  # C = 3
        # #print("CNN input shape:", observation_space["image"].shape)

        # #print(f'n_input_channels: {n_input_channels}')
        # state_dim = observation_space["state"].shape[0]
        # #print(f'state_dim: {state_dim}')

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        # #print('self.cnn is done')
        # # assert img.ndim == 4, f"Image tensor must be 4D, got shape {img.shape}"
        # # assert img.shape[1] == 3, f"Expected 3 channels, got {img.shape[1]}"

        # # compute CNN output size
        # with th.no_grad():
        #     # produce a sample with batch dim that matches expected channels (C, H, W)
        #     sample = observation_space["image"].sample()  # probably shape (C,H,W)
        #     sample = th.as_tensor(sample[None]).float()   # (1, C, H, W)
        #     if sample.ndim == 4 and sample.shape[1] not in (1,3,4) and sample.shape[-1] in (1,3,4):
        #         # if sample is (1, H, W, C) -> permute
        #         sample = sample.permute(0, 3, 1, 2)
        #     n_flatten = self.cnn(sample).shape[1]

        # with th.no_grad():
        #     sample = th.as_tensor(observation_space["image"].sample()[None]).float()
        #     n_flatten = self.cnn(sample).shape[1]
            #n_flatten = self.cnn(sample.permute(0, 3, 1, 2)).shape[1] ## headache
        #print('th.no_grad() is done')

        # MLP for low-dimensional state vector
        # self.mlp = nn.Sequential(
        #     nn.Linear(state_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU()
        # )
        # #print('self.mlp  is done')

        # # Combine CNN + MLP outputs
        # self.linear = nn.Sequential(
        #     nn.Linear(n_flatten + 64, 512),
        #     nn.ReLU(),
        # )
    
    # def forward(self, observations):
    #     img = observations["image"]
    #     state = observations["state"]
    #     # ensure batch dimension
    #     if img.ndim == 3:
    #         img = img.unsqueeze(0)
    #     # expected shape: (B, C, H, W)
    #     # if it's (B, H, W, C) -> permute
    #     if img.shape[-1] == 3 and img.shape[1] != 3:
    #         img = img.permute(0, 3, 1, 2)
    #     img = img.float() / 255.0
    #     cnn_out = self.cnn(img)
    #     # print(f"[DEBUG] CNN output shape: {cnn_out.shape}")
    #     mlp_out = self.mlp(state)
    #     concat = th.cat((cnn_out, mlp_out), dim=1)
    #     return self.linear(concat)

