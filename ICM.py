import torch
import numpy as np
import torch.nn as nn


class ICM(nn.Module):
    # Add swish activation
    def __init__(self, state_dim=172, encoding_size=256, num_layers=2, action_dim=5, activation=nn.Tanh()):
        super().__init__()
        self.act_dim = action_dim
        # Encoder
        layers = list()
        layers.append(nn.Linear(172, encoding_size))
        nn.init.normal_(layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / state_dim))
        layers.append(activation)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(encoding_size, encoding_size))
            nn.init.normal_(layers[-1].weight, mean=0.0, std=np.sqrt(1.0 / encoding_size))
            layers.append(activation)

        self.encoder = nn.Sequential(*layers)

        # Inverse model
        self.fc_i1 = nn.Linear(encoding_size * 2, 256)
        self.act_i1 = activation
        self.fc_i2 = nn.Linear(256, action_dim)

        # Forward model
        self.fc_f1 = nn.Linear(encoding_size + action_dim, 256)
        self.act_f1 = activation
        self.fc_f2 = nn.Linear(256, encoding_size)

    def forward(self, act, curr_obs, next_obs, mask):
        # Inverse model
        curr_enc = self.encoder(curr_obs)
        next_enc = self.encoder(next_obs)
        out = self.fc_i1(torch.cat((curr_enc, next_enc), dim=2))
        out = self.act_i1(out)
        pred_act = torch.transpose(self.fc_i2(out), 1, 2)
        inv_loss = (nn.CrossEntropyLoss(reduction='none')(pred_act, act) * mask).mean()

        # Forward model
        one_hot_act = nn.functional.one_hot(act, num_classes=self.act_dim)
        out = self.fc_f1(torch.cat((one_hot_act.float(), curr_enc), dim=2))
        out = self.act_f1(out)
        pred_next_enc = self.fc_f2(out)

        # Intrinsic reward
        intr_reward = 0.5 * nn.MSELoss(reduction='none')(pred_next_enc, next_enc)
        intr_reward = intr_reward.mean(dim=2) * mask

        # Forward loss
        forw_loss = intr_reward.mean()
        return intr_reward, inv_loss, forw_loss