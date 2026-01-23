import torch
import torch.nn as nn # 신경망 레이어, 활성함수, 손실함수 같은 것들이 있음

class DQN(nn.Module): #DQN = Q 함수를 근사하는 신경망
    def __init__(self, state_dim, action_dim): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
