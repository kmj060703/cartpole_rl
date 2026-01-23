import torch
import random

def select_action(state, q_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            q_values = q_net(torch.FloatTensor(state))
            return torch.argmax(q_values).item()
