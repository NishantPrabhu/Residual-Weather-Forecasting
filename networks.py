
""" 
Network definitions.
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F 


class BasicBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        horizon = config['horizon']
        lookback = config['lookback_horizon_ratio'] * horizon
        fc_dim = config['fc_dim']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fc_stack = nn.Sequential(
            nn.Linear(lookback, fc_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim, bias=True),
            nn.ReLU(inplace=True)
        ).to(self.device)

        self.fc_b = nn.Linear(fc_dim, fc_dim, bias=False)
        self.fc_f = nn.Linear(fc_dim, fc_dim, bias=False)
        self.expansion_b = nn.Linear(fc_dim, lookback, bias=True)
        self.expansion_f = nn.Linear(fc_dim, horizon, bias=True)

    def forward(self, x):
        ''' x has size (bs, input_length) '''
        if x.dim() > 2:
            x = x.squeeze(-1)
        x = self.fc_stack(x)
        y_back = self.expansion_b(self.fc_b(x))
        y_fwd = self.expansion_f(self.fc_f(x))
        return y_back, y_fwd
         

class ResidualStack(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.horizon = config['horizon']
        self.num_blocks = config['blocks_per_stack']
        self.blocks = nn.ModuleList([BasicBlock(config) for _ in range(self.num_blocks)])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        stack_forecast = torch.FloatTensor(x.size(0), self.horizon).zero_().to(self.device)
        for i in range(self.num_blocks):
            x_back, x_fwd = self.blocks[i](x)
            x = x - x_back
            stack_forecast += x_fwd
        return x, stack_forecast


class Forecaster(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_stacks = config['num_stacks']
        self.horizon = config['horizon']
        self.stacks = nn.ModuleList([ResidualStack(config) for _ in range(self.num_stacks)])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, return_stack_frc=False):
        forecast = torch.FloatTensor(x.size(0), self.horizon).zero_().to(self.device)
        stack_frc_collect = {}
        for i in range(self.num_stacks):
            x, stack_frc = self.stacks[i](x)
            forecast += stack_frc
            stack_frc_collect[f'stack_{i}'] = stack_frc.detach().cpu().numpy().reshape(-1,).tolist()
        
        if return_stack_frc:
            return forecast, stack_frc_collect
        else:
            return forecast