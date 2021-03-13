
""" 
Evaluation metrics 
"""

import torch 
import torch.nn.functional as F 


def MAPE(output, target):
    """
    Mean absolute percentage error
    """
    percent_errs = torch.abs((output - target)/target)
    return float(percent_errs.mean().detach().cpu().numpy())