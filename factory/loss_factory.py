import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from munkres import Munkres
import numpy as np
import time

torch.manual_seed(0)

def MaskedNLL(target, probs, balance_weights=None):
    # adapted from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, ) which contains the index of the true
            class for each corresponding step.
        probs: A Variable containing a FloatTensor of size
            (batch, num_classes) which contains the
            softmax probability for each class.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    log_probs = torch.log(probs)

    if balance_weights is not None:
        balance_weights = balance_weights.cuda()
        log_probs = torch.mul(log_probs, balance_weights)

    losses = -torch.gather(log_probs, dim=1, index=target)
    return losses.squeeze()

def StableBalancedMaskedBCE(target, out, balance_weight = None):
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    if balance_weight is None:
        num_positive = target.sum()
        num_negative = (1 - target).sum()
        total = num_positive + num_negative
        balance_weight = num_positive / total

    max_val = (-out).clamp(min=0)
    # bce with logits
    loss_values =  out - out * target + max_val + ((-max_val).exp() + (-out - max_val).exp()).log()
    loss_positive = loss_values*target
    loss_negative = loss_values*(1-target)
    losses = (1-balance_weight)*loss_positive + balance_weight*loss_negative

    return losses.squeeze()

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

def batch_to_var(inputs, targets, device):
    """
    Turns the output of DataLoader into data and ground truth to be fed
    during training
    """
    x = Variable(inputs,requires_grad=False)
    y_mask = Variable(targets[:,:,:-3].float(),requires_grad=False)
    y_class = Variable(targets[:,:,-3].long(),requires_grad=False)
    sw_mask = Variable(targets[:,:,-2],requires_grad=False)
    sw_class = Variable(targets[:,:,-1],requires_grad=False)
    return x.cuda(device), y_mask.cuda(device), y_class.cuda(device), sw_mask.cuda(device), sw_class.cuda(device)


def softIoU(out, target, e=1e-6):

    """
    Args:
        target:
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    # print(out.shape, target.shape)
    out = torch.sigmoid(out)
    # clamp values to avoid nan loss
    #out = torch.clamp(out,min=e,max=1.0-e)
    #target = torch.clamp(target,min=e,max=1.0-e)

    num = (out*target).sum(1,True)
    # print(num.shape)
    den = (out+target-out*target).sum(1,True) + e
    # print(num.shape)
    iou = num / den
    # set iou to 0 for masks out of range
    # this way they will never be picked for hungarian matching
    cost = (1 - iou)

    return cost.squeeze()

class softIoULoss(nn.Module):

    def __init__(self):
        super(softIoULoss,self).__init__()
    def forward(self, y_pred, y_true, sw):
        costs = softIoU(y_pred,y_true).view(-1,1)
        print(np.unique(costs.detach().cpu().numpy()))
        costs = torch.mean(torch.masked_select(costs,sw.byte()))
        print(torch.mean(costs).shape)
        return torch.mean(costs)


# def match(masks, classes, overlaps):
#     """
#     Args:
#         masks - list containing [true_masks, pred_masks], both being [batch_size,T,N]
#         classes - list containing [true_classes, pred_classes] with shape [batch_size,T,]
#         overlaps - [batch_size,T,T] - matrix of costs between all pairs
#     Returns:
#         t_mask_cpu - [batch_size,T,N] permuted ground truth masks
#         t_class_cpu - [batch_size,T,] permuted ground truth classes
#         permute_indices - permutation indices used to sort the above
#     """
#
#     overlaps = (overlaps.data).cpu().numpy().tolist()
#     m = Munkres()
#
#     t_mask, p_mask = masks
#     t_class, p_class = classes
#
#     # get true mask values to cpu as well
#     t_mask_cpu = (t_mask.data).cpu().numpy()
#     t_class_cpu = (t_class.data).cpu().numpy()
#     # init matrix of permutations
#     permute_indices = np.zeros((t_mask.size(0),t_mask.size(1)),dtype=int)
#     # we will loop over all samples in batch (must apply munkres independently)
#     for sample in range(p_mask.size(0)):
#         # get the indexes of minimum cost
#         indexes = m.compute(overlaps[sample])
#         for row, column in indexes:
#             # put them in the permutation matrix
#             permute_indices[sample,column] = row
#
#         # sort ground according to permutation
#         t_mask_cpu[sample] = t_mask_cpu[sample,permute_indices[sample],:]
#         t_class_cpu[sample] = t_class_cpu[sample,permute_indices[sample]]
#     return t_mask_cpu, t_class_cpu, permute_indices


""" 
    Class that defines the Dice Loss function.
"""
class MaskedNLLLoss(nn.Module):
    def __init__(self, balance_weight=None):
        super(MaskedNLLLoss,self).__init__()
        self.balance_weight=balance_weight
    def forward(self, y_true, y_pred, sw):
        costs = MaskedNLL(y_true,y_pred, self.balance_weight).view(-1,1)
        # costs = torch.mean(torch.masked_select(costs,sw.byte()))
        costs = torch.masked_select(costs,sw.byte())
        return costs

class MaskedBCELoss(nn.Module):

    def __init__(self,balance_weight=None):
        super(MaskedBCELoss,self).__init__()
        self.balance_weight = balance_weight
    def forward(self, y_true, y_pred,sw):
        costs = StableBalancedMaskedBCE(y_true,y_pred,self.balance_weight).view(-1,1)
        costs = torch.masked_select(costs,sw.byte())
        return costs


class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def dice_coef(self, y_pred, y_true):
        pred_probs = torch.sigmoid(y_pred)
        y_true_f = y_true.view(-1)
        y_pred_f = pred_probs.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

    def forward(self, y_pred, y_true):
        return -self.dice_coef(y_pred, y_true)


""" 
    Class that defines the Root Mean Square Loss function.
"""


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


"""
    Class that defines the Cross Entropy Loss Function
"""


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return -torch.mean(torch.sum(y_true * torch.log(F.softmax(y_pred, dim=1)), dim=1))


"""
    Class that defines the Cross Entropy Loss Function
"""


class WCELoss(nn.Module):
    def __init__(self):
        super(WCELoss, self).__init__()

    def forward(self, y_pred, y_true, weights):
        y_true = y_true / (y_true.sum(2).sum(2, dtype=torch.float).unsqueeze(-1).unsqueeze(-1))
        y_true[y_true != y_true] = 0.0
        y_true = torch.sum(y_true, dim=1, dtype=torch.float).unsqueeze(1)
        y_true = y_true * weights.to(torch.float)
        old_range = torch.max(y_true) - torch.min(y_true)
        new_range = 100 - 1
        y_true = (((y_true - torch.min(y_true)) * new_range) / old_range) + 1
        return -torch.mean(torch.sum(y_true * torch.log(F.softmax(y_pred, dim=1)), dim=1))
