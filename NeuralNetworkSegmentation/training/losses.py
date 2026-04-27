import torch
import torch.nn as nn
from torch.nn.functional import softmax

# pass this to somewhere
def flatten(input, target, ignore_index):
    num_class = input.size(1) # gets number of classes
    if input.ndim == 5:
        input = input.permute(0, 2, 3, 4, 1).contiguous() # alters the dimensions
    else:
        input = input.permute(0, 2, 3, 1).contiguous() # alters the dimensions

    input_flatten = input.view(-1, num_class)
    target_flatten = target.view(-1)

    mask = (target_flatten != ignore_index)
    input_flatten = input_flatten[mask]
    target_flatten = target_flatten[mask]

    return input_flatten, target_flatten


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight = None, size_average=None, ignore_index = -100, reduce=None, reduction = "mean", label_smoothing = 0):
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, input, target):
        target = target.squeeze()
        return super().forward(input, target)


class SoftDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
      
    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]
            
            intersection = (input_c * target_c).sum()
            dice = (2.*intersection + self.smooth)/(input_c.sum() + target_c.sum() + self.smooth)
            
            losses.append(1-dice)
                
        losses = torch.stack(losses)
        loss = losses.mean()
        return loss


class DiceWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, alpha: float, weights: torch.Tensor, **kwargs):
        """
        alpha: how much CE contributes (0..1)
        weights: class weights tensor of shape [C]
        ce_kwargs: ignore_index, reduction, label_smoothing, etc.

        Formula:
        (WCE_loss * alpha + (1 - alpha) * Dice_loss)
        """
        super().__init__()
        self.alpha = alpha
        
        ce_kwargs = kwargs.get("ce", {})
        dice_kwargs = kwargs.get("dice", {})

        self.ce = CrossEntropyLoss(weight=weights, **ce_kwargs)
        self.dice = SoftDiceLoss(**dice_kwargs)

    def forward(self, input, target): 
        ce_loss = self.ce(input, target)
        dice_loss = self.dice(input, target)

        # print(f"CE LOSS: {ce_loss} \n DICE LOSS: {dice_loss}")

        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss


class FocalLoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=255, gamma=0.1):
        super().__init__(ignore_index=ignore_index, reduction='none')
        self.ignore_index = ignore_index
        self.gamma = gamma
      
    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)

        # If every pixel was masked out by ignore_index, the flattened tensors
        # are empty and a plain mean() would return NaN.
        if input.numel() == 0:
            return torch.tensor(0.0, device=target.device, dtype=torch.float32)

        # Changed this to compute focal loss from log-softmax directly.
        # Why: the previous version gathered probabilities with shape [N, 1]
        # and multiplied them by a cross-entropy tensor with shape [N], which
        # broadcasts incorrectly instead of applying an elementwise focal weight.
        log_prob = torch.log_softmax(input, dim=1)
        log_pt = torch.gather(log_prob, 1, target.unsqueeze(1)).squeeze(1)

        # Changed this to derive pt from log_pt for better numerical stability.
        # Why: focal loss uses pt, but working from log probabilities avoids
        # extra softmax/reduction mismatches and keeps the implementation closer
        # to the original focal-loss formula.
        pt = log_pt.exp()

        # Clamp pt away from the exact endpoints so the focal weighting stays
        # well-behaved even if upstream logits become numerically extreme.
        pt = pt.clamp(min=1e-8, max=1.0 - 1e-8)
        losses = -torch.pow(1 - pt, self.gamma) * log_pt

        return losses.mean()


# class OldFocalLoss(nn.CrossEntropyLoss):
#     def __init__(self, ignore_index=255, gamma=0.1):
#         super().__init__(reduction='none')
#         self.ignore_index = ignore_index
#         self.gamma = gamma
      
#     def forward(self, input, target):
#         input, target = flatten(input, target, self.ignore_index)
#         input_prob = torch.gather(softmax(input, dim=1), 1, target.unsqueeze(1))
#         cross_entropy = super().forward(input, target)
#         losses = torch.pow(1 - input_prob, self.gamma) * cross_entropy
#         loss = losses.mean()
#         return loss
