import time
import torch
import numpy as np
import logging
from torchmetrics.regression import ConcordanceCorrCoef
from torchmetrics import F1Score

# def accuracy_av(output, target):
#     with torch.no_grad():
#         # Assuming output and target are both torch tensors
#         device = output.device  # Get the device of the output tensor
#         concordance = ConcordanceCorrCoef(num_outputs=2).to(device)  # Move the metric to the correct device

#         # Ensure target is on the same device as output
#         target = target.to(device)

#         # Calculate the Concordance Correlation Coefficient (CCC)
#         cc = concordance(output, target)
        
#         # Calculate the average of the Arousal and Valence CCCs
#         #avg_cc = cc.mean() / batch_size
#         avg_cc = cc.mean() 

#         return avg_cc


def accuracy_av(output, target):
    with torch.no_grad():
        # Assuming output and target are both torch tensors
        device = output.device  # Get the device of the output tensor
        concordance = ConcordanceCorrCoef(num_outputs=2).to(device)  # Move the metric to the correct device

        # Ensure target is on the same device as output
        target = target.to(device)

        # Check for NaNs or Infs in output and target
        if torch.isnan(output).any() or torch.isnan(target).any():
            msg = "NaN detected in output or target"
            logging.info("%s",msg)
            return torch.tensor(0.0).to(device)
        if torch.isinf(output).any() or torch.isinf(target).any():
            msg = "INF detected in output or target"
            logging.info("%s",msg)

        # Calculate the Concordance Correlation Coefficient (CCC)
        cc = concordance(output, target)

        # Calculate the average of the Arousal and Valence CCCs
        avg_cc = cc.mean()

        return avg_cc



def accuracy_au(output, target, num_classes=12):
    with torch.no_grad():

        #USING F1 FOR AU

        # # Initialize tensors to hold true positives, false positives, false negatives
        # true_positives = torch.zeros(num_classes)
        # false_positives = torch.zeros(num_classes)
        # false_negatives = torch.zeros(num_classes)

        # # Calculate TP, FP, FN for each class
        # for i in range(num_classes):
        #     true_positives[i] = torch.sum((output == i) & (target == i))
        #     false_positives[i] = torch.sum((output == i) & (target != i))
        #     false_negatives[i] = torch.sum((output != i) & (target == i))

        # # Calculate precision, recall, and F1 for each class
        # precision = true_positives / (true_positives + false_positives + 1e-9)
        # recall = true_positives / (true_positives + false_negatives + 1e-9)
        # f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

        # # average F1 score
        # avgf1 = torch.mean(f1_scores)

        device = output.device

        f1 = F1Score(task="multiclass", num_classes=num_classes).to(device)
        target = target.to(device)
        avgf1 = f1(output, target)

        return avgf1


        
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        """Computes the precision@k for the specified values of k"""
        # print("Output: ",output)
        # print("target: ",target)
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum().item()
                res.append(correct_k*100.0 / batch_size)

            if len(res)==1:
                return res[0]
            else:
                return res



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


class Timer(object):
    """
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval
    


    