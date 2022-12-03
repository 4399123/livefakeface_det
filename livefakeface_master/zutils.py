
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




def warmup_learnig_rate(init_lr, warmup_lr, warmup_epochs,
                        iteration, epoch, dataset_len):
    lr = warmup_lr + (init_lr - warmup_lr) * \
         float(iteration + epoch * dataset_len) / (warmup_epochs * dataset_len)
    return lr


def adjust_learning_rate_cosine(epoch, iteration, dataset_len,
                                epochs, warmup_epochs):
    total_iter = (epochs - warmup_epochs) * dataset_len
    current_iter = iteration + (epoch - warmup_epochs) * dataset_len

    lr = 1 / 2 * (np.cos(np.pi * current_iter / total_iter) + 1)

    return lr


def adjust_learning_rate_linear(epoch, iteration, dataset_len,
                                epochs, warmup_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    total_iter = (epochs - warmup_epochs) * dataset_len
    current_iter = iteration + (epoch - warmup_epochs) * dataset_len

    lr = 1 - current_iter / total_iter

    return lr


def adjust_learning_rate_default(epoch, iteration, dataset_len,
                                 epochs, warmup_epochs):
    """
    LR schedule that should yield 76% converged accuracy
    with batch size 256"""
    factor = epoch // 30

    lr = 0.1 ** factor

    return lr

def adjust_learning_rate(optimizer, epoch, iteration, lr_decay_type,
                         epochs, train_len, warmup_lr, warmup_epochs,init_lr):
    if epoch < warmup_epochs:
        for param_group in optimizer.param_groups:
            lr = warmup_learnig_rate(init_lr=init_lr,
                                     warmup_lr=warmup_lr,
                                     warmup_epochs=warmup_epochs,
                                     epoch=epoch,
                                     iteration=iteration,
                                     dataset_len=train_len)
            param_group['lr'] = lr
    else:
        if lr_decay_type == 'cos':
            lr_function = adjust_learning_rate_cosine
        elif lr_decay_type == 'linear':
            lr_function = adjust_learning_rate_linear
        elif lr_decay_type == 'default':
            lr_function = adjust_learning_rate_default
        else:
            raise ValueError("Unknown lr decay type {}."
                             .format(lr_decay_type))

        lr_factor = lr_function(epoch=epoch,
                                iteration=iteration,
                                dataset_len=train_len,
                                epochs=epochs,
                                warmup_epochs=warmup_epochs)

        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr * lr_factor

    return optimizer.param_groups[0]['lr']

#标签平滑损失
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self,e=0.1):
        super(LabelSmoothingCrossEntropyLoss,self).__init__()
        self.e=e

    def forward(self,output,target):
        num_classes=output.size()[-1]
        log_preds=F.log_softmax(output,dim=-1)
        loss=(-log_preds.sum(dim=-1)).mean()
        nll=F.nll_loss(log_preds,target)
        final_loss=self.e*loss/num_classes+(1-self.e)*nll
        return  final_loss

class SoftTargetCrossEntropy(nn.Module):
    def forward(self, x, target):
        target=F.one_hot(target).float()
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

