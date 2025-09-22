import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
import numpy as np

class KLLoss(nn.Module):
    def __init__(self,reduction = 'batchmean'):
        super(KLLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction=reduction)

    def forward(self,y_pred,y_true):
        y_pred = rearrange(y_pred,'b h w -> b (h w)')
        y_true = rearrange(y_true, 'b h w -> b (h w)')
        y_pred = F.log_softmax(y_pred,dim=1)
        y_true = F.softmax(y_true, dim=1)
        out = self.kl(y_pred, y_true)
        return out

class CCLoss(nn.Module):
    def __init__(self):
        super(CCLoss, self).__init__()

    def forward(self,y_pred,y_true):
        y_pred = rearrange(y_pred, 'b h w -> b (h w)')
        y_true = rearrange(y_true, 'b h w -> b (h w)')
        """
        计算均值
        """
        mean_y_pred = torch.mean(y_pred,dim=1).unsqueeze(1)
        mean_y_true = torch.mean(y_true,dim=1).unsqueeze(1)
        # 计算协方差
        cov_xy = torch.mean((y_pred - mean_y_pred) * (y_true - mean_y_true),dim=1)
        # 计算标准差
        std_y_pred = torch.std(y_pred,dim=1)
        std_y_true = torch.std(y_true,dim=1)
        #计算cc
        cc_output = cov_xy / (std_y_pred * std_y_true)
        return cc_output.mean()

class SIMLoss(nn.Module):
    def __init__(self):
        super(SIMLoss, self).__init__()

    def forward(self,y_pred,y_true):
        y_pred = rearrange(y_pred, 'b h w -> b (h w)')
        y_true = rearrange(y_true, 'b h w -> b (h w)')
        y_pred = y_pred / y_pred.sum(dim=1, keepdim=True)
        y_true = y_true / y_true.sum(dim=1, keepdim=True)
        sims = torch.min(y_pred, y_true).sum(dim=1)
        return sims.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, output, target):
        num_classes = output.size(1)
        assert len(self.alpha) == num_classes, \
            'Length of weight tensor must match the number of classes'
        logp = F.cross_entropy(output, target, self.alpha)
        p = torch.exp(-logp)
        focal_loss = (1 - p) ** self.gamma * logp

        return torch.mean(focal_loss)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        max_m: The appropriate value for max_m depends on the specific dataset and the severity of the class imbalance.
        You can start with a small value and gradually increase it to observe the impact on the model's performance.
        If the model struggles with class separation or experiences underfitting, increasing max_m might help. However,
        be cautious not to set it too high, as it can cause overfitting or make the model too conservative.

        s: The choice of s depends on the desired scale of the logits and the specific requirements of your problem.
        It can be used to adjust the balance between the margin and the original logits. A larger s value amplifies
        the impact of the logits and can be useful when dealing with highly imbalanced datasets.
        You can experiment with different values of s to find the one that works best for your dataset and model.

        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class LMFLoss(nn.Module):
    def __init__(self, cls_num_list, weight, alpha=1, beta=1, gamma=2, max_m=0.5, s=30):
        super().__init__()
        weight = torch.tensor(weight).cuda()
        self.focal_loss = FocalLoss(weight, gamma)
        self.ldam_loss = LDAMLoss(cls_num_list, max_m, weight, s)
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        focal_loss_output = self.focal_loss(output, target)
        ldam_loss_output = self.ldam_loss(output, target)
        total_loss = self.alpha * focal_loss_output + self.beta * ldam_loss_output
        return total_loss


class Fuse_loss(nn.Module):
    def __init__(self,task_numbers = 2):
        super(Fuse_loss, self).__init__()
        # self.class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        weight = torch.tensor([1.266,6.562,17.248]).cuda()
        # self.class_criterion = FocalLoss(alpha=weight)
        self.class_criterion = nn.MSELoss()
        self.KL = KLLoss()
        self.SIM = SIMLoss()
        self.CC = CCLoss()
        self.reg_criterion = nn.MSELoss()
        self.auto_uw = AutomaticWeightedLoss(task_num=task_numbers)

    def forward(self,Risk_map_output, Risk_level,Risk_Map, Risk_Label,Road_map):
        # Risk_map_loss = self.KL(Risk_map_output, Risk_Map) - 0.2 * self.SIM(Risk_map_output, Risk_Map)
        Risk_map_output = torch.mul(Risk_map_output, Road_map.squeeze(1))
        Risk_map_mse = self.reg_criterion(Risk_map_output,Risk_Map)
        Risk_map_loss = Risk_map_mse
        Risk_level_loss = self.class_criterion(Risk_level, Risk_Label)

        sum_loss = Risk_level_loss + Risk_map_loss
        return sum_loss,Risk_level_loss,Risk_map_loss,Risk_map_mse

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, task_num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(task_num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum




# if __name__ == '__main__':
#     y_pred = torch.randn((4, 256,256))
#     y_true = torch.randn((4, 256, 256))
#     y_pred = torch.abs(y_pred)
#     y_true = torch.abs(y_true)
#     KLloss = KLLoss()
#     output = KLloss(y_pred,y_true)
#     pass
