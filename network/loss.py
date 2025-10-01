#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from .lovasz_losses import lovasz_softmax
import matplotlib.pyplot as plt


def _neg_loss(pred, gt):
    """Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        (https://github.com/tianweiy/CenterPoint)
    Arguments:
        pred (batch x c x h x w)
        gt (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    # loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    return -(pos_loss + neg_loss)

def get_ohem_loss(loss_mat, valid_mask=None, top_ratio=0, top_weight=1):
    loss_mat_valid = None
    valid_num = None
    topk_num = None
    if valid_mask is not None:
        loss_mat_valid = (loss_mat * valid_mask).view(-1)
        valid_num = int(valid_mask.sum())
        topk_num = int(valid_num * top_ratio)
    else:
        loss_mat_valid = loss_mat.view(-1)
        valid_num = loss_mat_valid.shape[0]
        topk_num = int(valid_num * top_ratio)
    
    loss_total = loss_mat_valid.sum() / (valid_num + 1e-12)
    if topk_num == 0:
        return loss_total
    else:
        loss_topk = torch.topk(loss_mat_valid, k=topk_num, dim=0, largest=True, sorted=False)[0]
        loss_total = loss_total + top_weight * loss_topk.mean()
        return loss_total

class FocalLoss(torch.nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class panoptic_loss(torch.nn.Module):
    def __init__(
        self,
        ignore_label=255,
        center_loss_weight=100,
        offset_loss_weight=1,
        center_loss="MSE",
        offset_loss="L1",
    ):
        super(panoptic_loss, self).__init__()
        
        self.class_frequencies = np.array(
        [
            # 5.41773033e09,
            7632350044,
            1.57835390e07,
            1.25136000e05,
            1.18809000e05,
            6.46799000e05,
            8.21951000e05,
            2.62978000e05,
            2.83696000e05,
            2.04750000e05,
            6.16887030e07,
            4.50296100e06,
            4.48836500e07,
            2.26992300e06,
            5.68402180e07,
            1.57196520e07,
            1.58442623e08,
            2.06162300e06,
            3.69705220e07,
            1.15198800e06,
            3.34146000e05,
        ]
    )   
        class_weights = self.get_class_weights()
        class_weights = class_weights.to(torch.device("cuda:0")).float()
        self.CE_loss = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_label, reduction="mean")
        assert center_loss in ["MSE", "FocalLoss"]
        assert offset_loss in ["L1", "SmoothL1"]
        if center_loss == "MSE":
            self.center_loss_fn = torch.nn.MSELoss()
        elif center_loss == "FocalLoss":
            self.center_loss_fn = FocalLoss()
        else:
            raise NotImplementedError
        if offset_loss == "L1":
            self.offset_loss_fn = torch.nn.L1Loss()
        elif offset_loss == "SmoothL1":
            self.offset_loss_fn = torch.nn.SmoothL1Loss()
        else:
            raise NotImplementedError
        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight

        print(
            "Using "
            + center_loss
            + " for heatmap regression, weight: "
            + str(center_loss_weight)
        )
        print(
            "Using "
            + offset_loss
            + " for offset regression, weight: "
            + str(offset_loss_weight)
        )

        self.lost_dict = {"aux_sem_loss": [], "aux_com_loss": [], "semantic_loss": [], "heatmap_loss": [], "offset_loss": []}

    def reset_loss_dict(self):
        self.lost_dict = {"aux_sem_loss": [], "aux_com_loss": [], "semantic_loss": [], "heatmap_loss": [], "offset_loss": []}
    
    def get_class_weights(self):
        """
        Class weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
        """
        epsilon_w = 0.001  # eps to avoid zero division
        weights = torch.from_numpy(1 / np.log(self.class_frequencies + epsilon_w))

        return weights

    def forward(
        self, prediction, center, offset, gt_label, gt_center, gt_offset, aux_ss_loss, aux_sc_loss, save_loss=True
    ):
        loss = 0
        # Auxiliary semantic loss and completion loss
        aux_loss_seg = sum(aux_ss_loss.values())
        aux_loss_com = sum(aux_sc_loss.values())
        if save_loss:
            self.lost_dict["aux_sem_loss"].append(aux_loss_seg.item())
            self.lost_dict["aux_com_loss"].append(aux_loss_com.item())
        
        loss += (aux_loss_seg + aux_loss_com)
        # semantic completion loss
        
        sem_loss = lovasz_softmax(
            torch.nn.functional.softmax(prediction), gt_label.long(), ignore=255
        ) + self.CE_loss(prediction, gt_label.long())
        sem_loss *= 3
        if save_loss:
            self.lost_dict["semantic_loss"].append(sem_loss.item())
        loss += sem_loss
        
        # center heatmap loss
        center_mask = (gt_center > 0) | (
            torch.min(torch.unsqueeze(gt_label, 1), dim=4)[0] < 255
        )
        center_loss = self.center_loss_fn(center, gt_center) * center_mask
        # safe division
        if center_mask.sum() > 0:
            center_loss = (
                center_loss.sum() / center_mask.sum() * self.center_loss_weight
            )
        else:
            center_loss = center_loss.sum() * 0
        if save_loss:
            self.lost_dict["heatmap_loss"].append(center_loss.item())
        loss += center_loss
        
        # offset loss
        offset_mask = gt_offset != 0
            
        offset_loss = self.offset_loss_fn(offset, gt_offset) * offset_mask
        # safe division
        if offset_mask.sum() > 0:
            offset_loss = (
                offset_loss.sum() / offset_mask.sum() * self.offset_loss_weight
            )
        else:
            offset_loss = offset_loss.sum() * 0
        if save_loss:
            self.lost_dict["offset_loss"].append(offset_loss.item())
        loss += offset_loss
        print(
            "aux_sem_loss: %.4f, aux_com_loss: %.4f, semantic_loss: %.4f, heatmap_loss: %.4f, offset_loss: %.4f"
            % (
                self.lost_dict["aux_sem_loss"][-1],
                self.lost_dict["aux_com_loss"][-1],
                self.lost_dict["semantic_loss"][-1],
                self.lost_dict["heatmap_loss"][-1],
                self.lost_dict["offset_loss"][-1],
            )
        )
        return loss

class BCE_OHEM(nn.Module):
    def __init__(self, top_ratio=0.3, top_weight=1.0):
        super(BCE_OHEM, self).__init__()
        self.top_ratio = top_ratio
        self.top_weight = top_weight
    
    def forward(self, pred, gt, valid_mask=None):
        #pdb.set_trace()
        # loss_mat = F.binary_cross_entropy(pred, gt, reduce=False)
        loss_mat = -1 * (gt * torch.log(pred + 1e-12) + (1 - gt) * torch.log(1 - pred + 1e-12))
        loss_result = get_ohem_loss(loss_mat, valid_mask, top_ratio=self.top_ratio, top_weight=self.top_weight)
        return loss_result

class panoptic_loss_CF(torch.nn.Module):
    def __init__(
        self,
        ignore_label=255,
        center_loss_weight=100,
        offset_loss_weight=1,
    ):
        super(panoptic_loss_CF, self).__init__()
        
        self.class_frequencies = np.array(
        [
            # 5.41773033e09,
            7632350044,
            1.57835390e07,
            1.25136000e05,
            1.18809000e05,
            6.46799000e05,
            8.21951000e05,
            2.62978000e05,
            2.83696000e05,
            2.04750000e05,
            6.16887030e07,
            4.50296100e06,
            4.48836500e07,
            2.26992300e06,
            5.68402180e07,
            1.57196520e07,
            1.58442623e08,
            2.06162300e06,
            3.69705220e07,
            1.15198800e06,
            3.34146000e05,
        ]
    )   
        class_weights = self.get_class_weights()
        class_weights = class_weights.to(torch.device("cuda:0")).float()
        self.CE_loss = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_label, reduction="mean")

        # instance loss
        self.center_loss = BCE_OHEM(top_ratio=0.1, top_weight=3.0)
        self.sigma = np.sqrt(-1 * 0.8 * 0.8 / (2 * np.log(0.2)))
        
        # self.offset_loss_fn = torch.nn.L1Loss()
        
        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        
        self.center_loss_fn = torch.nn.MSELoss()
        self.offset_loss_fn = torch.nn.L1Loss()
        self.center_loss_weight_2 = 300
        self.offset_loss_weight_2 = 30

        self.lost_dict = {"aux_sem_loss": [], "aux_com_loss": [], "semantic_loss": [], "heatmap_loss": [], "offset_loss": []}

    def reset_loss_dict(self):
        self.lost_dict = {"aux_sem_loss": [], "aux_com_loss": [], "semantic_loss": [], "heatmap_loss": [], "offset_loss": []}
    
    def get_class_weights(self):
        """
        Class weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
        """
        epsilon_w = 0.001  # eps to avoid zero division
        weights = torch.from_numpy(1 / np.log(self.class_frequencies + epsilon_w))

        return weights
    
    def forward(
        self, preds_list, gt_label, gt_BEV_instance, gt_hmap, gt_offset, aux_ss_loss, aux_sc_loss, save_loss=True
    ):
        loss = 0
        # Auxiliary semantic loss and completion loss
        aux_loss_seg = sum(aux_ss_loss.values())
        aux_loss_com = sum(aux_sc_loss.values())
        if save_loss:
            self.lost_dict["aux_sem_loss"].append(aux_loss_seg.item())
            self.lost_dict["aux_com_loss"].append(aux_loss_com.item())
        
        loss += (aux_loss_seg + aux_loss_com)
        
        # for i in range(len(preds_list)):
        # preds 1
        prediction, pred_hmap, offset = preds_list[0]
        batch_size = prediction.shape[0]
        # semantic completion loss
        sem_loss = lovasz_softmax(
            torch.nn.functional.softmax(prediction), gt_label.long(), ignore=255
        ) + self.CE_loss(prediction, gt_label.long())
        sem_loss *= 3
        loss += sem_loss
        
        # offset loss
        valid_mask = ((gt_label != 0) & (gt_label != 255)).max(dim=3)[0].unsqueeze(dim=3).view(batch_size,-1,1)
        fg_mask = (gt_BEV_instance > 1).view(batch_size,-1,1)
        
        fg_num = int(fg_mask.float().sum()) + 1e-12
        
        offset = offset.permute(0, 2, 3, 1).contiguous() # (BS, H, W, 2)
        gt_offset_mid = gt_offset.permute(0, 2, 3, 1).contiguous() # (BS, H, W, 2)
        offset_flatten = offset.view(batch_size,-1, 2) # (BS,H*W, 2)
        gt_offset_flatten = gt_offset_mid.view(batch_size,-1, 2) # (BS,H*W, 2)
        pred_hmap = pred_hmap.permute(0, 2, 3, 1).contiguous() # (BS, H, W, 1)
        pred_hmap = pred_hmap.view(batch_size,-1, 1) # (BS,H*W, 1)
        
        loss_point = (offset_flatten - gt_offset_flatten).pow(2).sum(dim=2, keepdim=True).sqrt()
        offset_loss = (loss_point * fg_mask.float()).sum() / fg_num
        offset_loss = offset_loss * self.offset_loss_weight
        loss += offset_loss
        
        # center heatmap loss
        gt_hmap_conf = torch.exp(-1 * loss_point.detach().pow(2) / (2 * self.sigma * self.sigma)) * fg_mask.float() #(BS, N, 1)
        center_loss = self.center_loss(pred_hmap[valid_mask], gt_hmap_conf[valid_mask])
        center_loss = center_loss * self.center_loss_weight
        loss += center_loss
        
        # preds 2
        prediction_2, pred_hmap_2, offset_2 = preds_list[1]
        # SSC loss
        sem_loss_2 = lovasz_softmax(
            torch.nn.functional.softmax(prediction_2), gt_label.long(), ignore=255
        ) + self.CE_loss(prediction, gt_label.long())
        sem_loss_2 *= 3
        sem_loss = sem_loss_2 + sem_loss
        loss += sem_loss_2
        # center heatmap loss
        center_mask = (gt_hmap > 0) | (
            torch.min(torch.unsqueeze(gt_label, 1), dim=4)[0] < 255
        )
        center_loss_2 = self.center_loss_fn(pred_hmap_2, gt_hmap) * center_mask
        # safe division
        if center_mask.sum() > 0:
            center_loss_2 = (
                center_loss_2.sum() / center_mask.sum() * self.center_loss_weight_2
            )
        else:
            center_loss_2 = center_loss_2.sum() * 0
        center_loss = center_loss + center_loss_2
        loss += center_loss_2
        
        # offset loss
        offset_mask = gt_offset != 0
            
        offset_loss_2 = self.offset_loss_fn(offset_2, gt_offset) * offset_mask
        # safe division
        if offset_mask.sum() > 0:
            offset_loss_2 = (
                offset_loss_2.sum() / offset_mask.sum() * self.offset_loss_weight_2
            )
        else:
            offset_loss_2 = offset_loss_2.sum() * 0
        offset_loss = offset_loss + offset_loss_2
        loss += offset_loss_2
        
        
        if save_loss:
            self.lost_dict["semantic_loss"].append(sem_loss.item())
            self.lost_dict["offset_loss"].append(offset_loss.item())
            self.lost_dict["heatmap_loss"].append(center_loss.item())
        
        print(
            "aux_sem_loss: %.4f, aux_com_loss: %.4f, semantic_loss: %.4f, heatmap_loss: %.4f, offset_loss: %.4f"
            % (
                self.lost_dict["aux_sem_loss"][-1],
                self.lost_dict["aux_com_loss"][-1],
                self.lost_dict["semantic_loss"][-1],
                self.lost_dict["heatmap_loss"][-1],
                self.lost_dict["offset_loss"][-1],
            )
        )
        return loss
    
class panoptic_loss_CFv1(torch.nn.Module):
    def __init__(
        self,
        ignore_label=255,
        center_loss_weight=None,
        offset_loss_weight=None,
    ):
        super(panoptic_loss_CFv1, self).__init__()
        
        self.class_frequencies = np.array(
        [
            # 5.41773033e09,
            7632350044,
            1.57835390e07,
            1.25136000e05,
            1.18809000e05,
            6.46799000e05,
            8.21951000e05,
            2.62978000e05,
            2.83696000e05,
            2.04750000e05,
            6.16887030e07,
            4.50296100e06,
            4.48836500e07,
            2.26992300e06,
            5.68402180e07,
            1.57196520e07,
            1.58442623e08,
            2.06162300e06,
            3.69705220e07,
            1.15198800e06,
            3.34146000e05,
        ]
    )   
        class_weights = self.get_class_weights()
        class_weights = class_weights.to(torch.device("cuda:0")).float()
        self.CE_loss = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_label, reduction="mean")

        # instance loss
        self.center_loss = BCE_OHEM(top_ratio=0.1, top_weight=3.0)
        self.sigma = np.sqrt(-1 * 0.8 * 0.8 / (2 * np.log(0.2)))
        
        # self.offset_loss_fn = torch.nn.L1Loss()
        
        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        
        self.lost_dict = {"aux_sem_loss": [], "aux_com_loss": [], "semantic_loss": [], "heatmap_loss": [], "offset_loss": []}

    def reset_loss_dict(self):
        self.lost_dict = {"aux_sem_loss": [], "aux_com_loss": [], "semantic_loss": [], "heatmap_loss": [], "offset_loss": []}
    
    def get_class_weights(self):
        """
        Class weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
        """
        epsilon_w = 0.001  # eps to avoid zero division
        weights = torch.from_numpy(1 / np.log(self.class_frequencies + epsilon_w))

        return weights
    
    def forward(
        self, prediction, pred_hmap, offset, gt_label, gt_offset, aux_ss_loss, aux_sc_loss, save_loss=True
    ):
        loss = 0
        # Auxiliary semantic loss and completion loss
        aux_loss_seg = sum(aux_ss_loss.values())
        aux_loss_com = sum(aux_sc_loss.values())
        if save_loss:
            self.lost_dict["aux_sem_loss"].append(aux_loss_seg.item())
            self.lost_dict["aux_com_loss"].append(aux_loss_com.item())
        
        loss += (aux_loss_seg + aux_loss_com)
        
        # for i in range(len(preds_list)):
        batch_size = prediction.shape[0]
        # semantic completion loss
        sem_loss = lovasz_softmax(
            torch.nn.functional.softmax(prediction), gt_label.long(), ignore=255
        ) + self.CE_loss(prediction, gt_label.long())
        sem_loss *= 3
        loss += sem_loss
        
        # offset loss
        valid_mask = ((gt_label != 0) & (gt_label != 255)).max(dim=3)[0].unsqueeze(dim=3)
        valid_mask_flatten = valid_mask.view(batch_size,-1,1)
        # fg_mask = (gt_BEV_instance > 1)
        # fg_mask_flatten = fg_mask.view(batch_size,-1,1)
        
        fg_mask_2 = torch.zeros_like(gt_label)
        for thing_class in [1,2,3,4,5,6,7,8]:
            fg_mask_2[gt_label == thing_class] = 1
        # [1, H, W, Z] --> [1, H, W]
        fg_mask_2 = (torch.max(fg_mask_2,dim=3)[0]) > 0
        fg_mask_flatten = fg_mask_2.view(batch_size,-1,1)
        
        # x,y = np.where(valid_mask[0].squeeze().cpu().detach().numpy() != 0)
        # fig = plt.figure(figsize=(10, 10))
        # ax1 = fig.add_subplot(141)
        # ax1.scatter(
        #     y,  # X-coordinates
        #     x,  # Y-coordinates
        #     c='blue',       # Color based on the heatmap intensity
        #     s=10,                    # Marker size
        #     label="Center"
        # )
        # ax1.set_xlabel("X-axis")
        # ax1.set_ylabel("Y-axis")
        # ax1.legend()
        
        # x1,y1 = np.where(fg_mask[0].cpu().detach().numpy() != 0)
        # ax2 = fig.add_subplot(142)
        # ax2.scatter(
        #     y1,  # X-coordinates
        #     x1,  # Y-coordinates
        #     c='red',       # Color based on the heatmap intensity
        #     s=10,                    # Marker size
        #     label="Center"
        # )
        # ax2.set_xlabel("X-axis")
        # ax2.set_ylabel("Y-axis")
        # ax2.legend()
        
        # x2,y2 = np.where(fg_mask_2[0].cpu().detach().numpy() != 0)
        # ax2 = fig.add_subplot(143)
        # ax2.scatter(
        #     y2,  # X-coordinates
        #     x2,  # Y-coordinates
        #     c='red',       # Color based on the heatmap intensity
        #     s=10,                    # Marker size
        #     label="Center"
        # )
        # ax2.set_xlabel("X-axis")
        # ax2.set_ylabel("Y-axis")
        # ax2.legend()
        
        # x3,y3,z3 = np.where(gt_label[0].cpu().detach().numpy() > 0)
        # values = gt_label[0, x3, y3, z3].cpu().detach().numpy()
        # colormap = plt.cm.get_cmap("tab20", 20)
        # colors = colormap(values / 19.0)
        
        # ax = fig.add_subplot(144, projection="3d")
        # ax.scatter(
        #     x3, y3, z3, c=colors, s=20, marker="o"
        # )
        
        # cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), ax=ax, shrink=0.5, aspect=10)
        # cbar.set_ticks(np.linspace(0, 1, 19))
        # cbar.set_ticklabels(range(19))
        # cbar.set_label("Values")
        
        # ax.set_title("3D Array Visualization with 19 Colors")
        # ax.set_xlabel("X-axis")
        # ax.set_ylabel("Y-axis")
        # ax.set_zlabel("Z-axis")
        
        # plt.tight_layout()
        # plt.show()
        
        fg_num = int(fg_mask_flatten.float().sum()) + 1e-12
        
        offset = offset.permute(0, 2, 3, 1).contiguous() # (BS, H, W, 2)
        gt_offset_mid = gt_offset.permute(0, 2, 3, 1).contiguous() # (BS, H, W, 2)
        offset_flatten = offset.view(batch_size,-1, 2) # (BS,H*W, 2)
        gt_offset_flatten = gt_offset_mid.view(batch_size,-1, 2) # (BS,H*W, 2)
        pred_hmap = pred_hmap.permute(0, 2, 3, 1).contiguous() # (BS, H, W, 1)
        pred_hmap = pred_hmap.view(batch_size,-1, 1) # (BS,H*W, 1)
        
        loss_point = (offset_flatten - gt_offset_flatten).pow(2).sum(dim=2, keepdim=True).sqrt()
        offset_loss = (loss_point * fg_mask_flatten.float()).sum() / fg_num
        offset_loss = offset_loss * self.offset_loss_weight
        loss += offset_loss
        
        # center heatmap loss
        gt_hmap_conf = torch.exp(-1 * loss_point.detach().pow(2) / (2 * self.sigma * self.sigma)) * fg_mask_flatten.float() #(BS, N, 1)
        center_loss = self.center_loss(pred_hmap[valid_mask_flatten], gt_hmap_conf[valid_mask_flatten])
        center_loss = center_loss * self.center_loss_weight
        loss += center_loss
                
        if save_loss:
            self.lost_dict["semantic_loss"].append(sem_loss.item())
            self.lost_dict["offset_loss"].append(offset_loss.item())
            self.lost_dict["heatmap_loss"].append(center_loss.item())
        
        print(
            "aux_sem_loss: %.4f, aux_com_loss: %.4f, semantic_loss: %.4f, heatmap_loss: %.4f, offset_loss: %.4f"
            % (
                self.lost_dict["aux_sem_loss"][-1],
                self.lost_dict["aux_com_loss"][-1],
                self.lost_dict["semantic_loss"][-1],
                self.lost_dict["heatmap_loss"][-1],
                self.lost_dict["offset_loss"][-1],
            )
        )
        return loss
    