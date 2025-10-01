import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
import torch_scatter
from network.SC_2D_UNet import BEV_UNet
from network.SC_3D_UNet import Asymm_3d_spconv

from .preprocess import PcPreprocessor
from .bev_net import BEVUNet, BEVUNetv1, BEVUNetv2, outconv, BEVUNetvCFFE
from .completion import CompletionBranch
from .semantic_segmentation import SemanticBranch
from network import backbone
import pytorch_lib
import time

# from networks.common.lovasz_losses import lovasz_softmax

# from networks.models.transformer_predictor import TranformerPredictor
# from networks.common.loss_panoptic import panoptic_loss

class Voxel2BEV_module(nn.Module):
    def __init__(self, output_size, scale_rate=None):
        super(Voxel2BEV_module, self).__init__()
        self.output_size = output_size
        self.scale_rate = scale_rate
        if self.scale_rate is None:
            self.scale_rate = [1 for i in range(len(self.output_size))]
    
    def forward(self, pcds_feat, pcds_ind):
        voxel_feat = pytorch_lib.VoxelMaxPool(pcds_feat=pcds_feat, pcds_ind=pcds_ind, output_size=self.output_size, scale_rate=self.scale_rate)
        return voxel_feat

class CatFusionCtx(nn.Module):
    def __init__(self, in_channel_list, out_channel, double_branch=True):
        super(CatFusionCtx, self).__init__()
        self.double_branch = double_branch
        if self.double_branch:
            out_channel = 2 * out_channel
        
        self.in_channel_list = in_channel_list
        assert len(self.in_channel_list) >= 2

        s = 0
        for in_channel in self.in_channel_list:
            s = s + in_channel
        
        cmid = s // 3
        self.conv1 = backbone.conv3x3_bn_relu(s, cmid, dilation=1)
        self.conv2 = backbone.conv3x3_bn_relu(cmid, cmid, dilation=2)
        self.conv4 = backbone.conv3x3_bn_relu(cmid, cmid, dilation=4)

        self.conv_merge = backbone.conv3x3_bn_relu(3 * cmid, out_channel, dilation=1)
    
    def forward(self, *x_list):
        x_cat = torch.cat(x_list, dim=1)

        x_cat_1 = self.conv1(x_cat)
        x_cat_2 = self.conv2(x_cat_1)
        x_cat_4 = self.conv4(x_cat_2)

        x_merge = torch.cat((x_cat_1, x_cat_2, x_cat_4), dim=1)
        x_out = self.conv_merge(x_merge)
        if self.double_branch:
            x1, x2 = x_out.chunk(2, dim=1)
            return x1.contiguous(), x2.contiguous()
        else:
            return x_out.contiguous(), x_out.contiguous()

class LW_PSC_CFFE(nn.Module):
    def __init__(
        self,
        class_num,
        input_dimensions,
    ):
        super().__init__()
        self.nbr_classes = class_num
        self.lims = [[0, 51.2], [-25.6, 25.6], [-2, 4.4]]
        self.grid_meters = [0.2, 0.2, 0.2]
        self.input_dimensions = input_dimensions  # (256, 256, 32) for KITTI
        self.n_height = self.input_dimensions[-1]
        self.dilation = 1
        self.bilinear = True
        self.group_conv = False
        self.input_batch_norm = True
        self.dropout = 0.5
        self.circular_padding = False
        self.dropblock = False
        phase='trainval'
        ss_weight = [55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858, 240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114, 9833174, 129609852, 4506626, 1168181]
        
        self.preprocess = PcPreprocessor(lims=self.lims, sizes=self.input_dimensions, grid_meters=self.grid_meters, init_size=self.n_height)
        self.sem_branch = SemanticBranch(sizes=self.input_dimensions, nbr_class=self.nbr_classes-1, init_size=self.n_height, class_frequencies=ss_weight, phase=phase)
        self.com_branch = CompletionBranch(init_size=self.n_height, nbr_class=self.nbr_classes, phase=phase)
        self.bev_model = BEVUNetvCFFE(self.nbr_classes*self.n_height, self.n_height, self.dilation, self.bilinear, self.group_conv,
                            self.input_batch_norm, self.dropout, self.circular_padding, self.dropblock)
        self.voxel2bev = Voxel2BEV_module(output_size=(self.input_dimensions[0], self.input_dimensions[1]), scale_rate = (1, 1))
        self.bev_cffe_concat = CatFusionCtx(in_channel_list=(128,20), out_channel=128)
        
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.0)
        self.sem_outc = outconv(128, 640)
        
        self.i_outc_offset = outconv(128, 2)
        self.i_outc_heatmap = outconv(128, 1)
        
    def forward(self, occupancy, points, labels, aux_com):
        batch_size = len(points)
        # cur_dev = occupancy.get_device()
        
        with torch.no_grad():
            indicator = [0]
            pc_ibatch = []
            for i in range(batch_size):
                pc_i = points[i]
                pc_ibatch.append(pc_i)
                indicator.append(pc_i.size(0) + indicator[-1])
            pc = torch.cat(pc_ibatch, dim=0)
        vw_feature, coord_ind, full_coord, info = self.preprocess(pc, indicator)  # N, C; B, C, W, H, D
        coord = torch.cat([coord_ind[:, 0].reshape(-1, 1), torch.flip(coord_ind, dims=[1])[:, :3]], dim=1)
        bev_dense = self.sem_branch.bev_projection(vw_feature, coord, np.array(self.input_dimensions, np.int32)[::-1], batch_size) # B, C, H, W
        torch.cuda.empty_cache()
        
        ss_data_dict = {}
        ss_data_dict['vw_features'] = vw_feature
        ss_data_dict['coord_ind'] = coord_ind
        ss_data_dict['full_coord'] = full_coord
        ss_data_dict['info'] = info
        ss_out_dict = self.sem_branch(ss_data_dict, labels)  # B, C, D, H, W
        
        sc_data_dict = {}
        occupancy = occupancy.permute(0, 3, 1, 2)  # [B, 32, 256, 256]
        sc_data_dict['vw_dense'] = occupancy.unsqueeze(1)
        sc_out_dict = self.com_branch(sc_data_dict, aux_com)

        inputs = torch.cat([occupancy, bev_dense], dim=1)  # B, C, H, W
        s_x, x_pred, center_pred, offset_pred = self.bev_model(inputs, ss_out_dict['mss_bev_dense'], sc_out_dict['mss_bev_dense'])
        new_shape = [x_pred.shape[0], self.nbr_classes, self.n_height, *x_pred.shape[-2:]]    # [B, 20, 32, 256, 256]
        x_pred = x_pred.view(new_shape)
        x_pred = x_pred.permute(0,1,4,3,2)   # [B,20,256,256,32]
        
        # preds_list = [(x_pred, center_pred, offset_pred)]
        
        center_pred_sig = self.sig(center_pred)
        ### Center focus feature phase
        semantic = torch.argmax(x_pred, dim=1)
        batch_size = semantic.shape[0]
        bs, x, y, z = torch.where(semantic != 0)
        bev_cffe_feat_list = []
        for b in range(batch_size):
            
            x, y, z = torch.where(semantic[b] != 0)
            sem_diff0_ind = torch.stack((x.float(), y.float(), z.float()), axis=-1)
            
            sem_diff0_feat = x_pred[b,:, x, y, z].unsqueeze(dim=0).unsqueeze(dim=3)
            sem_diff0_center_sig = center_pred_sig[b,:, x, y].unsqueeze(dim=0).unsqueeze(dim=3)
            sem_diff0_offset = offset_pred[b,:, x, y].unsqueeze(dim=0).unsqueeze(dim=3)
            
            # torch.cuda.synchronize()
            # start_time = time.time()
            pred_offset_high_conf = (sem_diff0_offset.detach() * (sem_diff0_center_sig > 0.2).float()).squeeze(dim=3).permute(0, 2, 1)
            x_shifted = sem_diff0_ind[:,0:1].unsqueeze(dim=0) + pred_offset_high_conf[:, :, [0]]
            y_shifted = sem_diff0_ind[:,1:2].unsqueeze(dim=0) + pred_offset_high_conf[:, :, [1]]
            ind_reproj = torch.stack((x_shifted, y_shifted), dim=2)
            
            bev_cffe_feat = self.voxel2bev(sem_diff0_feat , ind_reproj) # [1, 20, 256, 256]
            # torch.cuda.synchronize()
            # print("CFFE1: ", time.time() - start_time)
            bev_cffe_feat_list.append(bev_cffe_feat)
        bev_cffe_feat_concat = torch.cat(bev_cffe_feat_list, dim=0)
        # torch.cuda.synchronize()
        # start_time = time.time()
        sem_cffe_feat_conv, ins_cffe_feat_conv = self.bev_cffe_concat(s_x, bev_cffe_feat_concat)
            
        sem_pred_CFFE = self.sem_outc(self.dropout(sem_cffe_feat_conv))
        center_pred_CFFE = self.i_outc_heatmap(self.dropout(ins_cffe_feat_conv)) # [B, 1, 256, 256]
        offset_pred_CFFE = self.i_outc_offset(self.dropout(ins_cffe_feat_conv)) # [B, 2, 256, 256]
        # torch.cuda.synchronize()
        # print("CFFE2: ", time.time() - start_time)
        sem_pred_CFFE = sem_pred_CFFE.view(new_shape)
        sem_pred_CFFE = sem_pred_CFFE.permute(0,1,4,3,2)   # [B,20,256,256,32]
        
        # preds_list.append((sem_pred_CFFE, center_pred_CFFE, offset_pred_CFFE))
        # torch.cuda.synchronize()
        # print("CFFE: ", time.time() - start_time)
        aux_ss_loss = ss_out_dict['loss']
        aux_sc_loss = sc_out_dict['loss']

        return sem_pred_CFFE, center_pred_CFFE, offset_pred_CFFE, aux_ss_loss, aux_sc_loss
