import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
import torch_scatter
from network.SC_2D_UNet import BEV_UNet
from network.SC_3D_UNet import Asymm_3d_spconv

from .preprocess import PcPreprocessor
from .bev_net import BEVUNet, BEVUNetv1, BEVUNetv2
from .completion import CompletionBranch
from .semantic_segmentation import SemanticBranch
from .polarBEV import polarBEV

# from networks.common.lovasz_losses import lovasz_softmax

# from networks.models.transformer_predictor import TranformerPredictor
# from networks.common.loss_panoptic import panoptic_loss


class LW_PSC_addPolar(nn.Module):
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
        
        self.polar_preprocess = polarBEV(pt_model = 'pointnet', grid_size = [256,256,32], fea_dim = 9, max_pt_per_encode = 256,
                            out_pt_fea_dim = 512, kernal_size = 1, pt_selection = 'random', fea_compre = 32)
        self.preprocess = PcPreprocessor(lims=self.lims, sizes=self.input_dimensions, grid_meters=self.grid_meters, init_size=self.n_height)
        self.sem_branch = SemanticBranch(sizes=self.input_dimensions, nbr_class=self.nbr_classes-1, init_size=self.n_height, class_frequencies=ss_weight, phase=phase)
        self.com_branch = CompletionBranch(init_size=self.n_height, nbr_class=self.nbr_classes, phase=phase)
        
        self.bev_model = BEVUNetv2(self.nbr_classes*self.n_height, self.n_height, self.dilation, self.bilinear, self.group_conv,
                            self.input_batch_norm, self.dropout, self.circular_padding, self.dropblock)

    def forward(self, occupancy, points, labels, aux_com, distance_feature, grid_polar_ind, polar_fea):
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
        
        polar_feat_list = self.polar_preprocess(polar_fea, grid_polar_ind, distance_feature)
        
        x_pred, center_pred, offset_pred = self.bev_model(inputs, ss_out_dict['mss_bev_dense'], sc_out_dict['mss_bev_dense'], polar_feat_list)
        new_shape = [x_pred.shape[0], self.nbr_classes, self.n_height, *x_pred.shape[-2:]]    # [B, 20, 32, 256, 256]
        x_pred = x_pred.view(new_shape)
        x_pred = x_pred.permute(0,1,4,3,2)   # [B,20,256,256,32]
        
        aux_ss_loss = ss_out_dict['loss']
        aux_sc_loss = sc_out_dict['loss']

        return x_pred, center_pred, offset_pred, aux_ss_loss, aux_sc_loss
