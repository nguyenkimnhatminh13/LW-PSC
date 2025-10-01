import os
import torch
import pickle

import os
import argparse
import sys
import numpy as np
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

from network.LW_PSC_all import LW_PSC_all
from dataloader.dataset import (
    collate_fn_BEV_addPolar,
    SemKITTI,
    SemKITTI_label_name,
    # spherical_dataset,
    voxel_polar_dataset,
)
from network.instance_post_processing import get_panoptic_segmentation
from network.loss import panoptic_loss
from utils.eval_pq import PanopticEval
from utils.configs import merge_configs
import time

# ignore weird np warning
# from utils.helper_kitti_mayavi import draw_instance, draw_semantic, draw_panoptic, draw_pcd
# import warnings
# from mayavi import mlab
# try:
#     engine = mayavi.engine
# except NameError:
#     from mayavi.api import Engine
#     engine = Engine()
#     engine.start()


def dict_to(_dict, device):
    for key, value in _dict.items():
      if type(_dict[key]) is dict:
        _dict[key] = dict_to(_dict[key], device)
      if type(_dict[key]) is list:
          _dict[key] = [v.to(device) for v in _dict[key]]
      else:
        _dict[key] = _dict[key].to(device)
    return _dict

# def visualize_semantic(ssc_pred, filename=None, figure=None):
#     from_coords =np.argwhere(ssc_pred != 0)
#     from_feature = ssc_pred[from_coords[:, 0], from_coords[:, 1], from_coords[:, 2]]
#     draw_semantic(from_coords, from_feature, filename=filename, voxel_size=0.2, figure=figure)

def main(args):
    # frame_ids = list(range(0, 4080, 1))
    # frame_ids = [int(frame_id) for frame_id in frame_ids]
    # figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1), engine=engine)
    # frame_ids = ["{:06d}".format(int(number)) for number in frame_ids]

    data_path = args["dataset"]["path"]
    val_batch_size = args["model"]["val_batch_size"]
    pretrained_model = args["model"]["pretrained_model"]
    # output_path = args["dataset"]["output_path"]
    grid_size = args["dataset"]["grid_size"]
    pytorch_device = torch.device("cuda:0")
    
    # prepare miou fun
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[:]
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label]

    # prepare model
    my_model = LW_PSC_all(
        class_num=len(unique_label),
        input_dimensions=grid_size,
    )
    
    if os.path.exists(pretrained_model):
        my_model.load_state_dict(torch.load(pretrained_model))
    pytorch_total_params = sum(p.numel() for p in my_model.parameters())
    print("params: ", pytorch_total_params)
    my_model.to(pytorch_device)
    my_model.eval()

    # prepare dataset
    val_pt_dataset = SemKITTI(
        data_path + "/sequences/",
        imageset="val",
        instance_pkl_path=args["dataset"]["instance_pkl_path"],
    )

    val_dataset = voxel_polar_dataset(
        val_pt_dataset, args["dataset"], grid_size=grid_size, ignore_label=0
    )

    val_dataset_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        collate_fn=collate_fn_BEV_addPolar,
        shuffle=False,
        num_workers=4,
    )
    
    # validation
    print("*" * 80)
    print("Test network performance on validation split")
    print("*" * 80)
    pbar = tqdm(total=len(val_dataset_loader))
    time_list = []
    pp_time_list = []
    curr_index = 0
    with open("semantic-kitti.yaml", "r") as stream:
        semkittiyaml = yaml.safe_load(stream)
    learnning_map_inv = semkittiyaml["learning_map_inv"]
    save_folder = "output"
    frame_id = 0
    with torch.no_grad():
        for i_iter_val, (
            val_occupancy,
            val_points,
            val_labels,
            _,
            _,
            _,
            val_flatten_labels,
            val_flatten_ints,
            val_sem_diff255_ind,
            val_aux_com,
            val_distance_feature,
            val_grid_polar_ind,
            val_return_polar_fea,
        ) in enumerate(val_dataset_loader):
            val_occupancy_ten = val_occupancy.to(pytorch_device)
            val_points_ten = [
                torch.from_numpy(i).to(pytorch_device)
                .type(torch.float32)
                for i in val_points
            ]
            val_labels_ten = [
                torch.from_numpy(i).to(pytorch_device).type(torch.int32)
                for i in val_labels
            ]

            val_aux_com_ten = dict_to(val_aux_com, pytorch_device)
            
            val_distance_feature_ten = val_distance_feature.to(pytorch_device)
            val_grid_polar_ind_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in val_grid_polar_ind]
            val_return_polar_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_return_polar_fea]

            sem_prediction, center, offset,_,_  = my_model(
                val_occupancy_ten, val_points_ten, val_labels_ten, val_aux_com_ten, val_distance_feature_ten, val_grid_polar_ind_ten, val_return_polar_fea_ten
            )
            
            for count, i_val_grid in enumerate(val_sem_diff255_ind):
                # get foreground_mask
                for_mask = torch.zeros(
                    1,
                    grid_size[0],
                    grid_size[1],
                    grid_size[2],
                    dtype=torch.bool,
                ).to(pytorch_device)
                for_mask[
                    0,
                    val_sem_diff255_ind[count][:, 0],
                    val_sem_diff255_ind[count][:, 1],
                    val_sem_diff255_ind[count][:, 2],
                ] = True
                
                torch.cuda.synchronize()
                start_time = time.time()
                # post processing
                panoptic_labels, center_points = get_panoptic_segmentation(
                    torch.unsqueeze(sem_prediction[count], 0),
                    torch.unsqueeze(center[count], 0),
                    torch.unsqueeze(offset[count], 0),
                    val_pt_dataset.thing_list,
                    threshold=args["model"]["post_proc"]["threshold"],
                    nms_kernel=args["model"]["post_proc"]["nms_kernel"],
                    top_k=args["model"]["post_proc"]["top_k"],
                    polar=False,
                    foreground_mask=for_mask,
                )
                torch.cuda.synchronize()
                pp_time_list.append(time.time() - start_time)
                panoptic_labels = (
                    panoptic_labels.cpu().detach().numpy().astype(np.int32)
                )
                panoptic = panoptic_labels >> 16
                # xyz = np.where(val_occupancy == 1)
                
                semantic_pred = torch.argmax(torch.unsqueeze(sem_prediction[count], 0), dim=1)
                semantic_pred = semantic_pred.cpu().detach().numpy().astype(np.int32)
                panoptic_unique = np.unique(panoptic)
                predSegmentsInfos = [[]]
                for k in range(1, (panoptic_unique[-1] + 1)):
                    predSegmentsInfo = {"id": int(k), "isthing": True, "category_id": 1}
                    predSegmentsInfos[0].append(predSegmentsInfo)
                
                output = {
                    "ssc_pred": semantic_pred,
                    "panoptic_pred": panoptic,
                    "pred_segments_info": predSegmentsInfos,
                }
                # sem_filename = "{}/output_sem_{}.png".format(save_folder, f"{frame_id:06d}")
                # visualize_semantic(semantic_visual, filename=sem_filename, figure=figure)
                
                os.makedirs(save_folder, exist_ok=True)
                filepath = os.path.join(save_folder, "{}_output.pkl".format(f"{frame_id:06d}"))
                with open(filepath, "wb") as handle:
                    pickle.dump(output, handle)
                    print("wrote to", filepath)

                frame_id = frame_id + 1
                

if __name__ == "__main__":
    # Testing settings
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--data_dir", default="data")
    parser.add_argument(
        "-p",
        "--pretrained_model",
        default="LW-PSC-all-best2.pt",
    )
    parser.add_argument(
        "-c", "--configs", default="configs/SemanticKITTI_model/Panoptic-PolarNet.yaml"
    )

    args = parser.parse_args()
    with open(args.configs, "r") as s:
        new_args = yaml.safe_load(s)
    args = merge_configs(args, new_args)

    print(" ".join(sys.argv))
    print(args)
    main(args)
