#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import sys
import yaml
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import errno
import pickle

from network.LW_PSC_all import LW_PSC_all
from dataloader.dataset import (
    collate_fn_BEV_addPolar,
    SemKITTI,
    SemKITTI_label_name,
    voxel_polar_dataset,
)
from network.instance_post_processing import get_panoptic_segmentation
from utils.eval_pq import PanopticEval
from utils.configs import merge_configs
from logger import get_logger

# ignore weird np warning
import warnings

warnings.filterwarnings("ignore")
def _create_directory(directory):
    '''
    Create directory if doesn't exists
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
    return

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    return label - 1  # uint8 trick

def floodfill(volume, mask, x, y, z, instance_id):
        stack = [(x, y, z)]
        while len(stack) > 0:
            x, y, z = stack.pop()
            mask[x, y, z] = instance_id
            volume[x, y, z] = 0
            for x_offset in [-1, 0, 1]:
                for y_offset in [-1, 0, 1]:
                    for z_offset in [-1, 0, 1]:
                        if x_offset == 0 and y_offset == 0 and z_offset == 0:
                            continue
                        x_next = x + x_offset
                        y_next = y + y_offset
                        z_next = z + z_offset
                        if x_next < 0 or x_next > (256 - 1):
                            continue
                        if y_next < 0 or y_next > (256 - 1):
                            continue
                        if z_next < 0 or z_next > (32 - 1):
                            continue
                        if (
                            mask[x_next, y_next, z_next] == 0
                            and volume[x_next, y_next, z_next] != 0
                        ):
                            stack.append((x_next, y_next, z_next))
                            
def dict_to(_dict, device):
    for key, value in _dict.items():
      if type(_dict[key]) is dict:
        _dict[key] = dict_to(_dict[key], device)
      if type(_dict[key]) is list:
          _dict[key] = [v.to(device) for v in _dict[key]]
      else:
        _dict[key] = _dict[key].to(device)
    return _dict

def remap(labels: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    labels[labels == 255] = 0
    return label_map[labels]

def main(args):
    data_path = args["dataset"]["path"]
    # val_batch_size = args["model"]["val_batch_size"]
    test_batch_size = args["model"]["test_batch_size"]
    pretrained_model = args["model"]["pretrained_model"]
    # output_path = args["dataset"]["output_path"]
    compression_model = args["dataset"]["grid_size"][2]
    grid_size = args["dataset"]["grid_size"]
    visibility = args["model"]["visibility"]
    pytorch_device = torch.device("cuda:0")
    if args["model"]["polar"]:
        fea_dim = 9
        circular_padding = True
    else:
        fea_dim = 7
        circular_padding = False
    class_frequencies = np.array(
        [
            5.41773033e09,
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
        batch_size=test_batch_size,
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
    # learnning_map_inv = semkittiyaml["learning_map_inv"]
    evaluator = PanopticEval(len(unique_label), None, min_points=50)
    logger = get_logger("/media/anda/hdd31/minh/LW-PSC", "logs_val-test.log")
    pbar_val = tqdm(total=len(val_dataset_loader))
    my_model.eval()
    evaluator.reset()
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
            val_return_polar_fea
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

            torch.cuda.synchronize()
            start_time = time.time()
            sem_prediction, center, offset,_,_  = my_model(
                val_occupancy_ten, val_points_ten, val_labels_ten, val_aux_com_ten, val_distance_feature_ten, val_grid_polar_ind_ten, val_return_polar_fea_ten
            )
            torch.cuda.synchronize()
            time_list.append(time.time() - start_time)
            
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
                panoptic = panoptic_labels[
                    0,
                    val_sem_diff255_ind[count][:, 0],
                    val_sem_diff255_ind[count][:, 1],
                    val_sem_diff255_ind[count][:, 2],
                ]
                evaluator.addBatch(
                    panoptic & 0xFFFF,
                    panoptic,
                    np.squeeze(val_flatten_labels[count]),
                    np.squeeze(val_flatten_ints[count]),
                )
            del (
                val_occupancy_ten,
                val_points_ten,
                val_labels_ten,
                # val_ssclabels_ten,
                # val_gt_center,
                # val_gt_center_ten,
                # val_gt_offset,
                # val_gt_offset_ten,
                sem_prediction,
                center,
                offset,
                panoptic_labels,
                center_points,
                val_sem_diff255_ind,
            )
            pbar_val.update(1)
    pbar_val.close()
    my_model.train()
    (
        class_PQ,
        class_SQ,
        class_RQ,
        class_all_PQ,
        class_all_SQ,
        class_all_RQ,
    ) = evaluator.getPQ()
    miou, ious = evaluator.getSemIoU()
    PQ_dagger = np.mean(
        [class_all_PQ[c] for c in [1, 2, 3, 4, 5, 6, 7, 8]] + [ious[c] for c in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
    )
    PQ_th = np.mean([class_all_PQ[c] for c in [1, 2, 3, 4, 5, 6, 7, 8]])
    SQ_th = np.mean([class_all_SQ[c] for c in [1, 2, 3, 4, 5, 6, 7, 8]])
    RQ_th = np.mean([class_all_RQ[c] for c in [1, 2, 3, 4, 5, 6, 7, 8]])

    PQ_st = np.mean([class_all_PQ[c] for c in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    SQ_st = np.mean([class_all_SQ[c] for c in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    RQ_st = np.mean([class_all_RQ[c] for c in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])

    Completion_meanIoU = evaluator.getSSCmIoU()
    Completion_IoU = evaluator.getOccIoU()

    logger.info('============ Valuation: "%s" ============\n')
    logger.info("Validation per class PQ, SQ, RQ and IoU: ")
    print("Validation per class PQ, SQ, RQ and IoU: ")
    for class_name, class_pq, class_sq, class_rq, class_iou in zip(
        unique_label_str,
        class_all_PQ[:],
        class_all_SQ[:],
        class_all_RQ[:],
        ious[:],
    ):
        print(
            "%15s : %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%"
            % (
                class_name,
                class_pq * 100,
                class_sq * 100,
                class_rq * 100,
                class_iou * 100,
            )
        )
        logger.info(
            "%15s : %6.2f%%  %6.2f%%  %6.2f%%  %6.2f%%"
            % (
                class_name,
                class_pq * 100,
                class_sq * 100,
                class_rq * 100,
                class_iou * 100,
            )
        )

    logger.info("Completion score: %6.2f%%; Semantic Completion score: %6.2f%%" %(Completion_IoU * 100, Completion_meanIoU * 100))
    # save model if performance is improved
    # if best_val_PQ < class_PQ:
    #     best_val_PQ = class_PQ
    # if best_val_PQ_dagger < PQ_dagger:
    #     best_val_PQ_dagger = PQ_dagger
    #     torch.save(my_model.state_dict(), model_save_path)
    logger.info(
        "Current val PQ dagger is %.3f "
        % (PQ_dagger * 100)
    )
    logger.info(
        "Current val PQ: %.3f, val SQ: %.3f , val RQ: %.3f "
        % (class_PQ * 100, class_SQ * 100, class_RQ * 100)
    )
    logger.info(
        "Current val PQ thing: %.3f, val SQ thing: %.3f , val RQ thing: %.3f "
        % (PQ_th * 100, SQ_th * 100, RQ_th * 100)
    )
    logger.info(
        "Current val PQ stuff: %.3f, val SQ stuff: %.3f , val RQ stuff: %.3f "
        % (PQ_st * 100, SQ_st * 100, RQ_st * 100)
    )
    logger.info("Current val miou is %.3f" % (miou * 100))
    logger.info(
        "Inference time per %d is %.4f seconds, postprocessing time is %.4f seconds per scan"
        % (test_batch_size, np.mean(time_list[1:]), np.mean(pp_time_list[1:]))
    )


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
