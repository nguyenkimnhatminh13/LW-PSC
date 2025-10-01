#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import sys
import numpy as np
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
from logger import get_logger

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
import warnings

warnings.filterwarnings("ignore")

def load_pretrained_model(model, pretrained_model):
    model_dict = model.state_dict()
    pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict}
    model_dict.update(pretrained_model)
    model.load_state_dict(model_dict)
    return model

def dict_to(_dict, device):
    for key, value in _dict.items():
      if type(_dict[key]) is dict:
        _dict[key] = dict_to(_dict[key], device)
      if type(_dict[key]) is list:
          _dict[key] = [v.to(device) for v in _dict[key]]
      else:
        _dict[key] = _dict[key].to(device)

    return _dict

def main(args):
    data_path = args["dataset"]["path"]
    train_batch_size = args["model"]["train_batch_size"]
    val_batch_size = args["model"]["val_batch_size"]
    check_iter = args["model"]["check_iter"]
    model_save_path = args["model"]["model_save_path"]
    pretrained_model = args["model"]["pretrained_model"]
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

    if os.path.exists(model_save_path):
        my_model = load_pretrained_model(my_model, torch.load(model_save_path))
    elif os.path.exists(pretrained_model):
        my_model = load_pretrained_model(my_model, torch.load(pretrained_model))
    my_model.to(pytorch_device)
    
    pytorch_total_params = sum(p.numel() for p in my_model.parameters())
    print("params: ", pytorch_total_params)
    
    optimizer = optim.Adam(my_model.parameters())
    loss_fn = panoptic_loss(
        center_loss_weight=args["model"]["center_loss_weight"],
        offset_loss_weight=args["model"]["offset_loss_weight"],
        center_loss=args["model"]["center_loss"],
        offset_loss=args["model"]["offset_loss"],
    )

    # prepare dataset
    train_pt_dataset = SemKITTI(
        data_path + "/sequences/",
        imageset="train",
        instance_pkl_path=args["dataset"]["instance_pkl_path"],
    )
    val_pt_dataset = SemKITTI(
        data_path + "/sequences/",
        imageset="val",
        instance_pkl_path=args["dataset"]["instance_pkl_path"],
    )

    train_dataset = voxel_polar_dataset(
        train_pt_dataset,
        args["dataset"],
        grid_size=grid_size,
        ignore_label=0,
        use_aug=True,
    )
    val_dataset = voxel_polar_dataset(
        val_pt_dataset, args["dataset"], grid_size=grid_size, ignore_label=0
    )
    
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        collate_fn=collate_fn_BEV_addPolar,
        shuffle=True,
        num_workers=4,
    )
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        collate_fn=collate_fn_BEV_addPolar,
        shuffle=False,
        num_workers=4,
    )

    # training
    epoch = 0
    best_val_PQ_dagger = 0
    start_training = False
    my_model.train()
    global_iter = 1
    exce_counter = 0
    evaluator = PanopticEval(len(unique_label), None, min_points=50)
    logger = get_logger("/media/anda/hdd31/minh/LW-PSC", "logs_val-all.log")
    time_list = []
    pp_time_list = []

    while epoch < args["model"]["max_epoch"]:
        pbar = tqdm(total=len(train_dataset_loader))
        for i_iter, (
            train_occupancy,
            train_points,
            train_labels,
            train_ssclabels,
            train_gt_center,
            train_gt_offset,
            _,
            _,
            _,
            train_aux_com,
            train_distance_feature,
            train_grid_polar_ind,
            train_return_polar_fea
        ) in enumerate(train_dataset_loader):
            # validation
            if global_iter % check_iter == 0:
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
                if best_val_PQ_dagger < PQ_dagger:
                    best_val_PQ_dagger = PQ_dagger
                    torch.save(my_model.state_dict(), model_save_path)
                logger.info(
                    "Current val PQ dagger is %.3f while the best val PQ dagger is %.3f"
                    % (PQ_dagger * 100, best_val_PQ_dagger * 100)
                )
                logger.info(
                    "Current val PQ is %.3f "
                    % (class_PQ * 100)
                )
                logger.info("Current val miou is %.3f" % (miou * 100))
                logger.info(
                    "Inference time per %d is %.4f seconds, postprocessing time is %.4f seconds per scan"
                    % (val_batch_size, np.mean(time_list), np.mean(pp_time_list))
                )
                
                if start_training:
                    aux_sem_loss, aux_com_loss, sem_l, hm_l, os_l = (
                        np.mean(loss_fn.lost_dict["aux_sem_loss"]),
                        np.mean(loss_fn.lost_dict["aux_com_loss"]),
                        np.mean(loss_fn.lost_dict["semantic_loss"]),
                        np.mean(loss_fn.lost_dict["heatmap_loss"]),
                        np.mean(loss_fn.lost_dict["offset_loss"]),
                    )
                    print(
                        "epoch %d iter %5d, loss: %.3f, aux_sem_loss: %.3f, aux_com_loss: %.3f,  semantic loss: %.3f, heatmap loss: %.3f, offset loss: %.3f\n"
                        % (epoch, i_iter, aux_sem_loss + aux_com_loss + sem_l + hm_l + os_l, aux_sem_loss, aux_com_loss, sem_l, hm_l, os_l)
                    )
                    logger.info("epoch %d iter %5d, loss: %.3f, aux_sem_loss: %.3f, aux_com_loss: %.3f,  semantic loss: %.3f, heatmap loss: %.3f, offset loss: %.3f\n"
                        % (epoch, i_iter, aux_sem_loss + aux_com_loss + sem_l + hm_l + os_l, aux_sem_loss, aux_com_loss, sem_l, hm_l, os_l))
                print("%d exceptions encountered during last training\n" % exce_counter)
                logger.info("%d exceptions encountered during last training\n" % exce_counter)
                exce_counter = 0
                loss_fn.reset_loss_dict()

            # training
            # try:
            train_occupancy_ten = train_occupancy.to(pytorch_device)
            train_points_ten = [
                torch.from_numpy(i).to(pytorch_device).type(torch.float32)
                for i in train_points
            ]
            train_labels_ten = [
                torch.from_numpy(i).to(pytorch_device).type(torch.int32)
                for i in train_labels
            ]
            train_ssclabels_ten = train_ssclabels.type(torch.LongTensor).to(pytorch_device)
            
            train_gt_center_tensor = train_gt_center.to(pytorch_device)
            train_gt_offset_tensor = train_gt_offset.to(pytorch_device)
            train_aux_com_ten = dict_to(train_aux_com, pytorch_device)
            
            train_distance_feature_ten = train_distance_feature.to(pytorch_device)
            train_grid_polar_ind_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid_polar_ind]
            train_return_polar_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_return_polar_fea]
                        
            # forward
            sem_prediction, center, offset, aux_ss_loss, aux_sc_loss = my_model(
                train_occupancy_ten, train_points_ten, train_labels_ten, train_aux_com_ten, train_distance_feature_ten, train_grid_polar_ind_ten, train_return_polar_fea_ten
            )
            # loss
            loss = loss_fn(
                sem_prediction,
                center,
                offset,
                train_ssclabels_ten,
                train_gt_center_tensor,
                train_gt_offset_tensor,
                aux_ss_loss,
                aux_sc_loss
            )
            # backward + optimize
            loss.backward()
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()
            pbar.update(1)
            start_training = True
            global_iter += 1
        pbar.close()
        epoch += 1


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--data_dir", default="data")
    parser.add_argument("-p", "--model_save_path", default="./LW-PSC-all.pt")
    parser.add_argument(
        "-c", "--configs", default="configs/SemanticKITTI_model/Panoptic-PolarNet.yaml"
    )
    parser.add_argument("--pretrained_model", default="empty")

    args = parser.parse_args()
    with open(args.configs, "r") as s:
        new_args = yaml.safe_load(s)
    args = merge_configs(args, new_args)

    print(" ".join(sys.argv))
    print(args)
    main(args)
