#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemKITTI dataloader
"""
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
import pickle
import errno
from torch.utils import data
import dataloader.io_data as SemanticKittiIO
from glob import glob

from .process_panoptic import PanopticLabelGenerator
from .instance_augmentation import instance_augmentation
from scipy.ndimage import distance_transform_edt


def mask_op(data, x_min, x_max):
    mask = (data > x_min) & (data < x_max)
    return mask

def get_mask(pc, lims):
    mask_x = mask_op(pc[:, 0], lims[0][0] + 0.0001, lims[0][1] - 0.0001)
    mask_y = mask_op(pc[:, 1], lims[1][0] + 0.0001, lims[1][1] - 0.0001)
    mask_z = mask_op(pc[:, 2], lims[2][0] + 0.0001, lims[2][1] - 0.0001)
    mask = (mask_x) & (mask_y) & mask_z
    return mask

def augmentation_random_flip(data, flip_type, is_scan=False):
    if flip_type==1:
        if is_scan:
            data[:, 0] = 51.2 - data[:, 0]
        else:
            data = np.flip(data, axis=0).copy()
    elif flip_type==2:
        if is_scan:
            data[:, 1] = -data[:, 1]
        else:
            data = np.flip(data, axis=1).copy()
    elif flip_type==3:
        if is_scan:
            data[:, 0] = 51.2 - data[:, 0]
            data[:, 1] = -data[:, 1]
        else:
            data = np.flip(np.flip(data, axis=0), axis=1).copy()
    return data

class KITTI360(data.Dataset):
    def __init__(
        self, imageset="train"
    ):
        self.voxel_size =  0.2
        self.grid_size = np.array([256, 256, 32], dtype=np.int32)
        self.min_bound = np.array([0, -25.6, -2], dtype=np.float32)
        self.max_bound = np.array([51.2, 25.6, 4.4], dtype=np.float32)
        self.thing_ids = [1, 2, 3, 4, 5, 6]
        self.n_classes = 19
        # self.x_range = (0, 51.2)
        # self.y_range = (-25.6, 25.6)
        # self.z_range = (-2, 4.4)
        
        splits = {
            "train": ["2013_05_28_drive_0004_sync", "2013_05_28_drive_0000_sync", 
                      "2013_05_28_drive_0010_sync","2013_05_28_drive_0002_sync", 
                      "2013_05_28_drive_0003_sync", "2013_05_28_drive_0005_sync", "2013_05_28_drive_0007_sync"],
            "val": ["2013_05_28_drive_0006_sync"],
            "test": ["2013_05_28_drive_0009_sync"],
        }
        
        self.kitti360_root = "dataset-kitti360/KITTI-360"
        self.kitti360_label_root = "dataset-kitti360/SSCBench-KITTI-360"
        self.kitti360_psc_root = "dataset-kitti360/instance_labels_kitti360"
        
        # self.instance_label_root = os.path.join(self.kitti360_preprocess_root, "instance_labels_kitti360")
        self.label_root = os.path.join(self.kitti360_label_root, "labels")
        
        self.imageset = imageset
        if imageset == "train":
            split = splits["train"]
        elif imageset == "val":
            split = splits["val"]
        elif imageset == "test":
            split = splits["test"]
        else:
            raise Exception("Split must be train/val/test")
        
        
        id_map = self.get_match_id()
            
        self.scan = []
        for i_folder in split:
            labels_path = os.path.join(self.label_root, i_folder, "*_1_1.npy")
            
            for label_path in glob(labels_path):
                filename = os.path.basename(label_path)
                frame_id = os.path.splitext(filename)[0][:6]
            
                self.scan.append(
                    {
                        "sequence": i_folder,
                        "frame_id": frame_id,
                        "original_id": id_map[i_folder][frame_id],
                    }
                    
                )
        
        with open("kitti-360.yaml", "r") as stream:
            kitti360 = yaml.safe_load(stream)
        self.learning_map = kitti360["learning_map"]
        thing_class = kitti360["thing_class"]
        self.thing_list = [cl for cl, ignored in thing_class.items() if ignored]

        self.kitti360 = kitti360
        
    def __len__(self):
        "Denotes the total number of samples"
        return len(self.scan)

    def __getitem__(self, index):
        
        scan = self.scan[index]
        sequence = scan["sequence"]
        frame_id = scan["frame_id"]
        original_id = scan["original_id"]
        
        psc_data = os.path.join(self.kitti360_psc_root, sequence, "{}_1_1.pkl". format(frame_id))
        with open(psc_data, "rb") as f:
            psc_data = pickle.load(f)
            semantic_label = psc_data["semantic_labels"].astype(np.uint8)
            instance_label = psc_data["instance_labels"].astype(np.uint8)
        
        pc_path = os.path.join(self.kitti360_root, "data_3d_raw",  sequence, "velodyne_points/data", "{:010d}.bin".format(int(original_id)))
        pc = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))
        # data_tuple = (pc, semantic_label, instance_label)
        xyz = pc[:, :3]
        
        # process semantic point cloud labels
        # voxel_indices = np.floor((xyz - self.min_bound) / self.voxel_size).astype(np.int32)
        # valid_mask = (
        #     (0 <= voxel_indices[:, 0]) & (voxel_indices[:, 0] < self.grid_size[0]) &
        #     (0 <= voxel_indices[:, 1]) & (voxel_indices[:, 1] < self.grid_size[1]) &
        #     (0 <= voxel_indices[:, 2]) & (voxel_indices[:, 2] < self.grid_size[2])
        # )
        # D, H, W = semantic_label.shape
        # valid_mask = (
        #     (0 <= voxel_indices[:, 2]) & (voxel_indices[:, 2] < D) &
        #     (0 <= voxel_indices[:, 1]) & (voxel_indices[:, 1] < H) &
        #     (0 <= voxel_indices[:, 0]) & (voxel_indices[:, 0] < W)
        # )

        # labels = np.full(len(xyz), fill_value=0, dtype=np.int32)  # -1 = invalid

        # valid_indices = voxel_indices[valid_mask]
        # labels[valid_mask] = semantic_label[
        #     valid_indices[:, 2], valid_indices[:, 1], valid_indices[:, 0]
        # ]
        # labels[labels == 255] = 0  # set invalid labels to 0
        # labels_path = os.path.join(self.kitti360_root, "data_3d_labels",  sequence, "{:010d}.label".format(int(original_id)))
        # os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        # labels.tofile(labels_path)
        data_tuple = (pc, semantic_label, instance_label)
        

        # process occupancy grid input
        occupancy = np.zeros(self.grid_size, dtype=np.uint8)
        
        indices_occ = ((xyz - np.array(self.min_bound)) / self.voxel_size).astype(np.int32)
        
        valid_mask = (
            (indices_occ[:, 0] >= 0) & (indices_occ[:, 0] < self.grid_size[0]) &
            (indices_occ[:, 1] >= 0) & (indices_occ[:, 1] < self.grid_size[1]) &
            (indices_occ[:, 2] >= 0) & (indices_occ[:, 2] < self.grid_size[2])
        )
        indices_occ = indices_occ[valid_mask]
        occupancy[indices_occ[:, 0], indices_occ[:, 1], indices_occ[:, 2]] = 1
        data_tuple += (occupancy,)
        
        semantic_label_torch = torch.from_numpy(semantic_label).long()
        # load 3D_OCCUPANCY.bin and 3D_OCCLUDED.occluded
        ssc_labels = {}
        temp = semantic_label_torch.clone().long()
        temp[temp == 255] = self.n_classes
        sem_label_oh = torch.nn.functional.one_hot(temp, num_classes=self.n_classes + 1).permute(3, 0, 1, 2).float()
        scales = [2, 4, 8]
        for scale in scales:
            sem_label_oh_occ = sem_label_oh.clone()
            sem_label_oh_occ[0, :, :, :] = 0
            sem_label_oh_occ[self.n_classes, :, :, :] = 0
            downscaled_sem_label = torch.nn.functional.avg_pool3d(sem_label_oh_occ.unsqueeze(0), kernel_size=scale, stride=scale).squeeze(0)
            downscaled_sem_label = torch.argmax(downscaled_sem_label, dim=0)
            
            sem_label_oh_0_255 = sem_label_oh.clone()
            sem_label_oh_0_255[1:self.n_classes, :, :, :] = 0
            downscaled_sem_label_0_255_oh = torch.nn.functional.avg_pool3d(sem_label_oh_0_255.unsqueeze(0), kernel_size=scale, stride=scale).squeeze(0)
            downscaled_sem_label_0_255 = torch.full_like(downscaled_sem_label, 255)
            downscaled_sem_label_0_255[downscaled_sem_label_0_255_oh[self.n_classes, :, :, :] == 1] = 0
            
            empty_mask = downscaled_sem_label == 0
            downscaled_sem_label[empty_mask] = downscaled_sem_label_0_255[empty_mask]
            
            ssc_labels['1_{}'.format(scale)] = downscaled_sem_label.type(torch.uint8)
                
        ssc_label_1_2 = ssc_labels['1_2'].numpy()
        ssc_label_1_4 = ssc_labels['1_4'].numpy()
        ssc_label_1_8 = ssc_labels['1_8'].numpy()
        data_tuple += (ssc_label_1_2, ssc_label_1_4, ssc_label_1_8)

        return data_tuple

 
    def get_match_id(self):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''
        # Define a dictionary to store the data
        data_dict = {}
        # Get the path to the current file
        current_file_path = os.path.abspath(__file__)

        # Get the folder containing the current file
        current_folder_path = os.path.dirname(current_file_path)

        # Create a new path for the file kitti_360_match.txt in the same folder
        file_path = os.path.join(current_folder_path, 'kitti_360_match.txt')
                                     
        # Open the file for reading
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()  # Split the line into parts based on spaces
                if len(parts) == 3:
                    sequence, id1, id2 = parts  # Assign the values to variables
                    # Check if the sequence already exists in the dictionary
                    id1 = id1.rsplit(".", 1)[0]
                    id2 = id2.rsplit(".", 1)[0]
                    if sequence in data_dict:
                        data_dict[sequence][id2] = id1
                    else:
                        data_dict[sequence] = {id2: id1}

        return data_dict

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


class voxel_polar_dataset(data.Dataset):
    def __init__(
        self,
        in_dataset,
        args,
        grid_size,
        ignore_label=0,
        fixed_volume_space=True,
        use_aug=False,
        max_volume_space=[51.2, 25.6, 4.4],
        min_volume_space=[0, -25.6, -2],
    ):
        "Initialization"
        self.point_cloud_dataset = in_dataset

        self.grid_size = np.asarray(grid_size)
        self.lims = [[0, 51.2], [-25.6, 25.6], [-2, 4.4]]
        self.rotate_aug = args["rotate_aug"] if use_aug else False
        self.ignore_label = ignore_label
        self.flip_aug = args["flip_aug"] if use_aug else False
        self.shuffle_index = args["shuffle_index"] if use_aug else False
        self.instance_aug = args["inst_aug"] if use_aug else False
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

        self.panoptic_label_generator = PanopticLabelGenerator(
            self.grid_size, sigma=args["gt_generator"]["sigma"]
        )

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        "Generates one sample of data"
        data = self.point_cloud_dataset[index]
        flip_type = np.random.randint(0, 4)
        # load more dense labels and instances
        (
            points, dense_label_sem, dense_label_ins, occupancy, label_1_2, label_1_4, label_1_8
        ) = data
        
        # if len(labels.shape) == 1:
        #     labels = labels[..., np.newaxis]
        # if len(insts.shape) == 1:
        #     insts = insts[..., np.newaxis]

        if self.shuffle_index:
            pt_idx = np.random.permutation(np.arange(0, points.shape[0]))
            points = points[pt_idx]
            # if self.point_cloud_dataset.imageset != 'test':
            #     labels = labels[pt_idx]
            
        filter_mask = get_mask(points, self.lims)
        points = points[filter_mask]
        # if self.point_cloud_dataset.imageset != 'test':
        #     labels = labels[filter_mask]
        
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            points[:, :2] = np.dot(points[:, :2], j)
                
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
            
        # random data augmentation by flip
        if self.flip_aug:
            points = augmentation_random_flip(points, flip_type, is_scan=True)
        
        # ===== polar analysis =====
        xyz = points[:,:3]
        ref = points[:,3]
        xyz_pol = cart2polar(points)
        # if len(labels.shape) == 1: labels = labels[..., np.newaxis]
        if len(ref.shape) == 1: ref = ref[..., np.newaxis]
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1)

        if (intervals==0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int32)
        
        current_grid = grid_ind[:np.size(xyz)]
        
        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        
        # process labels
        # processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        # label_voxel_pair = np.concatenate([current_grid,labels],axis = 1)
        # label_voxel_pair = label_voxel_pair[np.lexsort((current_grid[:,0],current_grid[:,1],current_grid[:,2])),:]
        # processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        
        # prepare visiblity feature
        # find max distance index in each angle,height pair
        # valid_label = np.zeros_like(processed_label,dtype=bool)
        valid_label = np.zeros(self.grid_size,dtype=bool)
        valid_label[current_grid[:,0],current_grid[:,1],current_grid[:,2]] = True
        valid_label = valid_label[::-1]
        max_distance_index = np.argmax(valid_label,axis=0)
        max_distance = max_bound[0]-intervals[0]*(max_distance_index)
        distance_feature = np.expand_dims(max_distance, axis=2)-np.transpose(voxel_position[0],(1,2,0))
        distance_feature = np.transpose(distance_feature,(1,2,0))
        # convert to boolean feature
        distance_feature = (distance_feature>0)*-1.
        distance_feature[current_grid[:,2],current_grid[:,0],current_grid[:,1]]=1.

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]),axis = 1)

        return_polar_fea = np.concatenate((return_xyz,ref),axis = 1)
        
        # Panoptic scene completion labels
        if self.flip_aug:
            dense_label_sem = augmentation_random_flip(dense_label_sem, flip_type, is_scan=False)
            dense_label_ins = augmentation_random_flip(dense_label_ins, flip_type, is_scan=False)
                
        learnning_map_inv = self.point_cloud_dataset.kitti360["learning_map_inv"]
        label_map = np.empty(len(learnning_map_inv), dtype=np.uint32)
        label_map[list(learnning_map_inv)] = list(learnning_map_inv.values())

        x, y, z = np.where(dense_label_sem != 255)
        sem_diff255_ind = np.stack((x, y, z), axis=-1)
        flatten_semantic_labels = dense_label_sem[x, y, z].astype(np.uint32)
        flatten_semantic_labels_inv = remap(flatten_semantic_labels, label_map)
        flatten_instance_labels = dense_label_ins[x, y, z].astype(np.uint32)
        flatten_labels = (flatten_instance_labels << 16) | (
            flatten_semantic_labels_inv & 0xFFFF
        )
        flatten_sem_data = flatten_semantic_labels
        flatten_inst_data = flatten_labels


        x_ins, y_ins, z_ins = np.where(dense_label_ins != 0)
        ins_xyz_grid = np.stack((x_ins, y_ins, z_ins), axis=-1)
        fill_label = 0
        # get thing mask
        thing_list = [1, 2, 3, 4, 5, 6]
        mask = np.zeros_like(flatten_sem_data, dtype=bool)
        for label in thing_list:
            mask[flatten_sem_data == label] = True
        inst_label = flatten_inst_data[mask].squeeze()
        unique_label = np.unique(inst_label)
        unique_label_dict = {label: idx + 1 for idx, label in enumerate(unique_label)}
        if inst_label.size > 1:
            inst_label = np.vectorize(unique_label_dict.__getitem__)(inst_label)
            processed_inst = np.ones(self.grid_size[:2], dtype=np.uint8) * fill_label
            inst_voxel_pair = np.concatenate(
                [ins_xyz_grid[:,:2], inst_label[..., np.newaxis]],
                axis=1,
            )
            inst_voxel_pair = inst_voxel_pair[
                np.lexsort((ins_xyz_grid[:,0], ins_xyz_grid[:,1])),
                :,
            ]
            processed_inst = nb_process_inst(np.copy(processed_inst), inst_voxel_pair)
        else:
            processed_inst = None

        # process voxel position
        intervals_voxel = np.asarray([0.2, 0.2, 0.2])
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1 
        voxel_position = (np.indices(self.grid_size)+ 0.5)*intervals_voxel.reshape(dim_array) + min_bound.reshape(dim_array)
        
        xyz_inst = ins_xyz_grid * intervals_voxel + min_bound
        # xyz_inst = xyz_inst.astype(np.float32)

        # weight_map = compute_boundary_aware_weights(processed_inst) # Add weight map in each instance
        
        center_psc, center_points_psc, offsets_psc = self.panoptic_label_generator(
            flatten_inst_data[mask],
            xyz_inst,
            processed_inst,
            voxel_position[:2, :, :, 0],
            unique_label_dict,
            min_bound,
            intervals_voxel,
        )

        occupancy = occupancy.reshape(self.grid_size)
        if self.flip_aug:
            occupancy = augmentation_random_flip(occupancy, flip_type, is_scan=False)

        # label_1_2 = label_1_2.reshape(self.grid_size // 2)
        # invalid_1_2 = invalid_1_2.reshape(self.grid_size // 2)
        # label_1_4 = label_1_4.reshape(self.grid_size // 4)
        # invalid_1_4 = invalid_1_4.reshape(self.grid_size // 4)
        # label_1_8 = label_1_8.reshape(self.grid_size // 8)
        # invalid_1_8 = invalid_1_8.reshape(self.grid_size // 8)
        if self.flip_aug:
            label_1_2 = augmentation_random_flip(label_1_2, flip_type, is_scan=False)
            label_1_4 = augmentation_random_flip(label_1_4, flip_type, is_scan=False)
            label_1_8 = augmentation_random_flip(label_1_8, flip_type, is_scan=False)
        
        
        data_tuple = (
            occupancy,
            points,
            # labels,
            dense_label_sem,
            center_psc,
            offsets_psc,
            flatten_sem_data,
            flatten_inst_data,
            sem_diff255_ind,
            label_1_2,
            label_1_4,
            label_1_8,
            distance_feature,
            grid_ind,
            return_polar_fea,
        )
        return data_tuple


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


# @nb.jit("u1[:,:,:](u1[:,:,:],i8[:,:])", nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = (
                np.argmax(counter)
            )
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(
        counter
    )
    return processed_label


# @nb.jit("u1[:,:](u1[:,:],i8[:,:])", nopython=True, cache=True, parallel=False)
def nb_process_inst(processed_inst, sorted_inst_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_inst_voxel_pair[0, 2]] = 1
    cur_sear_ind = sorted_inst_voxel_pair[0, :2]
    for i in range(1, sorted_inst_voxel_pair.shape[0]):
        cur_ind = sorted_inst_voxel_pair[i, :2]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_inst[cur_sear_ind[0], cur_sear_ind[1]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_inst_voxel_pair[i, 2]] += 1
    processed_inst[cur_sear_ind[0], cur_sear_ind[1]] = np.argmax(counter)
    return processed_inst


def collate_fn_BEV_addPolar(data):
    occupancy_stack = np.stack([d[0] for d in data])
    points_stack = [d[1] for d in data]
    ssclabel2stack = np.stack([d[2] for d in data])
    center2stack = np.stack([d[3] for d in data])
    offset2stack = np.stack([d[4] for d in data])
    flatten_label = [d[5] for d in data]
    flatten_inst = [d[6] for d in data]
    sem_diff255_ind = [d[7] for d in data]
    label_1_2 = np.stack([d[8] for d in data])
    label_1_4 = np.stack([d[10] for d in data])
    label_1_8 = np.stack([d[11] for d in data])
    aux_com = {
        "label_1_2": torch.from_numpy(label_1_2),
        "label_1_4": torch.from_numpy(label_1_4),
        "label_1_8": torch.from_numpy(label_1_8),
    }
    distance_feature = np.stack([d[12] for d in data]).astype(np.float32)
    grid_polar_ind = [d[13] for d in data]
    return_polar_feat = [d[14] for d in data]
    
    return (
        torch.from_numpy(occupancy_stack),
        points_stack,
        torch.from_numpy(ssclabel2stack),
        torch.from_numpy(center2stack),
        torch.from_numpy(offset2stack),
        flatten_label,
        flatten_inst,
        sem_diff255_ind,
        aux_com,
        torch.from_numpy(distance_feature),
        grid_polar_ind,
        return_polar_feat
    )


def remap(labels: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    labels[labels == 255] = 0
    return label_map[labels]


# load Semantic KITTI class info
with open("kitti-360.yaml", "r") as stream:
    semkittiyaml = yaml.safe_load(stream)
SemKITTI_label_name = dict()
for i in sorted(list(semkittiyaml["learning_map"].keys()))[::-1]:
    SemKITTI_label_name[semkittiyaml["learning_map"][i]] = semkittiyaml["labels"][i]
