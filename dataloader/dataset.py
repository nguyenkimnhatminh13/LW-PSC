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

class SemKITTI(data.Dataset):
    def __init__(
        self, data_path, imageset="train", instance_pkl_path="data"
    ):
        voxel_PSC_dataset_path = (
            "./instance_labels_v3"
        )
        with open("semantic-kitti.yaml", "r") as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml["learning_map"]
        thing_class = semkittiyaml["thing_class"]
        self.thing_list = [cl for cl, ignored in thing_class.items() if ignored]
        self.imageset = imageset
        if imageset == "train":
            split = semkittiyaml["split"]["train"]
        elif imageset == "val":
            split = semkittiyaml["split"]["valid"]
        elif imageset == "test":
            split = semkittiyaml["split"]["test"]
        else:
            raise Exception("Split must be train/val/test")

        self.im_idx = []
        self.psc_idx = []
        self.occ_idx = []
        self.label_1_2_idx = []
        self.invalid_1_2_idx = []
        self.label_1_4_idx = []
        self.invalid_1_4_idx = []
        self.label_1_8_idx = []
        self.invalid_1_8_idx = []
        for i_folder in split:
            self.im_idx += sorted(glob(os.path.join(data_path, str(i_folder).zfill(2), 'velodyne', '*.bin')))
            self.psc_idx += sorted(glob(os.path.join(voxel_PSC_dataset_path, str(i_folder).zfill(2), '*.pkl')))
            self.occ_idx += sorted(glob(os.path.join(data_path, str(i_folder).zfill(2), 'voxels', '*.bin')))
            
            self.label_1_2_idx += sorted(glob(os.path.join(data_path, str(i_folder).zfill(2), 'voxels', '*.label_1_2')))
            self.invalid_1_2_idx += sorted(glob(os.path.join(data_path, str(i_folder).zfill(2), 'voxels', '*.invalid_1_2')))
            self.label_1_4_idx += sorted(glob(os.path.join(data_path, str(i_folder).zfill(2), 'voxels', '*.label_1_4')))
            self.invalid_1_4_idx += sorted(glob(os.path.join(data_path, str(i_folder).zfill(2), 'voxels', '*.invalid_1_4')))
            self.label_1_8_idx += sorted(glob(os.path.join(data_path, str(i_folder).zfill(2), 'voxels', '*.label_1_8')))
            self.invalid_1_8_idx += sorted(glob(os.path.join(data_path, str(i_folder).zfill(2), 'voxels', '*.invalid_1_8')))
            if i_folder == 8:
                self.im_idx = [file for idx, file in enumerate(self.im_idx) if idx % 5 == 0]
                self.psc_idx = [file for idx, file in enumerate(self.psc_idx) if idx % 5 == 0]
                self.occ_idx = [file for idx, file in enumerate(self.occ_idx) if idx % 5 == 0]
                self.label_1_2_idx= [file for idx, file in enumerate(self.label_1_2_idx) if idx % 5 == 0]
                self.invalid_1_2_idx = [file for idx, file in enumerate(self.invalid_1_2_idx) if idx % 5 == 0]
                self.label_1_4_idx = [file for idx, file in enumerate(self.label_1_4_idx) if idx % 5 == 0]
                self.invalid_1_4_idx = [file for idx, file in enumerate(self.invalid_1_4_idx) if idx % 5 == 0]
                self.label_1_8_idx = [file for idx, file in enumerate(self.label_1_8_idx) if idx % 5 == 0]
                self.invalid_1_8_idx = [file for idx, file in enumerate(self.invalid_1_8_idx) if idx % 5 == 0]
        
        # get class distribution weight
        epsilon_w = 0.001
        origin_class = semkittiyaml["content"].keys()
        weights = np.zeros(
            (len(semkittiyaml["learning_map_inv"]) - 1,), dtype=np.float32
        )
        for class_num in origin_class:
            if semkittiyaml["learning_map"][class_num] != 0:
                weights[semkittiyaml["learning_map"][class_num] - 1] += semkittiyaml[
                    "content"
                ][class_num]
        self.CLS_LOSS_WEIGHT = 1 / (weights + epsilon_w)
        self.instance_pkl_path = instance_pkl_path
        self.semkittiyaml = semkittiyaml

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))

        if self.imageset == "test":
            sem_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            inst_data = np.expand_dims(
                np.zeros_like(raw_data[:, 0], dtype=np.uint32), axis=1
            )
        else:
            # annotated_data = np.fromfile(
            #     self.im_idx[index].replace("velodyne", "labels")[:-3] + "label",
            #     dtype=np.uint32,
            # ).reshape((-1, 1))
            annotated_data = np.fromfile(
                self.im_idx[index].replace("velodyne", "labels")[:-3] + "label",
                dtype=np.uint32,
            ).reshape(-1)
            sem_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            sem_data = np.vectorize(self.learning_map.__getitem__)(sem_data)
            inst_data = annotated_data
        data_tuple = (raw_data, sem_data.astype(np.uint8), inst_data)

        with open(self.psc_idx[index], "rb") as f:
            psc_data = pickle.load(f)
        sparse_data_sem = psc_data["semantic_labels"].astype(np.uint8)
        sparse_data_ins = psc_data["instance_labels"].astype(np.uint8)


        data_tuple += (sparse_data_sem, sparse_data_ins)

        # load 3D_OCCUPANCY.bin and 3D_OCCLUDED.occluded
        occupancy_path = self.occ_idx[index]
        occupancy = SemanticKittiIO._read_occupancy_SemKITTI(occupancy_path)
        
        label_com_1_2_path = self.label_1_2_idx[index]
        label_1_2 = SemanticKittiIO._read_label_SemKITTI(label_com_1_2_path)
        invalid_1_2_path = self.invalid_1_2_idx[index]
        invalid_1_2 = SemanticKittiIO._read_invalid_SemKITTI(invalid_1_2_path)
        
        label_com_1_4_path = self.label_1_4_idx[index]
        label_1_4 = SemanticKittiIO._read_label_SemKITTI(label_com_1_4_path)
        invalid_1_4_path = self.invalid_1_4_idx[index]
        invalid_1_4 = SemanticKittiIO._read_invalid_SemKITTI(invalid_1_4_path)
        
        label_com_1_8_path = self.label_1_8_idx[index]
        label_1_8 = SemanticKittiIO._read_label_SemKITTI(label_com_1_8_path)
        invalid_1_8_path = self.invalid_1_8_idx[index]
        invalid_1_8 = SemanticKittiIO._read_invalid_SemKITTI(invalid_1_8_path)
        
        data_tuple += (occupancy, label_1_2, invalid_1_2, label_1_4, invalid_1_4, label_1_8, invalid_1_8)

        return data_tuple

    def save_instance(self, out_dir, min_points=10):
        "instance data preparation"
        instance_dict = {label: [] for label in self.thing_list}
        for data_path in self.im_idx:
            print("process instance for:" + data_path)
            # get x,y,z,ref,semantic label and instance label
            raw_data = np.fromfile(data_path, dtype=np.float32).reshape((-1, 4))
            annotated_data = np.fromfile(
                data_path.replace("velodyne", "labels")[:-3] + "label", dtype=np.uint32
            ).reshape((-1, 1))
            sem_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            sem_data = np.vectorize(self.learning_map.__getitem__)(sem_data)
            inst_data = annotated_data

            # instance mask
            mask = np.zeros_like(sem_data, dtype=bool)
            for label in self.thing_list:
                mask[sem_data == label] = True

            # create unqiue instance list
            inst_label = inst_data[mask].squeeze()
            unique_label = np.unique(inst_label)
            num_inst = len(unique_label)

            inst_count = 0
            for inst in unique_label:
                # get instance index
                index = np.where(inst_data == inst)[0]
                # get semantic label
                class_label = sem_data[index[0]]
                # skip small instance
                if index.size < min_points:
                    continue
                # save
                _, dir2 = data_path.split("/sequences/", 1)
                new_save_dir = (
                    out_dir
                    + "/sequences/"
                    + dir2.replace("velodyne", "instance")[:-4]
                    + "_"
                    + str(inst_count)
                    + ".bin"
                )
                if not os.path.exists(os.path.dirname(new_save_dir)):
                    try:
                        os.makedirs(os.path.dirname(new_save_dir))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                inst_fea = raw_data[index]
                inst_fea.tofile(new_save_dir)
                instance_dict[int(class_label)].append(new_save_dir)
                inst_count += 1
        with open(out_dir + "/instance_path.pkl", "wb") as f:
            pickle.dump(instance_dict, f)


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


class voxel_dataset(data.Dataset):
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
        if self.instance_aug: # False
            self.inst_aug = instance_augmentation(
                self.point_cloud_dataset.instance_pkl_path + "/instance_path.pkl",
                self.point_cloud_dataset.thing_list,
                self.point_cloud_dataset.CLS_LOSS_WEIGHT,
                random_flip=args["inst_aug_type"]["inst_global_aug"],
                random_add=args["inst_aug_type"]["inst_os"],
                random_rotate=args["inst_aug_type"]["inst_global_aug"],
                local_transformation=args["inst_aug_type"]["inst_loc_aug"],
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
            points, labels, _, sparse_data_sem, sparse_data_ins, occupancy, label_1_2, invalid_1_2, label_1_4, invalid_1_4, label_1_8, invalid_1_8
        ) = data
        
        # if len(labels.shape) == 1:
        #     labels = labels[..., np.newaxis]
        # if len(insts.shape) == 1:
        #     insts = insts[..., np.newaxis]

        if self.shuffle_index:
            pt_idx = np.random.permutation(np.arange(0, points.shape[0]))
            points = points[pt_idx]
            if self.point_cloud_dataset.imageset != 'test':
                labels = labels[pt_idx]
            
        filter_mask = get_mask(points, self.lims)
        points = points[filter_mask]
        if self.point_cloud_dataset.imageset != 'test':
            labels = labels[filter_mask]
        
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
        
        # process PSC label
        sem_label = sparse_data_sem[..., -1:]
        sem_xyz_grid = sparse_data_sem[..., :3]
        ins_label = sparse_data_ins[..., -1:]
        ins_xyz_grid = sparse_data_ins[..., :3]

        fill_label = 0
        grid_size = np.asarray(self.grid_size).astype(np.int32)

        # dense semantic voxel process
        label_voxel_pair_sem = np.concatenate([sem_xyz_grid, sem_label], axis=-1)
        label_voxel_pair_sem = label_voxel_pair_sem[
            np.lexsort((sem_xyz_grid[:, 0], sem_xyz_grid[:, 1], sem_xyz_grid[:, 2])),
            :,
        ].astype(np.int32)
        dense_label_sem = np.ones(grid_size) * fill_label
        dense_label_sem = nb_process_label(dense_label_sem, label_voxel_pair_sem)

        # dense instance voxel process
        label_voxel_pair_ins = np.concatenate([ins_xyz_grid, ins_label], axis=-1)
        label_voxel_pair_ins = label_voxel_pair_ins[
            np.lexsort((ins_xyz_grid[:, 0], ins_xyz_grid[:, 1], ins_xyz_grid[:, 2])),
            :,
        ].astype(np.int32)
        dense_label_ins = np.ones(grid_size) * fill_label

        if len(label_voxel_pair_ins) != 0:
            dense_label_ins = nb_process_label(dense_label_ins, label_voxel_pair_ins)
        
        if self.flip_aug:
            dense_label_sem = augmentation_random_flip(dense_label_sem, flip_type, is_scan=False)
            dense_label_ins = augmentation_random_flip(dense_label_ins, flip_type, is_scan=False)
                
        learnning_map_inv = self.point_cloud_dataset.semkittiyaml["learning_map_inv"]
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

        # get thing mask
        thing_list = [1, 2, 3, 4, 5, 6, 7, 8]
        mask = np.zeros_like(flatten_sem_data, dtype=bool)
        for label in thing_list:
            mask[flatten_sem_data == label] = True

        inst_label = flatten_inst_data[mask].squeeze()
        unique_label = np.unique(inst_label)
        unique_label_dict = {label: idx + 1 for idx, label in enumerate(unique_label)}
        if inst_label.size > 1:
            inst_label = np.vectorize(unique_label_dict.__getitem__)(inst_label)
            processed_inst = np.ones(grid_size[:2], dtype=np.uint8) * fill_label
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
        if inst_label.size > 1:
            weight_map = compute_boundary_aware_weights(processed_inst) # Add weight map in each instance
        else:
            weight_map = np.ones(self.grid_size[:2], dtype=np.float32)
            
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

        label_1_2 = label_1_2.reshape(self.grid_size // 2)
        invalid_1_2 = invalid_1_2.reshape(self.grid_size // 2)
        label_1_4 = label_1_4.reshape(self.grid_size // 4)
        invalid_1_4 = invalid_1_4.reshape(self.grid_size // 4)
        label_1_8 = label_1_8.reshape(self.grid_size // 8)
        invalid_1_8 = invalid_1_8.reshape(self.grid_size // 8)
        if self.flip_aug:
            label_1_2 = augmentation_random_flip(label_1_2, flip_type, is_scan=False)
            invalid_1_2 = augmentation_random_flip(invalid_1_2, flip_type, is_scan=False)
            label_1_4 = augmentation_random_flip(label_1_4, flip_type, is_scan=False)
            invalid_1_4 = augmentation_random_flip(invalid_1_4, flip_type, is_scan=False)
            label_1_8 = augmentation_random_flip(label_1_8, flip_type, is_scan=False)
            invalid_1_8 = augmentation_random_flip(invalid_1_8, flip_type, is_scan=False)
        
        # aux_com = {
        #     "label_1_2": torch.from_numpy(label_1_2),
        #     "invalid_1_2": torch.from_numpy(invalid_1_2),
        #     "label_1_4": torch.from_numpy(label_1_4),
        #     "invalid_1_4": torch.from_numpy(invalid_1_4),
        #     "label_1_8": torch.from_numpy(label_1_8),
        #     "invalid_1_8": torch.from_numpy(invalid_1_8),
        # }
        
        data_tuple = (
            occupancy,
            points,
            labels,
            dense_label_sem,
            center_psc,
            offsets_psc,
            flatten_sem_data,
            flatten_inst_data,
            sem_diff255_ind,
            label_1_2,
            invalid_1_2,
            label_1_4,
            invalid_1_4,
            label_1_8,
            invalid_1_8,
            weight_map
        )

        return data_tuple


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
        if self.instance_aug: # False
            self.inst_aug = instance_augmentation(
                self.point_cloud_dataset.instance_pkl_path + "/instance_path.pkl",
                self.point_cloud_dataset.thing_list,
                self.point_cloud_dataset.CLS_LOSS_WEIGHT,
                random_flip=args["inst_aug_type"]["inst_global_aug"],
                random_add=args["inst_aug_type"]["inst_os"],
                random_rotate=args["inst_aug_type"]["inst_global_aug"],
                local_transformation=args["inst_aug_type"]["inst_loc_aug"],
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
            points, labels, _, sparse_data_sem, sparse_data_ins, occupancy, label_1_2, invalid_1_2, label_1_4, invalid_1_4, label_1_8, invalid_1_8
        ) = data
        
        # if len(labels.shape) == 1:
        #     labels = labels[..., np.newaxis]
        # if len(insts.shape) == 1:
        #     insts = insts[..., np.newaxis]

        if self.shuffle_index:
            pt_idx = np.random.permutation(np.arange(0, points.shape[0]))
            points = points[pt_idx]
            if self.point_cloud_dataset.imageset != 'test':
                labels = labels[pt_idx]
            
        filter_mask = get_mask(points, self.lims)
        points = points[filter_mask]
        if self.point_cloud_dataset.imageset != 'test':
            labels = labels[filter_mask]
        
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
        if len(labels.shape) == 1: labels = labels[..., np.newaxis]
        if len(ref.shape) == 1: ref = ref[..., np.newaxis]
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1)

        if (intervals==0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int32)
        
        current_grid = grid_ind[:np.size(labels)]
        
        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        
        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([current_grid,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((current_grid[:,0],current_grid[:,1],current_grid[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        
        # prepare visiblity feature
        # find max distance index in each angle,height pair
        valid_label = np.zeros_like(processed_label,dtype=bool)
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
        
        # process PSC label
        sem_label = sparse_data_sem[..., -1:]
        sem_xyz_grid = sparse_data_sem[..., :3]
        ins_label = sparse_data_ins[..., -1:]
        ins_xyz_grid = sparse_data_ins[..., :3]

        fill_label = 0
        grid_size = np.asarray(self.grid_size).astype(np.int32)

        # dense semantic voxel process
        label_voxel_pair_sem = np.concatenate([sem_xyz_grid, sem_label], axis=-1)
        label_voxel_pair_sem = label_voxel_pair_sem[
            np.lexsort((sem_xyz_grid[:, 0], sem_xyz_grid[:, 1], sem_xyz_grid[:, 2])),
            :,
        ].astype(np.int32)
        dense_label_sem = np.ones(grid_size) * fill_label
        dense_label_sem = nb_process_label(dense_label_sem, label_voxel_pair_sem)

        # dense instance voxel process
        label_voxel_pair_ins = np.concatenate([ins_xyz_grid, ins_label], axis=-1)
        label_voxel_pair_ins = label_voxel_pair_ins[
            np.lexsort((ins_xyz_grid[:, 0], ins_xyz_grid[:, 1], ins_xyz_grid[:, 2])),
            :,
        ].astype(np.int32)
        dense_label_ins = np.ones(grid_size) * fill_label

        if len(label_voxel_pair_ins) != 0:
            dense_label_ins = nb_process_label(dense_label_ins, label_voxel_pair_ins)
        
        if self.flip_aug:
            dense_label_sem = augmentation_random_flip(dense_label_sem, flip_type, is_scan=False)
            dense_label_ins = augmentation_random_flip(dense_label_ins, flip_type, is_scan=False)
                
        learnning_map_inv = self.point_cloud_dataset.semkittiyaml["learning_map_inv"]
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

        # get thing mask
        thing_list = [1, 2, 3, 4, 5, 6, 7, 8]
        mask = np.zeros_like(flatten_sem_data, dtype=bool)
        for label in thing_list:
            mask[flatten_sem_data == label] = True

        inst_label = flatten_inst_data[mask].squeeze()
        unique_label = np.unique(inst_label)
        unique_label_dict = {label: idx + 1 for idx, label in enumerate(unique_label)}
        if inst_label.size > 1:
            inst_label = np.vectorize(unique_label_dict.__getitem__)(inst_label)
            processed_inst = np.ones(grid_size[:2], dtype=np.uint8) * fill_label
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

        label_1_2 = label_1_2.reshape(self.grid_size // 2)
        invalid_1_2 = invalid_1_2.reshape(self.grid_size // 2)
        label_1_4 = label_1_4.reshape(self.grid_size // 4)
        invalid_1_4 = invalid_1_4.reshape(self.grid_size // 4)
        label_1_8 = label_1_8.reshape(self.grid_size // 8)
        invalid_1_8 = invalid_1_8.reshape(self.grid_size // 8)
        if self.flip_aug:
            label_1_2 = augmentation_random_flip(label_1_2, flip_type, is_scan=False)
            invalid_1_2 = augmentation_random_flip(invalid_1_2, flip_type, is_scan=False)
            label_1_4 = augmentation_random_flip(label_1_4, flip_type, is_scan=False)
            invalid_1_4 = augmentation_random_flip(invalid_1_4, flip_type, is_scan=False)
            label_1_8 = augmentation_random_flip(label_1_8, flip_type, is_scan=False)
            invalid_1_8 = augmentation_random_flip(invalid_1_8, flip_type, is_scan=False)
        
        
        data_tuple = (
            occupancy,
            points,
            labels,
            dense_label_sem,
            center_psc,
            offsets_psc,
            flatten_sem_data,
            flatten_inst_data,
            sem_diff255_ind,
            label_1_2,
            invalid_1_2,
            label_1_4,
            invalid_1_4,
            label_1_8,
            invalid_1_8,
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


def collate_fn_BEV(data):
    occupancy_stack = np.stack([d[0] for d in data])
    points_stack = [d[1] for d in data]
    labels = [d[2] for d in data]
    ssclabel2stack = np.stack([d[3] for d in data])
    center2stack = np.stack([d[4] for d in data])
    offset2stack = np.stack([d[5] for d in data])
    flatten_label = [d[6] for d in data]
    flatten_inst = [d[7] for d in data]
    sem_diff255_ind = [d[8] for d in data]
    label_1_2 = np.stack([d[9] for d in data])
    invalid_1_2 = np.stack([d[10] for d in data])
    label_1_4 = np.stack([d[11] for d in data])
    invalid_1_4 = np.stack([d[12] for d in data])
    label_1_8 = np.stack([d[13] for d in data])
    invalid_1_8 = np.stack([d[14] for d in data])
    aux_com = {
        "label_1_2": torch.from_numpy(label_1_2),
        "invalid_1_2": torch.from_numpy(invalid_1_2),
        "label_1_4": torch.from_numpy(label_1_4),
        "invalid_1_4": torch.from_numpy(invalid_1_4),
        "label_1_8": torch.from_numpy(label_1_8),
        "invalid_1_8": torch.from_numpy(invalid_1_8),
    }
    # weight_map = np.stack([d[15] for d in data]).astype(np.float32)
    return (
        torch.from_numpy(occupancy_stack),
        points_stack,
        labels,
        torch.from_numpy(ssclabel2stack),
        torch.from_numpy(center2stack),
        torch.from_numpy(offset2stack),
        flatten_label,
        flatten_inst,
        sem_diff255_ind,
        aux_com,
        # torch.from_numpy(weight_map)
    )

def compute_boundary_aware_weights(labeled_mask, weight_factors=(4, 2, 1), base_thresholds=(0.3, 0.5, 0.6)):
    """
    Compute boundary-aware weights for each pixel in a labeled 2D mask.

    Args:
        labeled_mask (np.ndarray): Labeled mask of shape (H, W) where each object has a unique label.
        weight_factors (tuple): Weight factors for the regions (closer to boundary -> higher weight).
        thresholds (tuple): Distance thresholds to divide the regions.

    Returns:
        np.ndarray: Weight map of shape (H, W) with boundary-aware weights.
    """
    H, W = labeled_mask.shape
    weight_map = np.zeros((H, W), dtype=np.float32)

    # Get unique object labels (excluding background, label 0)
    unique_labels = np.unique(labeled_mask)
    unique_labels = unique_labels[unique_labels != 0]
     
    for label in unique_labels:
        # Create a binary mask for the current object
        object_mask = (labeled_mask == label).astype(np.uint8)

        # Extract the boundary of the object
        # dilated_mask = binary_dilation(object_mask)
        # boundary_mask = dilated_mask ^ object_mask  # XOR to get the boundary

        # Compute the distance transform from the boundary
        # Compute the gradient of the binary mask
        distance_map = distance_transform_edt(object_mask)  # Distance from the boundary
        
        # Normalize thresholds based on the maximum distance for the object
        # max_distance = distance_map.max()
        # thresholds = [max_distance * t for t in base_thresholds]
        
        # Assign weights based on distance thresholds
        # region_weights = np.zeros_like(distance_map, dtype=np.float32)
        # region_weights[(distance_map < thresholds[0]) & (distance_map > 0)] = weight_factors[0]
        # region_weights[(distance_map < thresholds[1]) & (distance_map >= thresholds[0])] = weight_factors[1]
        # region_weights[(distance_map < thresholds[2]) & (distance_map >= thresholds[1])] = weight_factors[2]
        # region_weights[(distance_map > thresholds[2])] = weight_factors[3]
        
        region_weights = np.zeros_like(distance_map, dtype=np.float32)
        region_weights[(distance_map < 1.3) & (distance_map > 0)] = weight_factors[0]
        region_weights[(distance_map < 2.1) & (distance_map >= 1.3)] = weight_factors[1]
        # region_weights[(distance_map < thresholds[2]) & (distance_map >= thresholds[1])] = weight_factors[2]
        region_weights[(distance_map > 2.1)] = weight_factors[2]

        # Add the weights for this object to the final weight map
        weight_map += region_weights

    return weight_map

def collate_fn_BEV_addPolar(data):
    occupancy_stack = np.stack([d[0] for d in data])
    points_stack = [d[1] for d in data]
    labels = [d[2] for d in data]
    ssclabel2stack = np.stack([d[3] for d in data])
    center2stack = np.stack([d[4] for d in data])
    offset2stack = np.stack([d[5] for d in data])
    flatten_label = [d[6] for d in data]
    flatten_inst = [d[7] for d in data]
    sem_diff255_ind = [d[8] for d in data]
    label_1_2 = np.stack([d[9] for d in data])
    invalid_1_2 = np.stack([d[10] for d in data])
    label_1_4 = np.stack([d[11] for d in data])
    invalid_1_4 = np.stack([d[12] for d in data])
    label_1_8 = np.stack([d[13] for d in data])
    invalid_1_8 = np.stack([d[14] for d in data])
    aux_com = {
        "label_1_2": torch.from_numpy(label_1_2),
        "invalid_1_2": torch.from_numpy(invalid_1_2),
        "label_1_4": torch.from_numpy(label_1_4),
        "invalid_1_4": torch.from_numpy(invalid_1_4),
        "label_1_8": torch.from_numpy(label_1_8),
        "invalid_1_8": torch.from_numpy(invalid_1_8),
    }
    distance_feature = np.stack([d[15] for d in data]).astype(np.float32)
    grid_polar_ind = [d[16] for d in data]
    return_polar_feat = [d[17] for d in data]
    
    return (
        torch.from_numpy(occupancy_stack),
        points_stack,
        labels,
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
with open("semantic-kitti.yaml", "r") as stream:
    semkittiyaml = yaml.safe_load(stream)
SemKITTI_label_name = dict()
for i in sorted(list(semkittiyaml["learning_map"].keys()))[::-1]:
    SemKITTI_label_name[semkittiyaml["learning_map"][i]] = semkittiyaml["labels"][i]
