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
from torch.utils import data


class SemKITTI(data.Dataset):
    def __init__(self, data_config_path, data_path, imageset='train', return_ref=False, residual=1,
                 residual_path=None, drop_few_static_frames=True,movable = False):
        self.return_ref = return_ref
        self.movable = movable
        with open(data_config_path, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.moving_learning_map = semkittiyaml['moving_learning_map']
        if self.movable:
            self.movable_learning_map = semkittiyaml['movable_learning_map']
        self.imageset = imageset
        if imageset == 'train':
            self.split = semkittiyaml['split']['train']
        elif imageset == 'val':
            self.split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            self.split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.scan_files = {}
        self.residual = residual
        self.residual_files = {}
        for seq in self.split:
            seq = '{0:02d}'.format(int(seq))
            scan_files = []
            scan_files += absoluteFilePaths('/'.join([data_path, str(seq).zfill(2), 'velodyne']))
            scan_files.sort()
            self.scan_files[seq] = scan_files
            if self.residual > 0:
                residual_files = []
                residual_files += absoluteFilePaths('/'.join(
                    [residual_path, str(seq).zfill(2), 'residual_images']))  # residual_images_4  residual_images
                residual_files.sort()
                self.residual_files[seq] = residual_files

        if imageset == 'train' and drop_few_static_frames:
            self.remove_few_static_frames()

        scan_files = []
        residual_files = []
        for seq in self.split:
            seq = '{0:02d}'.format(int(seq))
            scan_files += self.scan_files[seq]
            if self.residual > 0:
                residual_files += self.residual_files[seq]
        self.scan_files = scan_files
        if self.residual > 0:
            self.residual_files = residual_files

    def remove_few_static_frames(self):
        # Developed by Jiadai Sun 2021-11-07
        # This function is used to clear some frames, because too many static frames will lead to a long training time

        remove_mapping_path = "config/train_split_dynamic_pointnumber.txt"
        with open(remove_mapping_path) as fd:
            lines = fd.readlines()
            lines = [line.strip() for line in lines]

        pending_dict = {}  # 加载到dict中
        for line in lines:
            if line != '':
                seq, fid, _ = line.split()
                if int(seq) in self.split:
                    if seq in pending_dict.keys():
                        if fid in pending_dict[seq]:
                            raise ValueError(f"!!!! Duplicate {fid} in seq {seq} in .txt file")
                        pending_dict[seq].append(fid)
                    else:
                        pending_dict[seq] = [fid]

        total_raw_len = 0
        total_new_len = 0
        for seq in self.split:
            seq = '{0:02d}'.format(int(seq))
            if seq in pending_dict.keys():
                raw_len = len(self.scan_files[seq])

                # lidar scan files
                scan_files = self.scan_files[seq]
                useful_scan_paths = [path for path in scan_files if os.path.split(path)[-1][:-4] in pending_dict[seq]]
                self.scan_files[seq] = useful_scan_paths

                if self.residual:
                    residual_files = self.residual_files[seq]
                    useful_residual_paths = [path for path in residual_files if
                                             os.path.split(path)[-1][:-4] in pending_dict[seq]]
                    self.residual_files[seq] = useful_residual_paths
                    print("seq",seq)
                    # print("useful_scan_paths",len(useful_scan_paths))
                    # print("useful_residual_paths",len(useful_residual_paths))
                    assert (len(useful_scan_paths) == len(useful_residual_paths))
                new_len = len(self.scan_files[seq])
                print(f"Seq {seq} drop {raw_len - new_len}: {raw_len} -> {new_len}")
                total_raw_len += raw_len
                total_new_len += new_len
        print(f"Totally drop {total_raw_len - total_new_len}: {total_raw_len} -> {total_new_len}")

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.scan_files)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.scan_files[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            moving_labels = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            if self.movable:
                movable_labels = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            labels = np.fromfile(self.scan_files[index].replace('velodyne', 'labels')[:-3] + 'label',dtype=np.int32).reshape((-1, 1))
            labels = labels & 0xFFFF  # delete high 16 digits binary
            moving_labels = np.vectorize(self.moving_learning_map.__getitem__)(labels)
            if self.movable:
                movable_labels = np.vectorize(self.movable_learning_map.__getitem__)(labels)
        if self.movable:
        # add movable
            data_tuple = (raw_data[:, :3], moving_labels.astype(np.uint8),movable_labels.astype(np.uint8))
        else:
            data_tuple = (raw_data[:, :3], moving_labels.astype(np.uint8))

        if self.return_ref:
            data_tuple += (raw_data[:, 3],)

        if self.residual > 0:
            residual_data = np.load(self.residual_files[index])
            data_tuple += (residual_data,)  # (x y z), label, ref, residual_n

        # print("len(data_tuple)",len(data_tuple))
        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


class spherical_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size,
                 rotate_aug=False, flip_aug=False, transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 return_test=False,
                 fixed_volume_space=True,
                 max_volume_space=[50.15, np.pi, 2], min_volume_space=[1.85, -np.pi, -4],
                 ignore_label = 255):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.transform = transform_aug
        self.trans_std = trans_std
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        # max_volume_space = [50.15, np.pi, 2]  # 最里面一格和最外面用来收集ignore的点，防止与不ignore的点放在一起
        # min_volume_space = [1.85, -np.pi, -4]
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.ignore_label = ignore_label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if self.point_cloud_dataset.movable == False and len(data) == 4: # only moving label
            xyz, moving_labels,sig, residual = data
        elif len(data) == 5:  # with residual
            xyz, moving_labels, movable_labels,sig, residual = data
            # print("moving_labels",moving_labels.shape)
            # print("movable_labels",movable_labels.shape)
        else:
            raise Exception('Return invalid data tuple')

        # 因为只变化了坐标没有变索引，所以标签不需要变
        # random data augmentation by rotation
        # 做旋转 ， 进行数据增强
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        # 做x轴或y轴或z轴镜像
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),  # [0.1, 0.1, 0.1]
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        else:
            max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
            min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
            max_bound = np.max(xyz_pol[:, 1:], axis=0)
            min_bound = np.min(xyz_pol[:, 1:], axis=0)
            max_bound = np.concatenate(([max_bound_r], max_bound))
            min_bound = np.concatenate(([min_bound_r], min_bound))

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any():
            print("Zero interval!")
        # 得到每一个点对应的voxel的索引[rho_idx, theta_yaw, pitch_idx]
        # Clip (limit) the values in an array.
        # np.floor向下取整
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process labels
        # self.ignore_label = 255
        processed_moving_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        moving_label_voxel_pair = np.concatenate([grid_ind, moving_labels], axis=1)  # 每一个点对应的格子和label
        moving_label_voxel_pair = moving_label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]  # 按照pitch yaw rho顺序从小到大排序
        processed_moving_label = nb_process_label(np.copy(processed_moving_label), moving_label_voxel_pair)

        if self.point_cloud_dataset.movable:
            # process movable_labels    
            processed_movable_labels = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
            movable_labels_voxel_pair = np.concatenate([grid_ind, movable_labels], axis=1)  # 每一个点对应的格子和label
            movable_labels_voxel_pair = movable_labels_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]  # 按照pitch yaw rho顺序从小到大排序
            processed_movable_labels = nb_process_label(np.copy(processed_movable_labels), movable_labels_voxel_pair)

        # center data on each voxel for PTnet
        # 这里是每一个点所处的voxel中心的位置，在同一个voxel的点的位置是一样的
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        # 每一个点的位置相对于其voxel中心的偏移量
        return_xyz = xyz_pol - voxel_centers
        # [bias_rho, bias_yaw, bias_pitch, rho, yaw, pitch, x, y]
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 5 or len(data) == 4:  # reflectivity residual
            # [bias_rho, bias_theta, bias_z, rho, theta, z, x, y, reflectivity, residual(1-?)]
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis], residual), axis=1)
        else:
            raise NotImplementedError

        if self.point_cloud_dataset.movable:
            if self.return_test:
                return torch.from_numpy(processed_moving_label).type(torch.LongTensor), \
                        torch.from_numpy(processed_movable_labels).type(torch.LongTensor), \
                    torch.from_numpy(grid_ind), \
                    moving_labels, \
                    movable_labels, \
                    torch.from_numpy(return_fea).type(torch.FloatTensor), index
            else:
                return torch.from_numpy(processed_moving_label).type(torch.LongTensor), \
                torch.from_numpy(processed_movable_labels).type(torch.LongTensor), \
                    torch.from_numpy(grid_ind), \
                    moving_labels, \
                    movable_labels, \
                    torch.from_numpy(return_fea).type(torch.FloatTensor)
        else:
            if self.return_test:
                return torch.from_numpy(processed_moving_label).type(torch.LongTensor), \
                    torch.from_numpy(grid_ind), \
                    moving_labels, \
                    torch.from_numpy(return_fea).type(torch.FloatTensor), index
            else:
                return torch.from_numpy(processed_moving_label).type(torch.LongTensor), \
                    torch.from_numpy(grid_ind), \
                    moving_labels, \
                    torch.from_numpy(return_fea).type(torch.FloatTensor)



@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):  # 每个栅格赋予出现次数最多的label
    label_size = 2
    counter = np.zeros((label_size,), dtype=np.uint16)  # counter计算每个label的数量
    counter[sorted_label_voxel_pair[0, 3]] = 1  # 第一个初始化，先加一
    cur_sear_ind = sorted_label_voxel_pair[0, :3]  # 目标点的栅格坐标索引
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]  # 当前点的栅格坐标索引
        if not np.all(np.equal(cur_ind, cur_sear_ind)):  # 索引不一致，要移动到下一个栅格
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)  # 栅格使用出现次数最多的label
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1  # label计数
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

def collate_fn_BEV(data):
    moving_label = torch.stack([d[0] for d in data])
    grid_ind_stack = [d[1] for d in data]
    point_label = [d[2] for d in data]
    xyz = [d[3] for d in data]
    return moving_label, grid_ind_stack, point_label, xyz

def collate_fn_BEV_test(data):
    moving_label = torch.stack([d[0] for d in data])
    grid_ind_stack = [d[1] for d in data]
    point_label = [d[2] for d in data]
    xyz = [d[3] for d in data]
    index = [d[4] for d in data]
    return moving_label, grid_ind_stack, point_label, xyz, index

def collate_fn_BEV_MF(data):
    # print(len(data))
    moving_label = torch.stack([d[0] for d in data])
    movable_label =  torch.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label_moving = [d[3] for d in data]
    point_label_movable = [d[4] for d in data]
    xyz = [d[5] for d in data]
    return moving_label, movable_label, grid_ind_stack, point_label_moving,point_label_movable,xyz

def collate_fn_BEV_MF_test(data):
    # print(len(data))
    moving_label = torch.stack([d[0] for d in data])
    movable_label =  torch.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label_moving = [d[3] for d in data]
    point_label_movable = [d[4] for d in data]
    xyz = [d[5] for d in data]
    index = [d[6] for d in data]
    return moving_label, movable_label, grid_ind_stack, point_label_moving,point_label_movable,xyz,index

# load Semantic KITTI class info
def get_SemKITTI_label_name_MF(label_mapping):  #
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_moving_label_name = dict()
    SemKITTI_movable_label_name = dict()

    moving_inv_learning_map = semkittiyaml['moving_learning_map_inv']
    movable_inv_learning_map = semkittiyaml['movable_learning_map_inv']

    for i in sorted(list(semkittiyaml['moving_learning_map'].keys()))[::-1]:
        map_i = semkittiyaml['moving_learning_map'][i]
        map_inv_i = semkittiyaml['moving_learning_map_inv'][map_i]
        SemKITTI_moving_label_name[map_i] = semkittiyaml['labels'][map_inv_i]

    for i in sorted(list(semkittiyaml['movable_learning_map'].keys()))[::-1]:
        map_i = semkittiyaml['movable_learning_map'][i]
        map_inv_i = semkittiyaml['movable_learning_map_inv'][map_i]
        SemKITTI_movable_label_name[map_i] = semkittiyaml['labels'][map_inv_i]

    moving_label = np.asarray(sorted(list(SemKITTI_moving_label_name.keys())))[:]
    moving_label_str = [SemKITTI_moving_label_name[x] for x in moving_label]

    movable_label = np.asarray(sorted(list(SemKITTI_movable_label_name.keys())))[:]
    movable_label_str = [SemKITTI_movable_label_name[x] for x in movable_label]

    # print("moving_label",moving_label)
    print("moving_label_str",moving_label_str)
    # print("movable_label",movable_label)
    print("movable_label_str",movable_label_str)
    return moving_label, moving_label_str, moving_inv_learning_map,\
            movable_label,movable_label_str,movable_inv_learning_map


# load Semantic KITTI class info
def get_SemKITTI_label_name(label_mapping):  #
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_moving_label_name = dict()
    moving_inv_learning_map = semkittiyaml['moving_learning_map_inv']
    for i in sorted(list(semkittiyaml['moving_learning_map'].keys()))[::-1]:
        map_i = semkittiyaml['moving_learning_map'][i]
        map_inv_i = semkittiyaml['moving_learning_map_inv'][map_i]
        SemKITTI_moving_label_name[map_i] = semkittiyaml['labels'][map_inv_i]

    moving_label = np.asarray(sorted(list(SemKITTI_moving_label_name.keys())))[:]
    moving_label_str = [SemKITTI_moving_label_name[x] for x in moving_label]
    print("moving_label_str",moving_label_str)
    return moving_label, moving_label_str, moving_inv_learning_map
