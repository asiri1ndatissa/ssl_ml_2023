import numpy as np
import torch
from torch.utils.data import Dataset
import random

MAX_FRAMES = 0

ALL  = np.arange(0,543).tolist()    #468

LHAND = np.arange(468, 489).tolist() # 21
RHAND = np.arange(522, 543).tolist() # 21
POSE  = np.arange(489, 522).tolist() # 33
FACE  = np.arange(0,468).tolist()    #468

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
][::2]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
][::2]
NOSE=[
    1,2,98,327
]
SLIP = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
]
SPOSE = (np.array([
    11,13,15,12,14,16,23,24,
])+489).tolist()

BODY = REYE+LEYE+NOSE+SLIP+SPOSE

def get_indexs(L):
    index_pairs = []
    for i in range(len(L)):
        for j in range(len(L)):
            if i > j:
                index_pairs.append(i + j * len(L))
    return sorted(index_pairs)

DIST_INDEX = get_indexs(RHAND)


LIP_DIST_INDEX = get_indexs(SLIP)

POSE_DIST_INDEX = get_indexs(SPOSE)

EYE_DIST_INDEX = get_indexs(REYE)

NOSE_DIST_INDEX = get_indexs(NOSE)

HAND_START = [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,0,5,9,13,0]
HAND_END = [1,2,3,4,6,7,8,10,11,12,14,15,16,18,19,20,5,9,13,17,17]


point_dim = len(LHAND+RHAND+REYE+LEYE+NOSE+SLIP+SPOSE)*2+len(LHAND+RHAND)*2+len(RHAND)+len(POSE_DIST_INDEX)+len(DIST_INDEX)*2 +len(EYE_DIST_INDEX)*2+len(LIP_DIST_INDEX)

def do_hflip_hand(lhand, rhand):
    rhand[...,0] *= -1
    lhand[...,0] *= -1
    rhand, lhand = lhand,rhand
    return lhand, rhand

def do_hflip_spose(spose):
    spose[...,0] *= -1
    spose = spose[:,[3,4,5,0,1,2,7,6]]
    return spose


def do_hflip_eye(reye,leye):
    reye[...,0] *= -1
    leye[...,0] *= -1
    reye, leye = leye,reye
    return reye, leye

def do_hflip_slip(slip):
    slip[...,0] *= -1
    slip = slip[:,[10,9,8,7,6,5,4,3,2,1,0]+[19,18,17,16,15,14,13,12,11]]
    return slip

def do_hflip_nose(nose):
    nose[...,0] *= -1
    nose = nose[:,[0,1,3,2]]
    return nose

def compute_pairwise_distance_between_keypoints(kpTensor, INDEXES, BODY_DIST_INDEX):
    kp_xy = kpTensor[:, :, :2]
    kp_xy_reshape = kp_xy.reshape(-1, len(INDEXES), 1, 2) - kp_xy.reshape(-1, 1, len(INDEXES), 2)
    kp_xy_euclidean_distance = torch.sqrt((kp_xy_reshape ** 2).sum(-1))
    kp_2D_square_matrix = kp_xy_euclidean_distance.reshape(-1, len(INDEXES) * len(INDEXES))
    kp_extract_pairwise_distances = kp_2D_square_matrix[:, BODY_DIST_INDEX]
    return kp_extract_pairwise_distances

def compute_delta_consecutive_frames_xy_hands(kpTensor):
    xyz_lr = kpTensor[:, :len(LHAND+RHAND), :]
    xyz_diff = xyz_lr[1:, :] - xyz_lr[:-1, :]
    x_diff = torch.cat([xyz_diff, torch.zeros((1, len(LHAND+RHAND), 2))], 0)
    return x_diff

def spatial_random_affine(xy,
    scale  = (0.8, 1.2),
    shear  = (-0.15, 0.15),
    shift  = (-0.1, 0.1),
    degree = (-30, 30),
):
    device = xy.device
    center = torch.tensor([0.5, 0.5]).to(device)

    if scale is not None:
        scale = torch.rand(1, device=device) * (scale[1] - scale[0]) + scale[0]
        xy = scale * xy

    if shear is not None:
        shear_x = shear_y = torch.rand(1, device=device) * (shear[1] - shear[0]) + shear[0]
        if torch.rand(1, device=device) < 0.5:
            shear_x = 0.
        else:
            shear_y = 0.
        shear_mat = torch.eye(2, device=device)
        shear_mat[0, 1] = shear_x
        shear_mat[1, 0] = shear_y
        xy = xy @ shear_mat
        center = center + torch.tensor([shear_y, shear_x]).to(device)

    if degree is not None:
        xy -= center
        degree = torch.rand(1, device=device) * (degree[1] - degree[0]) + degree[0]
        radian = degree / 180 * np.pi
        c = torch.cos(radian)
        s = torch.sin(radian)
        rotate_mat = torch.eye(2, device=device)
        rotate_mat[0, 1], rotate_mat[0, 0] = s, c
        rotate_mat[1, 0], rotate_mat[1, 1] = -s, c
        xy = xy @ rotate_mat
        xy += center

    if shift is not None:
        shift = torch.rand(1, device=device) * (shift[1] - shift[0]) + shift[0]
        xy = xy + shift

    return xy


def pre_process(xyz,aug):

    lip   = xyz[:, SLIP]#20
    lhand = xyz[:, LHAND]#21
    rhand = xyz[:, RHAND]#21
    pose = xyz[:, SPOSE]#8
    reye = xyz[:, REYE]#16
    leye = xyz[:, LEYE]#16
    nose = xyz[:, NOSE]#4

    if aug and random.random()>0.6:
        lhand, rhand = do_hflip_hand(lhand, rhand)
        pose = do_hflip_spose(pose)
        reye,leye = do_hflip_eye(reye,leye)
        lip = do_hflip_slip(lip)
        nose = do_hflip_nose(nose)

    xyz = torch.cat([ #(none, 106, 2)
        lhand,
        rhand,
        lip,
        pose,
        reye,
        leye,
        nose,
    ],1)

    delta_consecutive_frames_xy_hands=compute_delta_consecutive_frames_xy_hands(xyz)

    pose_distance = compute_pairwise_distance_between_keypoints(pose, SPOSE, POSE_DIST_INDEX)
    left_hand_distance = compute_pairwise_distance_between_keypoints(lhand, LHAND, DIST_INDEX)
    right_hand_distance = compute_pairwise_distance_between_keypoints(rhand, RHAND, DIST_INDEX)
    lip_distance = compute_pairwise_distance_between_keypoints(lip, SLIP, LIP_DIST_INDEX)
    right_eye_distance = compute_pairwise_distance_between_keypoints(reye, REYE, EYE_DIST_INDEX)
    left_eye_distance = compute_pairwise_distance_between_keypoints(leye, LEYE, EYE_DIST_INDEX)
    distance_between_hands = torch.sqrt(((lhand-rhand)**2).sum(-1))

    xyz = torch.cat([xyz.reshape(-1,(len(LHAND+RHAND+REYE+LEYE+NOSE+SLIP+SPOSE))*2), 
                         delta_consecutive_frames_xy_hands.reshape(-1,(len(LHAND+RHAND))*2),
                         left_hand_distance,
                         right_hand_distance,
                         lip_distance,
                         pose_distance,
                         right_eye_distance,
                         left_eye_distance,
                         distance_between_hands,
                        ],1)

    xyz[torch.isnan(xyz)] = 0

    return xyz


class D(Dataset):

    def __init__(self, array, num_classes=23, maxlen=80, training=False):

        self.data = array
        self.maxlen = maxlen # 537 actually
        self.training = training
        self.MAX_FRAMES = 0
        self.label_map = [[] for _ in range(num_classes)]
        for i, item in enumerate(self.data):
            label = item['label']
            self.label_map[label].append(i)
        # if training:
        #     self.augment = train_augment
        # else:
        #     self.augment = None


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, rec=False):

        tmp = self.data[idx]
        data = tmp['data']
        label = tmp['label']    

        xyz = data.reshape((-1, 543, 3))
        
        # only use the xy coords
        xyz = xyz[:,:,:2]
                
        xyz_flat = xyz.reshape(-1,2)
        m = np.nanmean(xyz_flat,0).reshape(1,1,2)
    
        # apply coords normalization
        xyz = xyz - m #noramlisation to common maen
        xyz = xyz / np.nanstd(xyz_flat, 0).mean() 

        xyz = torch.from_numpy(xyz).float()
        if self.training and random.random()>0.7:
            xyz = spatial_random_affine(xyz)

        xyz = pre_process(xyz,self.training)[:self.maxlen]
        # padding the sqeuence to a pre-defined max length
        data_pad = torch.zeros((self.maxlen, xyz.shape[1]), dtype=torch.float32)
        tot = xyz.shape[0]
        if tot <= self.maxlen:
            data_pad[:tot] = xyz
        else:
            data_pad[:] = xyz[:self.maxlen]

        if not self.training:
            # for validation
            return data_pad, label

        # if training, return a sample with two different augmentations
        if rec == False:
            data2 = self.__getitem__(idx, True)
            return data_pad, label, data2
        else:
            return data_pad