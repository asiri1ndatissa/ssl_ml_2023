import numpy as np
import random
import torch
from torch.utils.data import Dataset

# ALL  = np.arange(0,543).tolist()    #468

# def get_indexs(L):
#     return sorted([i + j * len(L) for i in range(len(L)) for j in range(len(L)) if i>j])

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

def get_indexes(L):
    return sorted([i + j * len(L) for i in range(len(L)) for j in range(len(L)) if i>j])

DIST_INDEX = get_indexes(RHAND)

LIP_DIST_INDEX = get_indexes(SLIP)

POSE_DIST_INDEX = get_indexes(SPOSE)

EYE_DIST_INDEX = get_indexes(REYE)

NOSE_DIST_INDEX = get_indexes(NOSE)


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

def pre_process(xy,aug):

    # select the lip, right/left hand, right/left eye, pose, nose parts.
    lip   = xy[:, SLIP]#20
    lhand = xy[:, LHAND]#21
    rhand = xy[:, RHAND]#21
    pose = xy[:, SPOSE]#8
    reye = xy[:, REYE]#16
    leye = xy[:, LEYE]#16
    nose = xy[:, NOSE]#4
    
    if aug and random.random()>0.7:
        lhand, rhand = do_hflip_hand(lhand, rhand)
        pose = do_hflip_spose(pose)
        reye,leye = do_hflip_eye(reye,leye)
        lip = do_hflip_slip(lip)
        nose = do_hflip_nose(nose)

    xy = torch.cat([ #(none, 106, 2)
        lhand,
        rhand,
        lip,
        pose,
        reye,
        leye,
        nose,
    ],1)


    # concatenate the frame delta information
    x = torch.cat([xy[1:,:len(LHAND+RHAND),:]-xy[:-1,:len(LHAND+RHAND),:],torch.zeros((1,len(LHAND+RHAND),2))],0)
    
    ld = lhand[:,:,:2].reshape(-1,len(LHAND),1,2)-lhand[:,:,:2].reshape(-1,1,len(LHAND),2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.reshape(-1,len(LHAND)*len(LHAND))[:,DIST_INDEX]
    
    rd = rhand[:,:,:2].reshape(-1,len(LHAND),1,2)-rhand[:,:,:2].reshape(-1,1,len(LHAND),2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.reshape(-1,len(LHAND)*len(LHAND))[:,DIST_INDEX]
    
    lipd = lip[:,:,:2].reshape(-1,len(SLIP),1,2)-lip[:,:,:2].reshape(-1,1,len(SLIP),2)
    lipd = torch.sqrt((lipd**2).sum(-1))
    lipd = lipd.reshape(-1,len(SLIP)*len(SLIP))[:,LIP_DIST_INDEX]
    
    posed = pose[:,:,:2].reshape(-1,len(SPOSE),1,2)-pose[:,:,:2].reshape(-1,1,len(SPOSE),2)
    posed = torch.sqrt((posed**2).sum(-1))
    posed = posed.reshape(-1,len(SPOSE)*len(SPOSE))[:,POSE_DIST_INDEX]
    
    reyed = reye[:,:,:2].reshape(-1,len(REYE),1,2)-reye[:,:,:2].reshape(-1,1,len(REYE),2)
    reyed = torch.sqrt((reyed**2).sum(-1))
    reyed = reyed.reshape(-1,len(REYE)*len(REYE))[:,EYE_DIST_INDEX]
    
    leyed = leye[:,:,:2].reshape(-1,len(LEYE),1,2)-leye[:,:,:2].reshape(-1,1,len(LEYE),2)
    leyed = torch.sqrt((leyed**2).sum(-1))
    leyed = leyed.reshape(-1,len(LEYE)*len(LEYE))[:,EYE_DIST_INDEX]

    dist_hand=torch.sqrt(((lhand-rhand)**2).sum(-1))

    xy = torch.cat([xy.reshape(-1,(len(LHAND+RHAND+REYE+LEYE+NOSE+SLIP+SPOSE))*2), 
                         x.reshape(-1,(len(LHAND+RHAND))*2),
                         ld,
                         rd,
                         lipd,
                         posed,
                         reyed,
                         leyed,
                         dist_hand,
                        ],1)
    
    # fill the nan value with 0
    xy[torch.isnan(xy)] = 0
    
    return xy

class D(Dataset):

    def __init__(self, path, num_classes=23, maxlen=100, training=False):

        self.data = np.load(path, allow_pickle=True)
        self.maxlen = maxlen # 537 actually
        self.training = training
        self.label_map = [[] for _ in range(num_classes)]
        for i, item in enumerate(self.data):
            label = item['label']
            self.label_map[label].append(i)
        if training:
            self.augment = train_augment
        else:
            self.augment = None


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, rec=False):

        tmp = self.data[idx]
        data = tmp['data']
        label = tmp['label']    

        xyz = data.reshape((-1, 543, 2))
        
        # only use the xy coords
        xyz = xyz[:,:,:2]
                
        xyz_flat = xyz.reshape(-1,2)
        m = np.nanmean(xyz_flat,0).reshape(1,1,2)
    
        # apply coords normalization
        xyz = xyz - m #noramlisation to common maen
        xyz = xyz / np.nanstd(xyz_flat, 0).mean() 

        aug=0
        if self.augment is not None:
            xyz = self.augment(xyz)
            aug = 1

        xyz = torch.from_numpy(xyz).float()
        xy = pre_process(xy,aug)[:self.maxlen]


        # xyz= xyz.reshape(xyz.shape[0],1086)
        xyz[torch.isnan(xyz)] = 0

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