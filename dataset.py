import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import bisect
import random
import scipy
from scipy.interpolate import interp1d

LHAND = np.arange(468, 489).tolist() # 21 (0,21)
RHAND = np.arange(512, 533).tolist() # 21 (21,42)
POSE  = np.arange(489, 512).tolist() # 33 (510,543)
FACE  = np.arange(0,468).tolist()    #468  (42,510)

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
    return sorted([i + j * len(L) for i in range(len(L)) for j in range(len(L)) if i>j])

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

def pre_process(xy,aug):

    # select the lip, right/left hand, right/left eye, pose, nose parts.
    lip   = xy[:, SLIP]#20
    lhand = xy[:, LHAND]#21
    rhand = xy[:, RHAND]#21
    pose = xy[:, SPOSE]#8
    reye = xy[:, REYE]#16
    leye = xy[:, LEYE]#16
    nose = xy[:, NOSE]#4
    print('nose',nose)
    
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
    
    
    # TODO
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

def do_random_affine(xy,
    scale  = (0.8,1.5),
    shift  = (-0.1,0.1),
    degree = (-15,15),
    p=0.5
):
    # random scale, shufle, degree augmentation
    if np.random.rand()<p:
        if scale is not None:
            scale_ = np.random.uniform(*scale)
            xy[:,:,0] = scale_*xy[:,:,0]
            scale_ = np.random.uniform(*scale)
            xy[:,:,1] = scale_*xy[:,:,1]

            scale_ = np.random.uniform(*scale)
            xy[:,LHAND,0] = scale_*xy[:,LHAND,0]
            scale_ = np.random.uniform(*scale)
            xy[:,LHAND,1] = scale_*xy[:,LHAND,1]

            scale_ = np.random.uniform(*scale)
            xy[:,RHAND,0] = scale_*xy[:,RHAND,0]
            scale_ = np.random.uniform(*scale)
            xy[:,RHAND,1] = scale_*xy[:,RHAND,1]

        if shift is not None:
            shift_ = np.random.uniform(*shift)
            xy[:,:,0] = xy[:,:,0] + shift_
            shift_ = np.random.uniform(*shift)
            xy[:,:,1] = xy[:,:,1] + shift_

            shift_ = np.random.uniform(*shift)
            xy[:,LHAND,0] = xy[:,LHAND,0] + shift_/2
            shift_ = np.random.uniform(*shift)
            xy[:,LHAND,1] = xy[:,LHAND,1] + shift_/2

            shift_ = np.random.uniform(*shift)
            xy[:,RHAND,0] = xy[:,RHAND,0] + shift_/2
            shift_ = np.random.uniform(*shift)
            xy[:,RHAND,1] = xy[:,RHAND,1] + shift_/2

        if degree is not None:
            degree_ = np.random.uniform(*degree)
            radian = degree_/180*np.pi
            c = np.cos(radian)
            s = np.sin(radian)
            rotate = np.array([
                [c,-s],
                [s, c],
            ]).T
            xy[:,:,:2] = xy[:,:,:2] @rotate

            degree_ = np.random.uniform(*degree)
            radian = degree_/180*np.pi
            c = np.cos(radian)
            s = np.sin(radian)
            rotate = np.array([
                [c,-s],
                [s, c],
            ]).T
            xy[:,RHAND,:2] = xy[:,RHAND,:2] @rotate

            degree_ = np.random.uniform(*degree)
            radian = degree_/180*np.pi
            c = np.cos(radian)
            s = np.sin(radian)
            rotate = np.array([
                [c,-s],
                [s, c],
            ]).T
            xy[:,LHAND,:2] = xy[:,LHAND,:2] @rotate

    return xy
#-----------------------------------------------------
def train_augment(xy):
    xy = do_random_affine(
        xy,
        scale  = (0.8,1.2),
        shift  = (-0.2,0.2),
        degree = (-5,5),
        p=0.7
    )
    return xy


def do_normalise_by_ref(xy, ref):  
    K = xy.shape[-1]
    xy_flat = ref.reshape(-1,K)
    m = np.nanmean(xy_flat,0).reshape(1,1,K)
    s = np.nanstd(xy_flat, 0).mean() 
    xy = xy - m
    xy = xy / s
    return xy

class D(Dataset):

    def __init__(self, path, training=False):

        self.data = np.load(path, allow_pickle=True)
        self.maxlen = 256 # 537 actually
        self.training = training
        self.label_map = [[] for _ in range(23)]
        self.label_map2 = {}
        self.label_map3 = {}
        for i, item in enumerate(self.data):
            label = item['label']
            pid = item['participant_id']

            self.label_map[label].append(i)

            k = pid+label
            if k in self.label_map2:
                self.label_map2[k].append(i)
            else:
                self.label_map2[k] = [i]

            k = pid
            if k in self.label_map3:
                self.label_map3[k].append(i)
            else:
                self.label_map3[k] = [i]

        self.label_map2 = list(self.label_map2.values())

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

        xy = data.reshape((-1, 533, 2))

        shift = 4 
        if self.augment is not None and random.random()>0.6 and len(xy)>5:
            # TODO
            k0 = np.random.randint(0,len(xy))
            k1 = np.random.randint(max(k0-shift,0), min(len(xy), k0+shift))
            xy = xy - np.nan_to_num(xy[k0:k0+1]) + np.nan_to_num(xy[k1:k1+1])
        
        if self.augment is not None and random.random()>0.5:
            # randomly select another sample with the same label
            new_idx = random.choice(self.label_map[label])
            new_xy = self.data[new_idx]['data'].reshape((-1, 533, 3))
            
            if random.random()>0.5:
                # mixup two samples with the same label
                l=min(len(xy),len(new_xy))
                xy[:l,:,:] = (xy[:l,:,:] + new_xy[:l,:,:]) / 2
            elif random.random()>0.5:
                # random select another sample with the same label, shuffle the original coords with the delta of the two selected samples

                new_idx = random.choice(self.label_map[label])
                new_xy2 = self.data[new_idx]['data'].reshape((-1, 533, 3))
                
                l=min(len(xy),len(new_xy),len(new_xy2))
                xy[:l,:,:] = xy[:l,:,:] + new_xy[:l,:,:] - new_xy2[:l,:,:]
            else:
                # randomly replace the right hand / left hand / body part with the selected samples
                l=min(len(xy),len(new_xy))
                if random.random()>0.5:
                    xy[:l,RHAND,:] = new_xy[:l,RHAND,:]
                elif random.random()>0.5:            
                    xy[:l,LHAND,:] = new_xy[:l,LHAND,:]
                else:
                    xy[:l,BODY,:] = new_xy[:l,BODY,:]
            
            # randomly select a slice from the original sequence
            l = len(xy)
            k1 = np.random.randint(0, 1+int(l*0.15))
            k2 = np.random.randint(0, 1+int(l*0.15))
            xy = xy[k1:len(xy)-k2]

        elif self.augment is not None and random.random()>0.5:
            # randomly select another sample with the same label, use the start position of the original sample and the moving information of the selected sample to construct a new sample
            new_idx = random.choice(self.label_map[label])
            new_xy = self.data[new_idx]['data'].reshape((-1, 533, 3))

            x0 = np.nan_to_num(xy[:1,:,:])
            x_diff = new_xy - np.nan_to_num(new_xy[:1,:,:])

            xy = x_diff + x0
            xy[xy==0] = np.nan

            l = len(xy)
            k1 = np.random.randint(0, 1+int(l*0.15))
            k2 = np.random.randint(0, 1+int(l*0.15))
            xy = xy[k1:len(xy)-k2]

        
        # only use the xy coords
        xy = xy[:,:,:2]
        
        if self.augment is not None and random.random()>0.8:
            # randomly resize the original sequence by interpolation
            l,dim,dim2 = xy.shape
            b=range(l)
            f=interp1d(b,xy,axis=0)
            step = np.random.uniform(low=0.5, high=2)
            new_b=list(np.arange(0,l-1,step))+[l-1]
            xy = f(new_b)

        
        xy_flat = xy.reshape(-1,2)
        m = np.nanmean(xy_flat,0).reshape(1,1,2)
    
        # apply coords normalization
        xy = xy - m #noramlisation to common maen
        xy = xy / np.nanstd(xy_flat, 0).mean() 

        aug = 0
        if self.augment is not None:
            # applying data augmentation
            xy = self.augment(xy)
            aug = 1


        xy = torch.from_numpy(xy).float()
        xy = pre_process(xy,aug)[:self.maxlen]
        
        xy[torch.isnan(xy)] = 0

        # padding the sqeuence to a pre-defined max length
        data_pad = torch.zeros((self.maxlen, xy.shape[1]), dtype=torch.float32)
        tot = xy.shape[0]

        if tot <= self.maxlen:
            data_pad[:tot] = xy
        else:
            data_pad[:] = xy[:self.maxlen]

        if not self.training:
            # for validation
            return data_pad, label

        # if training, return a sample with two different augmentations
        if rec == False:
            data2 = self.__getitem__(idx, True)
            return data_pad, label, data2
        else:
            return data_pad


class ConcatDataset(Dataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes