import numpy as np
import torch
from torch.utils.data import Dataset

ALL  = np.arange(0,543).tolist()    #468

def get_indexs(L):
    return sorted([i + j * len(L) for i in range(len(L)) for j in range(len(L)) if i>j])

DIST_INDEX = get_indexs(ALL)

# def pre_process(xyz):
#     all   = xyz[:, ALL]#20
#     print('all',all.shape)
#     xyz = torch.cat([ #(none, 106, 2)
#         all,
#     ],1)
#     print('xyz concat', xyz.shape)
#     rd = all[:,:,:2].reshape(-1,len(ALL),1,2)-all[:,:,:2].reshape(-1,1,len(ALL),2)
#     rd = torch.sqrt((rd**2).sum(-1))
#     rd = rd.reshape(-1,len(ALL)*len(ALL))[:,DIST_INDEX]

#     xyz = torch.cat([xyz.reshape(-1,(len(ALL))*2), 
#                          rd,
#                         ],1)
    
#     # fill the nan value with 0
#     xyz[torch.isnan(xyz)] = 0

#     return xyz

class D(Dataset):

    def __init__(self, array, num_classes=23, maxlen=100, training=False):

        self.data = array
        # print('data shape',self.data.shape)
        self.maxlen = maxlen # 537 actually
        self.training = training
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
        # xyz = pre_process(xyz)[:self.maxlen]
        xyz= xyz.reshape(xyz.shape[0],1086)
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