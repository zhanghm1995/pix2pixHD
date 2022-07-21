'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-07-21 10:57:43
Email: haimingzhang@link.cuhk.edu.cn
Description: The Face dataset to load masked face and 3DMM rendered face for training
pix2pixHD model
'''

import os
import os.path as osp
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class FaceDataset(BaseDataset):
    
    def initialize(self, opt):
        self.opt = opt
        self.data_root = opt.dataroot

        self.file_paths = default_flist_reader(opt.flist)

    def __getitem__(self, index):
        ## load the GT face image
        file_path = self.file_paths[index]

        face_img_path = osp.join(self.data_root, file_path + ".jpg")

        dir_path, file_name = osp.split(file_path)


        ## read the face image
        gt_face_img = Image.open(face_img_path).convert('RGB')
        
        params = get_params(self.opt, gt_face_img.size)
        transforms = get_transform(self.opt, params)

        gt_face_img_tensor = transforms(gt_face_img)

        ## read the mask image
        msk_path = osp.join(self.data_root, dir_path, "mask_fg", file_name + ".png")
        msk_img = Image.open(msk_path).convert('RGB')
        mask_img_tensor = transforms(msk_img)

        ## get the masked face image
        masked_face_img_tensor = gt_face_img_tensor * mask_img_tensor

        input_dict = {'label': masked_face_img_tensor,
                      'inst': 0,
                      'feat': 0,
                      'image': gt_face_img_tensor}

        return input_dict

    def __len__(self):
        return len(self.file_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FaceDataset'