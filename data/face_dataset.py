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
import torch
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

        img_dir, file_name = osp.split(file_path)
        video_name = osp.dirname(img_dir)

        ## read the face image
        gt_face_img = Image.open(face_img_path).convert('RGB')
        
        params = get_params(self.opt, gt_face_img.size)
        transforms = get_transform(self.opt, params, normalize=True)

        gt_face_img_tensor = transforms(gt_face_img)

        ## read the mask image
        msk_path = osp.join(self.data_root, video_name, "lower_mask", file_name + ".png")
        msk_img = Image.open(msk_path).convert('RGB')
        mask_img_tensor = transforms(msk_img)

        ## get the masked face image
        masked_face_img_tensor = gt_face_img_tensor * mask_img_tensor # (3, H, W)

        ## Get the 3DMM rendered face image
        rendered_face_img_path = osp.join(self.data_root, video_name, "deep3dface_512", file_name + ".png")
        
        rendered_face_img = Image.open(rendered_face_img_path).convert('RGB')
        rendered_face_img_tensor = transforms(rendered_face_img)

        input_img = torch.cat((masked_face_img_tensor, rendered_face_img_tensor), dim=0) # (6, H, W)

        input_dict = {'label': input_img,
                      'inst': 0,
                      'feat': 0,
                      'image': gt_face_img_tensor}

        return input_dict

    def __len__(self):
        return len(self.file_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FaceDataset'