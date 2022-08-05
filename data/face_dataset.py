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
from glob import glob
import torch
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
import PIL
from PIL import Image
import random
import cv2
from collections import defaultdict
import torchvision.transforms as transforms
from util.preprocess import DataProcessor
from .face_utils import get_masked_region


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    """Check a file name is an image file or not.

    Args:
        filename (str|Path): The file name with extension.

    Returns:
        Bool: True if the file is an image file.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class FaceDataset(BaseDataset):
    
    def initialize(self, opt):
        self.opt = opt
        self.data_root = opt.dataroot

        self.is_infer = False

        if osp.exists(opt.flist):
            self.file_paths = open(opt.flist).read().splitlines()

            self.video_name_to_imgs_list_dict = defaultdict(list)
            for line in self.file_paths:
                name = line.split('/')[0]
                self.video_name_to_imgs_list_dict[name].append(line)
        else:
            # The inference stage, we use the generated 3DMM rendered face
            # opt.flist is a video name in this case
            self.is_infer = True

            self.input_rendered_face_paths = sorted(glob(osp.join(opt.input_rendered_face, "*.png")))

            img_folder = osp.join(self.data_root, opt.flist, "face_image")
            image_list = sorted([osp.splitext(img_file)[0] \
                                for img_file in os.listdir(img_folder) if is_image_file(img_file)])
            
            self.file_paths = [osp.join(opt.flist, "face_image", f) for f in image_list]
            self.file_paths = self.file_paths[:len(self.input_rendered_face_paths)]

        transform_list = [transforms.ToTensor()]
        self.to_tensor_transform = transforms.Compose(transform_list)

        self.to_PIL_transform = transforms.ToPILImage()

        self.data_processor = DataProcessor()

        self.align_face = False

        self.use_blended_face = opt.use_blended_face

        self.noise = torch.rand((3, 256, 256))

    def __getitem__(self, index):
        if self.align_face:
            return self.read_aligned_face(index)

        if self.use_blended_face:
            return self.read_blended_face(index)

        ## load the GT face image
        file_path = self.file_paths[index]

        face_img_path = osp.join(self.data_root, file_path + ".jpg")

        img_dir, file_name = osp.split(file_path)
        video_name = osp.dirname(img_dir)

        ## read the face image
        gt_face_img = Image.open(face_img_path).convert('RGB')
        gt_face_img_tensor = self.to_tensor_transform(gt_face_img)
        
        ## read the mask image
        msk_path = osp.join(self.data_root, video_name, "lower_neck_mask", file_name + ".png")
        msk_img = Image.open(msk_path).convert('RGB')
        mask_img_tensor = self.to_tensor_transform(msk_img)

        ## get the masked face image
        # noise = torch.rand_like(gt_face_img_tensor) * mask_img_tensor
        masked_face_img_tensor = gt_face_img_tensor * (1.0 - mask_img_tensor) # (3, H, W)

        ## Get the 3DMM rendered face image
        if not self.is_infer:
            rendered_face_img_path = osp.join(self.data_root, video_name, "deep3dface_512", file_name + ".png")
        else:
            rendered_face_img_path = self.input_rendered_face_paths[index]
        
        rendered_face_img = Image.open(rendered_face_img_path).convert('RGB')

        params = get_params(self.opt, gt_face_img.size)
        transforms = get_transform(self.opt, params, normalize=True)

        masked_face_img_tensor = transforms(self.to_PIL_transform(masked_face_img_tensor))
        rendered_face_img_tensor = transforms(rendered_face_img)
        input_img = torch.cat((masked_face_img_tensor, rendered_face_img_tensor), dim=0) # (6, H, W)

        gt_face_img_tensor = transforms(self.to_PIL_transform(gt_face_img_tensor))

        input_dict = {'label': input_img,
                      'inst': 0,
                      'feat': 0,
                      'image': gt_face_img_tensor,
                      'path': face_img_path}

        return input_dict

    def __len__(self):
        return len(self.file_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FaceDataset'
    
    def read_blended_face(self, index):
        ## load the GT face image
        file_path = self.file_paths[index]

        face_img_path = osp.join(self.data_root, file_path + ".jpg")

        img_dir, file_name = osp.split(file_path)
        video_name = osp.dirname(img_dir)

        ## read the face image
        gt_face_img = cv2.imread(face_img_path)
        gt_face_img_tensor = self.to_tensor_transform(gt_face_img)

        ## Get the 3DMM rendered face image
        if not self.is_infer:
            rendered_face_img_path = osp.join(self.data_root, video_name, "deep3dface_512", file_name + ".png")
        else:
            rendered_face_img_path = self.input_rendered_face_paths[index]
        
        rendered_face_img = cv2.imread(rendered_face_img_path)
        rendered_face_img_tensor = self.to_tensor_transform(rendered_face_img)

        ## Get the binary mask image from the 3DMM rendered face image
        rendered_face_mask_img = get_masked_region(rendered_face_img)[..., None]
        rendered_face_mask_img_tensor = torch.FloatTensor(rendered_face_mask_img) / 255.0
        rendered_face_mask_img_tensor = rendered_face_mask_img_tensor.permute(2, 0, 1) # (1, H, W)

        blended_img_tensor = gt_face_img_tensor * (1 - rendered_face_mask_img_tensor) + \
                             rendered_face_img_tensor * rendered_face_mask_img_tensor
        blended_img_tensor = torch.flip(blended_img_tensor, dims=[0]) # to RGB

        ## Read arbitrary reference image
        curr_video_frames = self.video_name_to_imgs_list_dict[video_name]
        ref_img_path = random.choice(curr_video_frames)

        ref_face_img = Image.open(osp.join(self.data_root, ref_img_path + ".jpg")).convert("RGB")

        params = get_params(self.opt, ref_face_img.size)
        transforms = get_transform(self.opt, params, normalize=True)

        blended_face_img_tensor = transforms(self.to_PIL_transform(blended_img_tensor))
        ref_face_img_tensor = transforms(ref_face_img)
        input_img = torch.cat((blended_face_img_tensor, ref_face_img_tensor), dim=0) # (6, H, W)

        gt_face_img_tensor = transforms(self.to_PIL_transform(gt_face_img_tensor))

        input_dict = {'label': input_img,
                      'inst': 0,
                      'feat': 0,
                      'image': gt_face_img_tensor,
                      'path': face_img_path}

        return input_dict

    def read_aligned_face(self, index):
        ## load the GT face image
        file_path = self.file_paths[index]
        face_img_path = osp.join(self.data_root, file_path + ".jpg")

        img_dir, file_name = osp.split(file_path)
        video_name = osp.dirname(img_dir)

        img_src = PIL.Image.open(face_img_path).convert('RGB')

        abs_video_dir = osp.join(self.data_root, video_name)

        ## Load the landmarks
        lm_path = osp.join(abs_video_dir, "face_image", "landmarks", f"{file_name}.txt")
        # lm_path = osp.join(abs_video_dir, "landmarks", f"{file_name}.txt")

        raw_lm = np.loadtxt(lm_path).astype(np.float32) # (68, 2)
        raw_lm[:, -1] = img_src.size[0] - 1 - raw_lm[:, -1]

        ## Align the face
        img, lm_affine, mat, mat_inv = self.data_processor(np.array(img_src), raw_lm)
        img_tensor = self.to_tensor_transform(img) # to [0, 1]

        ### -------- Load the mask image
        msk_path = osp.join(abs_video_dir, "lower_neck_mask", file_name + ".png")
        msk_img = Image.open(msk_path).convert('RGB')

        msk_img = cv2.warpAffine(np.array(msk_img), mat, (256, 256), borderValue=(0,0,0))
        mask_img_tensor = self.to_tensor_transform(msk_img) # to [0, 1]
        
        ### ------- Create the masked GT image
        noise = self.noise * mask_img_tensor
        masked_face_img_tensor = img_tensor * (1.0 - mask_img_tensor) # (3, H, W), [0, 1]

        ### ------- Load the 3DMM rendered face
        rendered_face_img_path = osp.join(abs_video_dir, "deep3dface_512", file_name + ".png")
        rendered_face_img = Image.open(rendered_face_img_path).convert('RGB')
        rendered_face_img = cv2.warpAffine(np.array(rendered_face_img), mat, (256, 256), borderValue=(0,0,0))
        rendered_face_img =  PIL.Image.fromarray(rendered_face_img)

        params = get_params(self.opt, (img.shape[0], img.shape[1]))
        transforms = get_transform(self.opt, params, normalize=True)

        masked_face_img_tensor = transforms(self.to_PIL_transform(masked_face_img_tensor))
        rendered_face_img_tensor = transforms(rendered_face_img)
        
        input_img = torch.cat((masked_face_img_tensor, rendered_face_img_tensor), dim=0) # (6, H, W)

        gt_face_img_tensor = transforms(self.to_PIL_transform(img_tensor))

        input_dict = {'label': input_img,
                      'inst': 0,
                      'feat': 0,
                      'image': gt_face_img_tensor,
                      'path': face_img_path}

        return input_dict