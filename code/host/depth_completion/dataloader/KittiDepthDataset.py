"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

from PIL import Image
import torch
import numpy as np
import glob
import torchvision
import random
import cv2
from torch.utils.data import DataLoader, Dataset


class KittiDepthDataset(Dataset):
    def __init__(self, data_path, gt_path, setname='train', transform=None, norm_factor=256, invert_depth=False,
                 rgb_dir=None, rgb2gray=False, fill_depth = False, flip = False, blind = False):
        self.data_path = data_path
        self.gt_path = gt_path
        self.setname = setname
        self.transform = transform
        self.norm_factor = norm_factor
        self.invert_depth = invert_depth
        self.rgb_dir = rgb_dir
        self.rgb2gray = rgb2gray
        self.fill_depth = fill_depth
        self.flip = flip
        self.blind = blind

        self.data = list(sorted(glob.iglob(self.data_path + "/**/*.png", recursive=True)))
       
        if self.setname == 'test':
           self.gt=list(sorted(glob.iglob(self.gt_path + "/**/*.png", recursive=True)))

    def __len__(self):
        return len(self.data)

    def mask_create(self, x, is_dilate=False, is_random=False):
        #Shape of x: [H, W]
        H, W = x.shape
        FULL_KERNEL_15 = np.ones((3, 3), np.uint8)
        x_all = np.array((x > 0), dtype=np.float32)

        #Dilate
        C_x = np.array((x == 0), dtype=np.float32)
        if is_dilate:
            x_dilate = cv2.dilate(C_x, FULL_KERNEL_15)
            if 0 not in x_dilate:
                x_dilate = C_x
            x_dilate = 1 - x_dilate
            x_all = x_all * x_dilate

        #Random mask
        if is_random:
            mask_x = random.randint(0, W - 150)
            mask_y = random.randint(0, H - 150)
            mask_size_x = random.randint(100, 150)
            mask_size_y = random.randint(100, 150)
            mask = np.ones_like(x)
            mask[mask_y: mask_y + mask_size_y, mask_x: mask_x + mask_size_x] = 0
            x_all = x_all * mask

        '''plt.plot()
        plt.subplot(2, 1, 1)
        plt.imshow(mask)
        plt.subplot(2, 1, 2)
        plt.imshow(x_all)
        plt.show()
        assert False'''

        return x_all

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        # Check if Data filename is equal to GT filename
#        if self.setname == 'train' or self.setname == 'val':
#            data_path = self.data[item].split(self.setname)[1]
#            gt_path = self.gt[item].split(self.setname)[1]
#
#            assert (data_path[0:25] == gt_path[0:25])  # Check folder name
#
#            data_path = data_path.split('image')[1]
#            gt_path = gt_path.split('image')[1]
#
#            assert (data_path == gt_path)  # Check filename
#
#            # Set the certainty path
#            sep = str(self.data[item]).split('data_depth_velodyne')
#
#        elif self.setname == 'selval':
#            data_path = self.data[item].split('00000')[1]
#            gt_path = self.gt[item].split('00000')[1]
#            assert (data_path == gt_path)
#            # Set the certainty path
#            sep = str(self.data[item]).split('/velodyne_raw/')



        # Read images and convert them to 4D floats
        data = Image.open(str(self.data[item]))
        if self.setname == 'test':
           gt = Image.open(str(self.gt[item]))
        # Read RGB images
        if self.setname == 'train' or self.setname == 'val':
            data_path=str(self.data[item])
            idx=data_path.find('Scene')
            idx2=data_path.find('rgb_')
            scene=data_path[idx:idx2]
            num=data_path[idx2+3:-4]
            rgb_path=self.rgb_dir+"/"+scene+"frames/rgb/Camera_0/rgb"+num+".jpg"
            gt_path=self.gt_path+"/"+scene+"frames/depth/Camera_0/depth"+num+".png"
            rgb = Image.open(rgb_path)
            gt = Image.open(gt_path)

        elif self.setname == 'selval':
            data_path = str(self.data[item])
            idx = data_path.find('velodyne_raw')
            fname = data_path[idx + 12:]
            idx2 = fname.find('velodyne_raw')
            rgb_path = data_path[:idx] + 'image' + fname[:idx2] + 'image' + fname[idx2 + 12:]
            rgb = Image.open(rgb_path)

        elif self.setname == 'test':
            gt_path = str(self.gt[item])
            idx = gt_path.find('000')
            index=gt_path[idx:idx+6]
            rgb_path = self.rgb_dir + '/' + index + "_10.png"
            rgb = Image.open(rgb_path)





        if self.rgb2gray:
            t = torchvision.transforms.Grayscale(1)
            rgb = t(rgb)

        W, H = data.size





        # Apply transformations if given
        if self.transform is not None:
            data = self.transform(data)
            gt = self.transform(gt)
            rgb = self.transform(rgb)
        if self.transform is None and self.setname == 'train':
            crop_lt_u = random.randint(0, W - 1216)
            crop_lt_v = random.randint(0, H - 352)
            data = data.crop((crop_lt_u, crop_lt_v, crop_lt_u+1216, crop_lt_v+352))
            gt = gt.crop((crop_lt_u, crop_lt_v, crop_lt_u + 1216, crop_lt_v + 352))
            rgb = rgb.crop((crop_lt_u, crop_lt_v, crop_lt_u + 1216, crop_lt_v + 352))


        if self.flip and random.randint(0, 1) and self.setname == 'train':
            data = data.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)

        if self.setname == 'test':
           data=data.crop((0,0,1216,352))
           gt=gt.crop((0,0,1216,352))
           rgb=rgb.crop((0,0,1216,352))
        # Convert to numpy
        data = np.array(data, dtype=np.float32)
        gt = np.array(gt, dtype=np.float32)
        #median
        #data = cv2.medianBlur(data,3)
        if self.setname == 'train':
            gt = (gt / 100).astype(np.float32)  # depth clipped to 655.35m for sky
            valid = (gt > 0) & (gt < 655)  # depth clipped to 655.35m for sky
            # convert to disparity
            focal_length = 725.0087  # in pixels
            baseline = 0.532725  # meter
            gt = baseline * focal_length / gt
            gt[~valid] = 0
            gt=gt.astype(np.float32)

        #blind
        if self.blind and (self.setname == 'train'):
            blind_start = random.randint(100, H - 50)
            data[blind_start:blind_start+50, :] = 0

        #gt_fake = np.copy(data)
        #mask = self.mask_create(data, is_dilate=True, is_random=False)
        #data_fake = data * mask


        # define the certainty
        C = (data > 0).astype(float)
        #C = (data_fake > 0).astype(float)

        valid=(data<30000)
        data[~valid]=0
        # Normalize the data
        data = data / self.norm_factor # [0,1]
        #data_fake = data_fake / self.norm_factor

        if self.setname == 'test':
            gt = gt / self.norm_factor

        # Expand dims into Pytorch format
        data = np.expand_dims(data, 0)
        gt = np.expand_dims(gt, 0)
        C = np.expand_dims(C, 0)
        #data_fake = np.expand_dims(data_fake, 0)
        #gt_fake = np.expand_dims(gt_fake, 0)

        # Convert to Pytorch Tensors
        data = torch.tensor(data, dtype=torch.float)
        gt = torch.tensor(gt, dtype=torch.float)
        C = torch.tensor(C, dtype=torch.float)
        #data_fake = torch.tensor(data_fake, dtype=torch.float)
        #gt_fake = torch.tensor(gt_fake, dtype=torch.float)

        # Convert depth to disparity
        if self.invert_depth:
            data[data == 0] = -1
            data = 1 / data
            data[data == -1] = 0

            gt[gt == 0] = -1
            gt = 1 / gt
            gt[gt == -1] = 0

        # Convert RGB image to tensor
        rgb = np.array(rgb, dtype=np.float16)
        rgb /= 255
        if self.rgb2gray:
            rgb = np.expand_dims(rgb, 0)
        else:
            rgb = np.transpose(rgb, (2, 0, 1))
        rgb = torch.tensor(rgb, dtype=torch.float)

        #if self.setname == 'train':
        #    return data_fake, C, gt, item, rgb
        #elif self.setname == 'test':
        #    return data_fake, C, gt, item, rgb
        #else:
        return data, C, gt, item, rgb
