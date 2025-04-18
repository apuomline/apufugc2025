import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from albumentations  import Compose, Rotate, HorizontalFlip, ElasticTransform
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.exposure import equalize_adapthist
from PIL import Image

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


class JointTransform2D:
    """
    for segmentation
    """
    def __init__(self, img_size = (336,544),crop=(32,32), p_flip=0.0, p_rota=0.0, p_scale=0.0,p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.img_size = img_size

    def __call__(self, image, mask):
        #  gamma enhancement
       
        image = image.transpose(1,2,0).astype(np.uint8)
      
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)

        # print(f'jointtransform2d-image.shape:{image.shape}')
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size[0] * scale), int(self.img_size[1] * scale)
            image, mask = (F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR),
                           F.resize(mask, (new_h, new_w),InterpolationMode.NEAREST))

            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size[0], self.img_size[1]))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)

            noise = np.random.normal(loc=0, scale=1, size=(self.img_size[0],self.img_size[1],1)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (0.1, 0.1), (2, 2), (-45, 45),img_size=self.img_size)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)
        # transforming to tensor
        # image, mask = torch.tensor(image),torch.tensor(mask)

        image = F.resize(image,self.img_size)
        mask = F.resize(mask,self.img_size)
        image = F.to_tensor(image)
        image = image.numpy()
        image = torch.as_tensor(image, dtype=torch.float32)

        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)
        return image, mask
    



class NomaskJointTransform2D:
    """
    for segmentation
    """
    def __init__(self, img_size = [336,544],crop=(32,32), p_flip=0.0, p_rota=0.0, p_scale=0.0,p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        # self.long_mask = long_mask
        self.img_size = img_size

    def __call__(self, image):
        #  gamma enhancement
        # print(f'Nomask-jointtransform2d-img-mask-shape:{image.shape}')
        
        image = image.transpose(1,2,0).astype(np.uint8)
      
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
          
        # transforming to PIL image
        # print(f'nomask-jointtransform2d-image.shape:{image.shape}')
        image = F.to_pil_image(image)
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image  = F.crop(image, i, j, h, w)
        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image  = F.hflip(image)
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image = F.rotate(image, angle)
        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size[0] * scale), int(self.img_size[1] * scale)
            image = (F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR))

            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size[0], self.img_size[1]))
            image = F.crop(image, i, j, h, w)

        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)

            noise = np.random.normal(loc=0, scale=1, size=(self.img_size[0],self.img_size[1],1)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (0.1, 0.1), (2, 2), (-45, 45),img_size=self.img_size)
            image = F.affine(image, *affine_params)
    
        image = F.resize(image,self.img_size)
        image = F.to_tensor(image)
        image = image.numpy()
        image = torch.as_tensor(image, dtype=torch.float32)

        return image
    


###该验证转换类别，使用标准的归一化处理
class ValJointTransform2D:
    """
    Validation Joint Transform for Image and Mask.
    """
    def __init__(self, img_size=(336, 544), long_mask=True):
        self.img_size = img_size
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # 将图像和掩码转换为 PIL 图像
        
    
        image = image.transpose(1, 2, 0).astype(np.uint8)
        print(f'jointtransform2d--image.shape:{image.shape},type:{type(image)}')
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        image = F.resize(image,self.img_size)
        mask = F.resize(mask,self.img_size)
        # 对图像进行归一化处理
        image = F.to_tensor(image)  # 将图像转换为 [0, 1] 范围的张量
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准归一化

        # 将掩码转换为张量
        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask





