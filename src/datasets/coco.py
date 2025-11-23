import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from utils.helper_funcs import (
    calc_edge,
    calc_distance_map,
    normalize
)

np_normalize = lambda x: (x-x.min())/(x.max()-x.min())


class COCODatasetFast(Dataset):
    """
    COCO Dataset for DermoSegDiff
    Converts instance segmentation masks to binary masks
    """
    def __init__(self,
                 mode,
                 data_dir=None,
                 one_hot=True,
                 image_size=224,
                 aug=None,
                 aug_empty=None,
                 transform=None,
                 img_transform=None,
                 msk_transform=None,
                 add_boundary_mask=False,
                 add_boundary_dist=False,
                 logger=None,
                 merge_strategy='union',  # 'union' or 'largest'
                 min_mask_area=100,  # Minimum mask area to consider
                 **kwargs):
        self.print = logger.info if logger else print
        
        # pre-set variables
        self.data_dir = data_dir if data_dir else "/path/to/coco2014"

        # input parameters
        self.one_hot = one_hot
        self.image_size = image_size
        self.aug = aug
        self.aug_empty = aug_empty
        self.transform = transform
        self.img_transform = img_transform
        self.msk_transform = msk_transform
        self.mode = mode
        self.merge_strategy = merge_strategy
        self.min_mask_area = min_mask_area
        
        self.add_boundary_mask = add_boundary_mask
        self.add_boundary_dist = add_boundary_dist

        data_preparer = PrepareCOCO(
            data_dir=self.data_dir, 
            image_size=self.image_size, 
            mode=mode,
            merge_strategy=merge_strategy,
            min_mask_area=min_mask_area,
            logger=logger
        )
        data = data_preparer.get_data()
        X, Y = data["x"], data["y"]

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        
        # Use all data for the specified mode
        self.imgs = X
        self.msks = Y
        
        self.print(f"Loaded {len(self.imgs)} samples for {mode} mode")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data_id = idx
        img = self.imgs[idx]
        msk = self.msks[idx]

        if self.one_hot:
            if len(np.unique(msk)) > 1: 
                msk = (msk - msk.min()) / (msk.max() - msk.min())
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)

        if self.aug:
            if self.mode == "tr" and np.random.rand() > 0.5:
                img_ = np.uint8(torch.moveaxis(img*255, 0, -1).detach().numpy())
                msk_ = np.uint8(torch.moveaxis(msk*255, 0, -1).detach().numpy())
                augmented = self.aug(image=img_, mask=msk_)
                img = torch.moveaxis(torch.tensor(augmented['image'], dtype=torch.float32), -1, 0)
                msk = torch.moveaxis(torch.tensor(augmented['mask'], dtype=torch.float32), -1, 0)
            elif self.aug_empty:  # "tr", "vl", "te"
                img_ = np.uint8(torch.moveaxis(img*255, 0, -1).detach().numpy())
                msk_ = np.uint8(torch.moveaxis(msk*255, 0, -1).detach().numpy())
                augmented = self.aug_empty(image=img_, mask=msk_)
                img = torch.moveaxis(torch.tensor(augmented['image'], dtype=torch.float32), -1, 0)
                msk = torch.moveaxis(torch.tensor(augmented['mask'], dtype=torch.float32), -1, 0)
            img = img.nan_to_num(127)
            img = normalize(img)
            msk = msk.nan_to_num(0)
            msk = normalize(msk)

        if self.add_boundary_mask or self.add_boundary_dist:
            msk_ = np.uint8(torch.moveaxis(msk*255, 0, -1).detach().numpy())
                
        if self.add_boundary_mask:
            boundary_mask = calc_edge(msk_, mode='canny')
            boundary_mask = np_normalize(boundary_mask)
            msk = torch.concatenate([msk, torch.tensor(boundary_mask).unsqueeze(0)], dim=0)

        if self.add_boundary_dist:
            boundary_mask = boundary_mask if self.add_boundary_mask else calc_edge(msk_, mode='canny')
            distance_map = calc_distance_map(boundary_mask, mode='l2')
            distance_map = np_normalize(distance_map)
            msk = torch.concatenate([msk, torch.tensor(distance_map).unsqueeze(0)], dim=0)
        
        if self.img_transform:
            img = self.img_transform(img)
        if self.msk_transform:
            msk = self.msk_transform(msk)
            
        img = img.nan_to_num(0.5)
        msk = msk.nan_to_num(-1)

        sample = {"image": img, "mask": msk, "id": data_id}
        return sample


class PrepareCOCO:
    """
    Prepare COCO dataset by converting instance segmentation to binary masks
    """
    def __init__(self, data_dir, image_size, mode='train', merge_strategy='union', 
                 min_mask_area=100, logger=None, **kwargs):
        self.print = logger.info if logger else print
        
        self.data_dir = data_dir
        self.image_size = image_size
        self.mode = mode  # 'train' or 'val'
        self.merge_strategy = merge_strategy
        self.min_mask_area = min_mask_area
        
        # COCO paths
        if mode == 'tr' or mode == 'train':
            self.img_dir = os.path.join(data_dir, "train2014")
            self.ann_file = os.path.join(data_dir, "annotations", "instances_train2014.json")
            self.mode_name = "train2014"
        elif mode == 'vl' or mode == 'val':
            self.img_dir = os.path.join(data_dir, "val2014")
            self.ann_file = os.path.join(data_dir, "annotations", "instances_val2014.json")
            self.mode_name = "val2014"
        else:
            raise ValueError(f"Mode must be 'tr'/'train' or 'vl'/'val', got {mode}")
        
        self.npy_dir = os.path.join(self.data_dir, "np")
        
        self.print(f"Initializing COCO dataset for {self.mode_name}")
        self.print(f"Images: {self.img_dir}")
        self.print(f"Annotations: {self.ann_file}")

    def __get_data_path(self):
        strategy_str = self.merge_strategy
        x_path = f"{self.npy_dir}/X_{self.mode_name}_{self.image_size}x{self.image_size}_{strategy_str}.npy"
        y_path = f"{self.npy_dir}/Y_{self.mode_name}_{self.image_size}x{self.image_size}_{strategy_str}.npy"
        return {"x": x_path, "y": y_path}

    def __get_transforms(self):
        # transform for image
        img_transform = transforms.Compose([
            transforms.Resize(
                size=[self.image_size, self.image_size],
                interpolation=transforms.functional.InterpolationMode.BILINEAR,
            ),
        ])
        # transform for mask
        msk_transform = transforms.Compose([
            transforms.Resize(
                size=[self.image_size, self.image_size],
                interpolation=transforms.functional.InterpolationMode.NEAREST,
            ),
        ])
        return {"img": img_transform, "msk": msk_transform}

    def _merge_masks(self, masks, areas):
        """
        Merge multiple instance masks into a single binary mask
        Args:
            masks: list of binary masks
            areas: list of mask areas
        Returns:
            merged_mask: single binary mask
        """
        if len(masks) == 0:
            return None
            
        if self.merge_strategy == 'union':
            # Union: any pixel that is 1 in any mask
            merged = np.zeros_like(masks[0])
            for mask in masks:
                merged = np.logical_or(merged, mask)
            return merged.astype(np.uint8)
            
        elif self.merge_strategy == 'largest':
            # Take only the largest mask
            largest_idx = np.argmax(areas)
            return masks[largest_idx]
            
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

    def is_data_existed(self):
        for k, v in self.__get_data_path().items():
            if not os.path.isfile(v):
                return False
        return True

    def prepare_data(self):
        data_path = self.__get_data_path()
        
        self.print("Loading COCO annotations...")
        coco = COCO(self.ann_file)
        
        # Get all image IDs
        img_ids = coco.getImgIds()
        self.print(f"Found {len(img_ids)} images in {self.mode_name}")
        
        self.transforms = self.__get_transforms()
        
        imgs = []
        msks = []
        skipped = 0
        
        self.print("Processing images and masks...")
        for img_id in tqdm(img_ids):
            # Get image info
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            # Check if image exists
            if not os.path.exists(img_path):
                skipped += 1
                continue
            
            # Load image
            try:
                img = read_image(img_path, ImageReadMode.RGB)
            except Exception as e:
                self.print(f"Error loading image {img_path}: {e}")
                skipped += 1
                continue
            
            # Get all annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            if len(anns) == 0:
                # Skip images without annotations
                skipped += 1
                continue
            
            # Convert annotations to masks
            instance_masks = []
            instance_areas = []
            
            for ann in anns:
                # Skip small annotations
                if ann['area'] < self.min_mask_area:
                    continue
                
                # Convert segmentation to mask
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # Polygon format
                        rle = maskUtils.frPyObjects(ann['segmentation'], 
                                                    img_info['height'], 
                                                    img_info['width'])
                        mask = maskUtils.decode(rle)
                        if len(mask.shape) == 3:
                            mask = mask.max(axis=2)
                    else:
                        # RLE format
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], 
                                                        img_info['height'], 
                                                        img_info['width'])
                        else:
                            rle = [ann['segmentation']]
                        mask = maskUtils.decode(rle)
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]
                    
                    instance_masks.append(mask)
                    instance_areas.append(ann['area'])
            
            # Skip if no valid masks
            if len(instance_masks) == 0:
                skipped += 1
                continue
            
            # Merge instance masks into binary mask
            binary_mask = self._merge_masks(instance_masks, instance_areas)
            
            if binary_mask is None or binary_mask.sum() == 0:
                skipped += 1
                continue
            
            # Convert to tensor
            binary_mask = torch.from_numpy(binary_mask).unsqueeze(0).float()
            
            # Apply transforms
            img = self.transforms["img"](img)
            img = (img - img.min()) / (img.max() - img.min())
            
            binary_mask = self.transforms["msk"](binary_mask)
            if len(np.unique(binary_mask)) > 1:
                binary_mask = (binary_mask - binary_mask.min()) / (binary_mask.max() - binary_mask.min())
            elif binary_mask.sum():
                binary_mask = binary_mask / binary_mask.max()
            
            imgs.append(img.numpy())
            msks.append(binary_mask.numpy())
        
        X = np.array(imgs)
        Y = np.array(msks)
        
        self.print(f"Processed {len(X)} images ({skipped} skipped)")
        self.print(f"Image shape: {X.shape}, Mask shape: {Y.shape}")
        
        # Create directory
        Path(self.npy_dir).mkdir(exist_ok=True, parents=True)
        
        # Save data
        self.print("Saving data...")
        np.save(data_path["x"].split(".npy")[0], X)
        np.save(data_path["y"].split(".npy")[0], Y)
        self.print(f"Saved at:\n  X: {data_path['x']}\n  Y: {data_path['y']}")
        return

    def get_data(self):
        data_path = self.__get_data_path()

        self.print("Checking for pre-saved files...")
        if not self.is_data_existed():
            self.print("There are no pre-saved files.")
            self.print("Preparing data...")
            self.prepare_data()
        else:
            self.print(f"Found pre-saved files at {self.npy_dir}")

        self.print("Loading...")
        X = np.load(data_path["x"])
        Y = np.load(data_path["y"])
        self.print(f"Loaded X and Y in npy format: X {X.shape}, Y {Y.shape}")

        return {"x": X, "y": Y}
