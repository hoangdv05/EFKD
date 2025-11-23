from datasets.coco import COCODatasetFast
from torch.utils.data import DataLoader
from modules.transforms import DiffusionTransform, DataAugmentationTransform
import albumentations as A


def get_coco(config, logger=None, verbose=False):
    """
    Get COCO dataloaders for DermoSegDiff
    """
    if logger:
        print = logger.info

    INPUT_SIZE = config["dataset"]["input_size"]
    DT = DiffusionTransform((INPUT_SIZE, INPUT_SIZE))
    AUGT = DataAugmentationTransform((INPUT_SIZE, INPUT_SIZE))
    
    # Prepare augmentation transforms for training
    pixel_level_transform = AUGT.get_pixel_level_transform(config["augmentation"], img_path_list=[])
    spacial_level_transform = AUGT.get_spacial_level_transform(config["augmentation"])
    tr_aug_transform = A.Compose([
        A.Compose(pixel_level_transform, p=config["augmentation"]["levels"]["pixel"]["p"]), 
        A.Compose(spacial_level_transform, p=config["augmentation"]["levels"]["spacial"]["p"])
    ], p=config["augmentation"]["p"])

    # Get merge strategy and min_mask_area from config
    merge_strategy = config["dataset"].get("merge_strategy", "union")
    min_mask_area = config["dataset"].get("min_mask_area", 100)

    # Prepare training dataset
    tr_dataset = COCODatasetFast(
        mode="tr",
        data_dir=config["dataset"]["data_dir"],
        one_hot=False,
        image_size=config["dataset"]["input_size"],
        aug=tr_aug_transform,
        img_transform=DT.get_forward_transform_img(),
        msk_transform=DT.get_forward_transform_msk(),
        add_boundary_mask=config["dataset"].get("add_boundary_mask", False),
        add_boundary_dist=config["dataset"].get("add_boundary_dist", False),
        merge_strategy=merge_strategy,
        min_mask_area=min_mask_area,
        logger=logger,
    )
    
    # Prepare validation dataset
    vl_dataset = COCODatasetFast(
        mode="vl",
        data_dir=config["dataset"]["data_dir"],
        one_hot=False,
        image_size=config["dataset"]["input_size"],
        img_transform=DT.get_forward_transform_img(),
        msk_transform=DT.get_forward_transform_msk(),
        add_boundary_mask=config["dataset"].get("add_boundary_mask", False),
        add_boundary_dist=config["dataset"].get("add_boundary_dist", False),
        merge_strategy=merge_strategy,
        min_mask_area=min_mask_area,
        logger=logger,
    )
    
    # Note: COCO doesn't have a separate test set, using validation as test
    te_dataset = vl_dataset

    if verbose:
        print("COCO 2014:")
        print(f"├──> Length of training_dataset:   {len(tr_dataset)}")
        print(f"├──> Length of validation_dataset: {len(vl_dataset)}")
        print(f"└──> Length of test_dataset:       {len(te_dataset)}")
        print(f"Merge strategy: {merge_strategy}")
        print(f"Minimum mask area: {min_mask_area}")

    # Prepare dataloaders
    tr_dataloader = DataLoader(tr_dataset, **config["data_loader"]["train"])
    vl_dataloader = DataLoader(vl_dataset, **config["data_loader"]["validation"])
    te_dataloader = DataLoader(te_dataset, **config["data_loader"]["test"])

    return {
        "tr": {"dataset": tr_dataset, "loader": tr_dataloader},
        "vl": {"dataset": vl_dataset, "loader": vl_dataloader},
        "te": {"dataset": te_dataset, "loader": te_dataloader},
    }
