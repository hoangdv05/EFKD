import os
import sys
from importlib.machinery import SourceFileLoader

import torch
import torch.nn as nn

# Make sure project root is importable when called as a module
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _load_module_from_file(module_name: str, file_path: str):
    """Dynamically load a Python module from a source file path.
    Ensures sys.modules is populated so subsequent imports work.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    loader = SourceFileLoader(module_name, file_path)
    mod = loader.load_module()
    sys.modules[module_name] = mod
    return mod


def _ensure_transunet_imports(repo_root: str):
    """TransUNet file imports ViT via `from utils.vit import ViT`.
    Create a temporary module mapping so that path resolves to the local transunet/Vit implementation.
    """
    tu_dir = os.path.join(repo_root, "transunet")
    vit_path = os.path.join(tu_dir, "vit.py")
    # Load under canonical name 'utils.vit' expected by transunet.py
    # Create parent 'utils' package shim if needed
    if "utils" not in sys.modules:
        utils_pkg = type(sys)("utils")
        utils_pkg.__path__ = []  # mark as pkg
        sys.modules["utils"] = utils_pkg
    _load_module_from_file("utils.vit", vit_path)


class UNetWrapper(nn.Module):
    """Thin wrapper around the UNet implementation to unify constructor and forward.

    Params
    -----
    in_channels: int
    out_channels: int (1 for binary)
    bilinear: bool
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, bilinear: bool = False, repo_root: str = "."):
        super().__init__()
        try:
            from unet.unet_model import UNet
        except Exception:
            # Fallback: try to load when run from workspace root
            unet_path = os.path.join(repo_root, "unet", "unet_model.py")
            mod = _load_module_from_file("unet.unet_model", unet_path)
            UNet = getattr(mod, "UNet")
        self.model = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=bilinear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AttentionUNetWrapper(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, repo_root: str = "."):
        super().__init__()
        att_path = os.path.join(repo_root, "attention-unet", "model.py")
        mod = _load_module_from_file("attention_unet.model", att_path)
        AttentionUNet = getattr(mod, "AttentionUNet")
        # The referenced implementation uses `in_channel`/`out_channel`
        self.model = AttentionUNet(in_channel=in_channels, out_channel=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepLabV3PlusWrapper(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, output_stride: int = 16, pretrained_backbone: bool = False, repo_root: str = "."):
        super().__init__()
        dl_path = os.path.join(repo_root, "deeplabv3+", "deeplab_resnet.py")
        mod = _load_module_from_file("deeplabv3_plus.deeplab_resnet", dl_path)
        DeepLabv3_plus = getattr(mod, "DeepLabv3_plus")
        self.model = DeepLabv3_plus(nInputChannels=in_channels, n_classes=out_channels, os=output_stride, pretrained=pretrained_backbone, _print=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TransUNetWrapper(nn.Module):
    def __init__(
        self,
        img_dim: int = 224,
        in_channels: int = 3,
        embed_channels: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
        block_num: int = 8,
        patch_dim: int = 16,
        out_channels: int = 1,
        repo_root: str = ".",
    ):
        super().__init__()
        _ensure_transunet_imports(repo_root)
        tu_path = os.path.join(repo_root, "transunet", "transunet.py")
        mod = _load_module_from_file("transunet.transunet", tu_path)
        TransUNet = getattr(mod, "TransUNet")
        self.model = TransUNet(
            img_dim=img_dim,
            in_channels=in_channels,
            out_channels=embed_channels,
            head_num=num_heads,
            mlp_dim=mlp_dim,
            block_num=block_num,
            patch_dim=patch_dim,
            class_num=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MissFormerWrapper(nn.Module):
    def __init__(self, out_channels: int = 1, token_mlp_mode: str = "mix_skip", repo_root: str = "."):
        super().__init__()
        mf_path = os.path.join(repo_root, "missformer", "MISSFormer.py")
        mod = _load_module_from_file("missformer.MISSFormer", mf_path)
        MISSFormer = getattr(mod, "MISSFormer")
        # MISSFormer internally expects 3-channel input; it repeats if single-channel is given
        self.model = MISSFormer(num_classes=out_channels, token_mlp_mode=token_mlp_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SwinUNetWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        img_size: int = 224,
        repo_root: str = ".",
        **kwargs,
    ):
        super().__init__()
        # Load the vision_transformer module which contains SwinUnet class
        su_dir = os.path.join(repo_root, "swin u-net")
        vt_path = os.path.join(su_dir, "vision_transformer.py")
        mod = _load_module_from_file("swin_unet.vision_transformer", vt_path)
        
        # Get the SwinUnet class from the module
        SwinUnet = getattr(mod, "SwinUnet")
        
        # Create a mock config object for SwinUnet
        class MockConfig:
            def __init__(self, img_size, in_channels, out_channels):
                self.DATA = type('obj', (object,), {'IMG_SIZE': img_size})()
                self.MODEL = type('obj', (object,), {
                    'SWIN': type('obj', (object,), {
                        'PATCH_SIZE': 4,
                        'IN_CHANS': in_channels,
                        'EMBED_DIM': 96,
                        'DEPTHS': [2, 2, 6, 2],
                        'NUM_HEADS': [3, 6, 12, 24],
                        'WINDOW_SIZE': 7,
                        'MLP_RATIO': 4.0,
                        'QKV_BIAS': True,
                        'QK_SCALE': None,
                        'PATCH_NORM': True,
                        'APE': False,
                    })(),
                    'DROP_RATE': 0.0,
                    'DROP_PATH_RATE': 0.1,
                })()
                self.TRAIN = type('obj', (object,), {'USE_CHECKPOINT': False})()
        
        config = MockConfig(img_size, in_channels, out_channels)
        
        self.model = SwinUnet(
            config=config,
            img_size=img_size,
            num_classes=out_channels,
            zero_head=False,
            vis=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


