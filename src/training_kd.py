from pathlib import Path
import numpy as np
import random
import torch
from torch.optim import Adam, SGD, AdamW
from utils.helper_funcs import (
    load_config,
    get_model_path,
    get_conf_name,
    print_config,
)
from models import *
from forward.forward_schedules import ForwardSchedule
from forward.forward_process import ForwardProcess
from torch.optim import lr_scheduler
from train_validate_kd import train_kd, validate_kd, evaluate_teacher
from loaders.dataloaders import get_dataloaders
from torch.utils.tensorboard import SummaryWriter
import sys, os
from common.logging import get_logger
from argument import get_argparser, sync_config
import warnings
warnings.filterwarnings('ignore')


# ------------------- params --------------------
argparser = get_argparser()
args = argparser.parse_args(sys.argv[1:])

config = load_config(args.config_file)
config = sync_config(config, args)

logger = get_logger(filename=f"{config['model']['name']}_kd", dir=f"logs/{config['dataset']['name']}")
print_config(config, logger)
logger.info("=== Knowledge Distillation Training ===")

# Display KD configuration summary
logger.info(f"Training ID: {get_conf_name(config)}")
logger.info(f"Device: {config['run']['device']}")
logger.info(f"Epochs: {config['training']['epochs']}")
logger.info(f"Batch size: {config['data_loader']['train']['batch_size']}")

# Display configured loss components
training_loss = config.get("training", {}).get("loss", {})
loss_components = []
if "l2" in training_loss:
    l2_config = training_loss["l2"]
    loss_components.append(f"L2 (coeff={l2_config.get('coefficient', 1.0)})")
if "boundary" in training_loss:
    boundary_config = training_loss["boundary"]
    loss_components.append(f"Boundary (coeff={boundary_config.get('coefficient', 1.0)})")
if "boundary_privileged_kd" in training_loss:
    bpkd_config = training_loss["boundary_privileged_kd"]
    loss_components.append(f"BPKD (coeff={bpkd_config.get('coefficient', 1.0)})")

if loss_components:
    logger.info(f"Loss components: {', '.join(loss_components)}")
logger.info("=" * 60)

# create the writer for tensorboard
writer = SummaryWriter(f'{config["run"]["writer_dir"]}/{config["model"]["name"]}_kd')

# variables
timesteps = config["diffusion"]["schedule"]["timesteps"]
epochs = config["training"]["epochs"]
input_size = config["dataset"]["input_size"]
batch_size = config["data_loader"]["train"]["batch_size"]
img_channels = config["dataset"]["img_channels"]
msk_channels = config["dataset"]["msk_channels"]
ID = get_conf_name(config)

# device
device = torch.device(config["run"]["device"])
logger.info(f"Device is <{device}>")


start_epoch = 0
best_vl_loss = np.Inf
best_vl_losses = {}

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()


# --------- check required dirs --------------------
Path(config["model"]["save_dir"]).mkdir(exist_ok=True, parents=True)


forward_schedule = ForwardSchedule(**config["diffusion"]["schedule"])
forward_process = ForwardProcess(forward_schedule)

# --------------- Datasets and Dataloaders -----------------
tr_dataloader, vl_dataloader = get_dataloaders(config, ["tr", "vl"])


# --------------- Load Teacher Model -----------------
logger.info("=== Loading Teacher Model ===")

# Load teacher checkpoint path early to infer architecture from it if possible
teacher_checkpoint_path = config["model"]["teacher_checkpoint"]
teacher_checkpoint = None
if os.path.exists(teacher_checkpoint_path):
    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location="cpu")
else:
    logger.warning(f"Teacher checkpoint not found: {teacher_checkpoint_path}")
    logger.warning("Training with randomly initialized teacher...")

# Teacher architecture and params (allow override via config, but prefer checkpoint's config if available)
teacher_class_name = config["model"].get("teacher_class", "DermoSegDiff")
teacher_params = config["model"].get("teacher_params", config["model"]["params"])  # fallback to student params

# If the teacher checkpoint contains its training config, prefer that to avoid shape mismatches
use_ckpt_cfg = config["model"].get("teacher_use_ckpt_config", True)
if teacher_checkpoint is not None and use_ckpt_cfg and isinstance(teacher_checkpoint, dict) and ("config" in teacher_checkpoint):
    try:
        ckpt_cfg = teacher_checkpoint["config"]
        ckpt_model_cfg = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg, dict) else {}
        inferred_class = ckpt_model_cfg.get("class", ckpt_model_cfg.get("teacher_class", teacher_class_name))
        inferred_params = ckpt_model_cfg.get("params", ckpt_model_cfg.get("teacher_params", teacher_params))
        teacher_class_name = inferred_class
        teacher_params = inferred_params
        logger.info(f"Using teacher architecture from checkpoint config: class={teacher_class_name}")
    except Exception as e:
        logger.warning(f"Could not infer teacher architecture from checkpoint config ({e}). Falling back to current config.")

logger.info(f"Teacher model class: {teacher_class_name}")
logger.info(f"Teacher model params: {teacher_params}")

Teacher = globals()[teacher_class_name]
teacher_model = Teacher(**teacher_params)

# If we have a checkpoint, try to load its weights into the teacher model
if teacher_checkpoint is not None:
    teacher_state = teacher_checkpoint.get("model", teacher_checkpoint)
    try:
        missing, unexpected = teacher_model.load_state_dict(teacher_state, strict=False)
        if len(missing) or len(unexpected):
            logger.warning(f"Teacher state partially loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        logger.info(f"Loaded teacher from checkpoint: {teacher_checkpoint_path}")
    except Exception as e:
        logger.warning(f"Strict load failed ({e}); attempting shape-filtered load...")
        cur_state = teacher_model.state_dict()
        filtered = {k: v for k, v in teacher_state.items() if k in cur_state and v.shape == cur_state[k].shape}
        teacher_model.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded {len(filtered)}/{len(cur_state)} tensors from teacher checkpoint after filtering")

teacher_model = teacher_model.to(device)
teacher_model.eval()  # Always in eval mode

# Freeze teacher parameters
for param in teacher_model.parameters():
    param.requires_grad = False
logger.info("Teacher model weights frozen")

teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
logger.info(f"Teacher model parameters: {teacher_total_params:,}")


# --------------- Load Student Model -----------------
logger.info("=== Loading Student Model ===")
logger.info(f"Student model class: {config['model']['class']}")
logger.info(f"Student model params: {config['model']['params']}")

Student = globals()[config["model"]["class"]]
student_model = Student(**config["model"]["params"])
student_model = student_model.to(device)

student_total_params = sum(p.numel() for p in student_model.parameters())
student_trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
logger.info(f"Student model parameters: {student_total_params:,} (trainable: {student_trainable_params:,})")

# Compression ratio and detailed comparison
compression_ratio = teacher_total_params / student_total_params
parameter_reduction = (1 - 1/compression_ratio) * 100

logger.info("=== Model Architecture Comparison ===")
logger.info(f"Teacher: {teacher_class_name} | Student: {config['model']['class']}")

# Compare key parameters if available
teacher_cfg = teacher_params
student_cfg = config["model"]["params"]

if "dim_x" in teacher_cfg and "dim_x" in student_cfg:
    logger.info(f"dim_x: Teacher={teacher_cfg['dim_x']} | Student={student_cfg['dim_x']}")
if "dim_g" in teacher_cfg and "dim_g" in student_cfg:
    logger.info(f"dim_g: Teacher={teacher_cfg['dim_g']} | Student={student_cfg['dim_g']}")
if "dim_x_mults" in teacher_cfg and "dim_x_mults" in student_cfg:
    logger.info(f"dim_x_mults: Teacher={teacher_cfg['dim_x_mults']} | Student={student_cfg['dim_x_mults']}")
if "dim_g_mults" in teacher_cfg and "dim_g_mults" in student_cfg:
    logger.info(f"dim_g_mults: Teacher={teacher_cfg['dim_g_mults']} | Student={student_cfg['dim_g_mults']}")
if "resnet_block_groups" in teacher_cfg and "resnet_block_groups" in student_cfg:
    logger.info(f"resnet_block_groups: Teacher={teacher_cfg['resnet_block_groups']} | Student={student_cfg['resnet_block_groups']}")

logger.info(f"Parameters: Teacher={teacher_total_params:,} | Student={student_total_params:,}")
logger.info(f"Compression ratio: {compression_ratio:.2f}x")
logger.info(f"Parameter reduction: {parameter_reduction:.1f}%")

# Student model uses standard Conv2d blocks

logger.info("=" * 50)


tr_prms = config["training"]
optimizer = globals()[tr_prms["optimizer"]["name"]](
    student_model.parameters(), **tr_prms["optimizer"]["params"]
)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", **tr_prms["scheduler"])

# ------------------------ EMA -------------------------------
from ema_pytorch import EMA

try:
    if config["training"]["ema"]["use"]:
        ema = EMA(model=student_model, **config["training"]["ema"]["params"])
        ema.to(device)
        logger.info("EMA enabled for student model")
    else:
        ema = None
        logger.info("EMA disabled")
except KeyError:
    logger.exception("You need to determine the EMA parameters at <config.training>!")
    ema = None


# --------------- Load Student Checkpoint (if continuing) -----------------
if config["run"]["continue_training"] or config["training"]["intial_weights"]["use"]:
    if config["run"]["continue_training"]:
        model_path = get_model_path(name=ID, dir=config["model"]["save_dir"])
    else:
        model_path = config["training"]["intial_weights"]["file_path"]
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        if config["run"]["continue_training"]:
            if checkpoint["epochs"] > checkpoint["epoch"] + 1:
                best_vl_loss = checkpoint["vl_loss"]
                student_model.load_state_dict(checkpoint["model"])
                start_epoch = checkpoint["epoch"] + 1
                optimizer.load_state_dict(checkpoint["optimizer"])
                if ema:
                    ema.load_state_dict(checkpoint["ema"])

                logger.info(f"Loaded student model state (ep:{checkpoint['epoch']+1}/{checkpoint['epochs']}) to continue training from:")
                logger.info(f" -> {model_path}\n")
            else:
                logger.warning("the student model already trained!")
                sys.exit()
        else:
            student_model.load_state_dict(checkpoint["model"])
            if ema:
                ema = EMA(model=student_model, **config["training"]["ema"]["params"])
                ema.to(device)
            logger.info(f"Loaded student initial weights from:")
            logger.info(f" -> {model_path}\n")
            
    except Exception as e:
        logger.warning("There is a problem with loading the previous student model to continue training.")
        logger.warning(f" -> Exception: {e}")
        logger.warning(f" -> Attempted checkpoint path: {model_path}")

        # Debug: try to load checkpoint and report key differences
        try:
            ckpt_debug = torch.load(model_path, map_location="cpu")
            ckpt_sd = ckpt_debug.get("model", ckpt_debug) if isinstance(ckpt_debug, dict) else ckpt_debug
            cur_sd = student_model.state_dict()

            missing = [k for k in cur_sd.keys() if k not in ckpt_sd]
            unexpected = [k for k in ckpt_sd.keys() if k not in cur_sd]
            mismatched = [
                (k, tuple(cur_sd[k].shape), tuple(ckpt_sd[k].shape))
                for k in cur_sd.keys()
                if k in ckpt_sd and tuple(cur_sd[k].shape) != tuple(ckpt_sd[k].shape)
            ]

            logger.warning(f" -> Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}, Shape mismatches: {len(mismatched)}")
            for k in missing[:20]:
                logger.warning(f"    missing: {k}")
            for k in unexpected[:20]:
                logger.warning(f"    unexpected: {k}")
            for k, sm, sc in mismatched[:20]:
                logger.warning(f"    mismatch: {k}: model{sm} vs ckpt{sc}")

            if isinstance(ckpt_debug, dict) and "config" in ckpt_debug:
                ck = ckpt_debug["config"]
                ck_model = ck.get("model", {}) if isinstance(ck, dict) else {}
                logger.warning(f" -> Checkpoint embedded config: class={ck_model.get('class')}, name={ck_model.get('name')}")
        except Exception as ee:
            logger.warning(f" -> Secondary inspection failed: {ee}")

        logger.warning(" --> Do you want to train the student model from the beginning? (y/N):")
        user_decision = input()
        if (user_decision != "y"):
            exit()
else:
    model_path = get_model_path(name=ID, dir=config["model"]["save_dir"])
    if os.path.isfile(model_path):
        logger.warning(f"There is a student model weights at determined directory with desired name: {ID}")
        logger.warning(" --> Do you want to train the student model from the beginning? It will overwrite the current weights! (y/N):")
        user_decision = input()
        if (user_decision != "y"):
            exit()


# --------------- Evaluate Teacher Performance (Baseline) -----------------
teacher_eval_cfg = config.get("training", {}).get("teacher_eval", {})
skip_teacher_eval = bool(teacher_eval_cfg.get("skip", False))
teacher_eval_runs = int(teacher_eval_cfg.get("vl_runs", 1))

if not skip_teacher_eval:
    logger.info("=== Evaluating Teacher Performance (Baseline) ===")
    teacher_vl_losses = evaluate_teacher(
        teacher_model,
        vl_dataloader,
        forward_process,
        device,
        cfg=config,
        vl_runs=teacher_eval_runs,
        logger=logger
    )

    teacher_vl_loss = np.mean([l[0] for l in teacher_vl_losses])
    teacher_loss_names = teacher_vl_losses[0][1].keys()
    teacher_losses_dict = dict((n, np.mean([d[1][n] for d in teacher_vl_losses])) for n in teacher_loss_names)

    teacher_losses_txt = ", ".join([f"{ln}: {v:0.6f}" for ln, v in teacher_losses_dict.items()])
    logger.info(f"Teacher baseline performance: {teacher_vl_loss:.8f} | {teacher_losses_txt}")
else:
    logger.info("=== Skipping Teacher Baseline Evaluation (training.teacher_eval.skip=True) ===")
    teacher_vl_loss = float("nan")


# --------------- Knowledge Distillation Training Loop -----------------
logger.info(f"=== Starting Knowledge Distillation Training for {epochs} epochs ===")

# Log enabled methods
loss_config = config['training']['loss']
enabled_methods = []
for method_name in loss_config.keys():
    if method_name == 'boundary_privileged_kd':
        enabled_methods.append("Boundary-Privileged KD (BPKD)")
    elif method_name in ['l2', 'boundary']:
        enabled_methods.append(f"{method_name.upper()} loss")

logger.info(f"Enabled Methods ({len(enabled_methods)}):")
for i, method in enumerate(enabled_methods, 1):
    logger.info(f"  {i}. {method}")

# Log coefficients
coefficients = {k: v.get('coefficient', v) if isinstance(v, dict) else v 
                for k, v in loss_config.items()}
logger.info(f"Loss Coefficients: {coefficients}")

for epoch in range(start_epoch, epochs):

    # Train student with knowledge distillation
    tr_losses, student_model = train_kd(
        student_model,
        teacher_model,
        tr_dataloader,
        forward_process,
        device,
        optimizer,
        ema=ema,
        cfg=config,
        extra={"skip_steps": 10, "prefix": f"ep:{epoch+1}/{epochs}"},
        logger=logger
    )

    # Validate student
    vl_losses = validate_kd(
        ema.ema_model if ema else student_model,
        teacher_model,
        vl_dataloader,
        forward_process,
        device,
        cfg=config,
        vl_runs=3,
        logger=logger
    )

    tr_loss = np.mean([l[0] for l in tr_losses])
    vl_loss = np.mean([l[0] for l in vl_losses])

    # TensorBoard logging
    configured_losses = list(config["training"].get("loss", {}).keys())
    kd_label = "+".join(configured_losses) if len(configured_losses) > 1 else (configured_losses[0] if configured_losses else "kd")
    scalars = {"Train": tr_loss, "Validation": vl_loss}
    if not skip_teacher_eval:
        scalars["Teacher_Baseline"] = teacher_vl_loss
    writer.add_scalars(
        f"Loss/train vs validation/{kd_label}",
        scalars,
        epoch,
    )

    # ---------- tr losses -------------
    lns = tr_losses[0][1].keys()
    tr_losses_dict = dict((n, np.mean([d[1][n] for d in tr_losses])) for n in lns)
    
    # ---------- vl losses -------------
    assert lns == vl_losses[0][1].keys(), "Ops... reported losses are different between tr and vl!"
    vl_losses_dict = dict((n, np.mean([d[1][n] for d in vl_losses])) for n in lns)

    # --------- add tr, vl scalars (all losses) ------
    for ln, v in tr_losses_dict.items():
        writer.add_scalars(
            f"Losses/{ln.upper()}",
            {"train": v, "validation": vl_losses_dict[ln]},
            epoch,
        )

    # Learning rate scheduling
    scheduler.step(vl_loss)
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar("Learning_Rate", current_lr, epoch)

    # Enhanced logging with method breakdown
    bpkd_losses_tr = {k: v for k, v in tr_losses_dict.items() if k == 'boundary_privileged_kd'}
    base_losses_tr = {k: v for k, v in tr_losses_dict.items() if k in ['l2', 'boundary']}
    
    bpkd_losses_vl = {k: v for k, v in vl_losses_dict.items() if k == 'boundary_privileged_kd'}
    base_losses_vl = {k: v for k, v in vl_losses_dict.items() if k in ['l2', 'boundary']}
    
    # Main training info
    logger.info(f"\n{'-'*60}")
    logger.info(f"EPOCH {epoch+1:03d}/{epochs:03d} - KNOWLEDGE DISTILLATION")
    logger.info(f"{'-'*60}")
    logger.info(f"Overall Loss: Train={tr_loss:0.8f}, Validation={vl_loss:0.8f}")
    
    # Teacher comparison
    if not skip_teacher_eval and np.isfinite(teacher_vl_loss):
        performance_gap = ((vl_loss - teacher_vl_loss) / teacher_vl_loss) * 100
        logger.info(f"Teacher baseline: {teacher_vl_loss:0.8f}, Performance gap: {performance_gap:+.2f}%")
    
    # BPKD method breakdown
    if bpkd_losses_tr:
        logger.info(f"\nBPKD Method:")
        if 'boundary_privileged_kd' in bpkd_losses_tr:
            logger.info(f"  - BPKD           : tr={bpkd_losses_tr['boundary_privileged_kd']:0.6f}, vl={bpkd_losses_vl['boundary_privileged_kd']:0.6f}")
    
    # Base losses
    if base_losses_tr:
        logger.info(f"\nBase Losses:")
        for loss_name, tr_val in base_losses_tr.items():
            vl_val = base_losses_vl[loss_name]
            logger.info(f"  - {loss_name.upper():15s}: tr={tr_val:0.6f}, vl={vl_val:0.6f}")
    
    logger.info(f"{'-'*60}")
    
    # Save best student model
    if best_vl_loss > vl_loss:
        logger.info(
            f">>> Found a better student model: last-vl-loss:{best_vl_loss:0.8f}, new-vl-loss:{vl_loss:0.8f}"
        )
        best_vl_loss = vl_loss
        model_path = get_model_path(name=f"{ID}_best", dir=config["model"]["save_dir"])

        checkpoint = {
            "model": student_model.state_dict(),
            "epoch": epoch,
            "epochs": epochs,
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict() if ema else None,    
            "vl_loss": vl_loss,
            "teacher_vl_loss": teacher_vl_loss,
            "config": config,
        }

        torch.save(checkpoint, model_path)
        logger.info(f"Saved best student model: {model_path}")



# Final summary
if not skip_teacher_eval and np.isfinite(teacher_vl_loss):
    final_performance_gap = ((best_vl_loss - teacher_vl_loss) / teacher_vl_loss) * 100
else:
    final_performance_gap = float("nan")
compression_ratio = teacher_total_params / student_total_params

logger.info("=== Knowledge Distillation Training Completed ===")
if not skip_teacher_eval and np.isfinite(teacher_vl_loss):
    logger.info(f"Teacher baseline performance: {teacher_vl_loss:.6f}")
else:
    logger.info("Teacher baseline performance: skipped")
logger.info(f"Best student performance: {best_vl_loss:.6f}")
if not skip_teacher_eval and np.isfinite(final_performance_gap):
    logger.info(f"Final performance gap: {final_performance_gap:+.2f}%")
logger.info(f"Model compression ratio: {compression_ratio:.2f}x")
logger.info(f"Parameters reduced: {(1 - 1/compression_ratio)*100:.1f}%")

writer.flush()
writer.close()
