import torch
import numpy as np
from loss import p_losses_kd
from models import *


def get_print():
    try:
        from common.logging import get_logger
        logger = get_logger()
        print = logger.info
        return print
    except ImportError:
        return print


def train_kd(
    student_model,
    teacher_model,
    dataloader,
    forward_process,
    device,
    optimizer,
    cfg,
    ema=None,
    extra={"skip_steps": 10, "prefix": None},
    logger=None
):
    """
    Knowledge Distillation training function using L2, Boundary, and EFKD losses
    """
    if ema: 
        student_model = ema.model
    
    student_model.train()
    teacher_model.eval()  # Teacher is always in eval mode
    
    losses = []
    
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size = batch["image"].shape[0]
        batch_imgs = batch["image"].to(device)
        batch_msks = batch["mask"].to(device)

        # Random timestep for diffusion process
        t = torch.randint(
            1, forward_process.forward_schedule.timesteps, (batch_size,), device=device
        ).long()

        # Student forward pass with KD losses
        student_loss, student_losses_dict = p_losses_kd(
            forward_process,
            student_model,
            teacher_model,
            x_start=batch_msks,
            g=batch_imgs,
            t=t,
            cfg=cfg
        )

        losses.append((student_loss.item(), student_losses_dict, batch_size))

        # Backpropagation only for student
        student_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if ema: 
            ema.update()

        # Logging
        if "skip_steps" in extra.keys():
            if step % extra["skip_steps"] == 0:
                tr_x_total = np.sum([l[-1] for l in losses])

                # Calculate average losses
                tr_losses_dict = dict()
                for tr_loss in losses:
                    for ln, v in tr_loss[1].items():
                        try:
                            tr_losses_dict[ln] += v * tr_loss[-1]
                        except:
                            tr_losses_dict[ln] = v * tr_loss[-1]
                for k, v in tr_losses_dict.items():
                    tr_losses_dict[k] /= tr_x_total

                extra_tr_losses_txt = ", ".join(
                    [f"{ln}: {v:0.6f}" for ln, v in tr_losses_dict.items()]
                )

                prefix = extra.get("prefix", None)
                txt_items = ([prefix,] if prefix else [])
                txt_items.append(f"step:{step:03d}/{len(dataloader)}")
                txt_items.append(
                    f"student-losses > {extra_tr_losses_txt}"
                )
                
                if logger:
                    logger.info(", ".join(txt_items))
                else:
                    print(", ".join(txt_items))

    return losses, student_model


@torch.no_grad()
def validate_kd(
    student_model,
    teacher_model,
    dataloader,
    forward_process,
    device,
    cfg,
    vl_runs=3,
    logger=None
):
    """
    Knowledge Distillation validation function using L2, Boundary, and EFKD losses
    """
    
    losses = []
    student_model.eval()
    teacher_model.eval()
    
    for step, batch in enumerate(dataloader):

        batch_size = batch["image"].shape[0]
        batch_imgs = batch["image"].to(device)
        batch_msks = batch["mask"].to(device)
        
        _vl_losses = []
        for _ in range(vl_runs):
            t = torch.randint(
                1, forward_process.forward_schedule.timesteps, (batch_size,), device=device
            ).long()
            
            # Student validation loss
            loss, losses_dict = p_losses_kd(
                forward_process,
                student_model,
                teacher_model,
                x_start=batch_msks,
                g=batch_imgs,
                t=t,
                cfg=cfg
            )
            _vl_losses.append((loss.item(), losses_dict))
        
        _vl_avg_loss = np.mean([l[0] for l in _vl_losses])
        _vl_avg_losses_dict = {}
        for k in _vl_losses[0][1].keys():        
            v = np.mean([l[1][k] for l in _vl_losses])
            _vl_avg_losses_dict[k]=v
    
        losses.append((_vl_avg_loss, _vl_avg_losses_dict, batch_size))

    return losses


@torch.no_grad()
def evaluate_teacher(
    teacher_model,
    dataloader,
    forward_process,
    device,
    cfg,
    vl_runs=3,
    logger=None
):
    """
    Evaluate teacher model performance for comparison
    """
    # Import here to avoid circular import with loss.py
    from loss import p_losses
    
    losses = []
    teacher_model.eval()
    
    for step, batch in enumerate(dataloader):

        batch_size = batch["image"].shape[0]
        batch_imgs = batch["image"].to(device)
        batch_msks = batch["mask"].to(device)
        
        _vl_losses = []
        for _ in range(vl_runs):
            t = torch.randint(
                1, forward_process.forward_schedule.timesteps, (batch_size,), device=device
            ).long()
            
            # Teacher evaluation using regular p_losses (not KD version)
            loss, losses_dict = p_losses(
                forward_process,
                teacher_model,
                x_start=batch_msks,
                g=batch_imgs,
                t=t,
                cfg=cfg
            )
            _vl_losses.append((loss.item(), losses_dict))
        
        _vl_avg_loss = np.mean([l[0] for l in _vl_losses])
        _vl_avg_losses_dict = {}
        for k in _vl_losses[0][1].keys():        
            v = np.mean([l[1][k] for l in _vl_losses])
            _vl_avg_losses_dict[k]=v
    
        losses.append((_vl_avg_loss, _vl_avg_losses_dict, batch_size))

    return losses