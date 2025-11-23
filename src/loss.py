from utils.header import torch, nn, F
from common.logging import get_logger
from utils.helper_funcs import calc_boundary_att


"""Losses for KD training: L2, Boundary, and EFKD only."""


# Global variable for boundary loss instance
calc_boundary = None


class BoundaryLoss(torch.nn.Module):
    """Boundary Loss for edge-aware diffusion training"""
    def __init__(self, parameters={}):
        super().__init__()
        self.logger = get_logger()
        
        self.gamma = parameters.get("gamma", 1.5)
        root = parameters.get("root", "l2")
        if root == "l2":
            self.calc_root_loss = lambda p, t: torch.abs(p-t)**2
        elif root == "l1":
            self.calc_root_loss = lambda p, t: torch.abs(p-t)
        else:
            self.logger.exception("Not implemented!")

    def forward(self, x, t, T, predicted_noise, noise):
        boundary_att = calc_boundary_att(x, t, T=T, gamma=self.gamma)
        root_loss = self.calc_root_loss(predicted_noise, noise)
        return (boundary_att * root_loss).mean()


def edge_focused_kd_loss(student_pred, teacher_pred, x_start, t, T, efkd_config):
    """
    Edge-Focused Knowledge Distillation (EFKD) Loss
    Separates edge and body regions with different weights for edge preservation
    """
    edge_weight = efkd_config.get("edge_weight", 3.0)
    body_weight = efkd_config.get("body_weight", 1.0)
    
    # Calculate boundary attention map
    boundary_att = calc_boundary_att(x_start, t, T, gamma=1.5)
    
    # Create edge and body masks
    edge_threshold = efkd_config.get("edge_threshold", 0.7)
    edge_mask = (boundary_att > edge_threshold).float()
    body_mask = (boundary_att <= edge_threshold).float()
    
    # No temperature scaling
    student_scaled = student_pred
    teacher_scaled = teacher_pred.detach()
    
    # Compute separate losses for edge and body regions
    edge_loss = torch.nn.functional.mse_loss(
        student_scaled * edge_mask, 
        teacher_scaled * edge_mask,
        reduction='none'
    )
    
    body_loss = torch.nn.functional.mse_loss(
        student_scaled * body_mask,
        teacher_scaled * body_mask, 
        reduction='none'
    )
    
    # Weight and combine losses
    weighted_edge_loss = edge_weight * edge_loss
    weighted_body_loss = body_weight * body_loss
    
    # Normalize by mask areas to avoid bias
    edge_area = edge_mask.sum(dim=(1,2,3), keepdim=True) + 1e-8
    body_area = body_mask.sum(dim=(1,2,3), keepdim=True) + 1e-8
    
    normalized_edge_loss = (weighted_edge_loss.sum(dim=(1,2,3), keepdim=True) / edge_area).mean()
    normalized_body_loss = (weighted_body_loss.sum(dim=(1,2,3), keepdim=True) / body_area).mean()
    
    total_loss = normalized_edge_loss + normalized_body_loss
    return total_loss


def p_losses_kd(
    forward_process,
    student_model,
    teacher_model,
    x_start,
    g,
    t,
    cfg,
    noise=None,
):
    """
    KD objective: L2, Boundary loss, and Boundary-Privileged KD only.
    """
    logger = get_logger()
    
    T = cfg["diffusion"]["schedule"]["timesteps"]
    cfg_loss = cfg['training']['loss']

    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = forward_process.q_sample(x_start=x_start, t=t, noise=noise)
    
    # Import models
    from models import LiteDermoSegDiff, DermoSegDiff
    
    # Student forward pass
    if isinstance(student_model, (LiteDermoSegDiff, DermoSegDiff)):
        student_predicted_noise = student_model(x_noisy, g, t)
    else:
        logger.warning(f'Student model type {type(student_model)} may not be supported, attempting forward pass...')
        student_predicted_noise = student_model(x_noisy, g, t)
        
    # Teacher forward pass (no gradients)
    with torch.no_grad():
        if isinstance(teacher_model, (DermoSegDiff, LiteDermoSegDiff)):
            teacher_predicted_noise = teacher_model(x_noisy, g, t)
        else:
            logger.warning(f'Teacher model type {type(teacher_model)} may not be supported, attempting forward pass...')
            teacher_predicted_noise = teacher_model(x_noisy, g, t)
    
    losses = dict()
    
    # 1. Base L2 loss (basic diffusion loss)
    if "l2" in cfg_loss.keys():
        losses["l2"] = torch.nn.functional.mse_loss(student_predicted_noise, noise)

    # 2. Edge-Focused KD (EFKD)
    if "edge_focused_kd" in cfg_loss.keys():
        efkd_config = cfg_loss["edge_focused_kd"].get("params", {})
        losses["edge_focused_kd"] = edge_focused_kd_loss(
            student_predicted_noise, teacher_predicted_noise, x_start, t, T, efkd_config
        )

    # 3. Boundary loss (original)
    if "boundary" in cfg_loss.keys():
        if not hasattr(p_losses_kd, 'calc_boundary'):
            parameters = cfg_loss["boundary"].get("params", {})
            p_losses_kd.calc_boundary = BoundaryLoss(parameters)
        losses["boundary"] = p_losses_kd.calc_boundary(x_start, t, T, student_predicted_noise, noise)

    # Combine losses with coefficients
    configured = [k for k in cfg_loss.keys() if k in losses]
    if len(configured) == 1:
        only_key = configured[0]
        loss = losses[only_key]
    else:
        loss = 0
        for l_name in cfg_loss.keys():
            if l_name in losses:
                coeff = cfg_loss.get(l_name, {}).get("coefficient", cfg_loss.get(l_name, {}).get("cofficient", 1)) if isinstance(cfg_loss.get(l_name, {}), dict) else 1
                loss += coeff * losses[l_name]
        losses['hybrid'] = loss

    # Convert to items for logging
    losses = dict((k, v.item()) for k, v in losses.items())
    return loss, losses


def p_losses(
    forward_process,
    denoise_model,
    x_start,
    g,
    t,
    cfg,
    noise=None,
):
    """
    Regular loss function for teacher evaluation (L2 and Boundary only)
    """
    global calc_boundary
    logger = get_logger()
    
    T = cfg["diffusion"]["schedule"]["timesteps"]
    cfg_loss = cfg['training']['loss']

    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = forward_process.q_sample(x_start=x_start, t=t, noise=noise)
    
    from models import DermoSegDiff, Baseline
    
    if isinstance(denoise_model, DermoSegDiff):
        predicted_noise = denoise_model(x_noisy, g, t)
    elif isinstance(denoise_model, Baseline):
        predicted_noise = denoise_model(x=x_noisy, time=t, x_self_cond=g)
    else:
        logger.exception('given <denoise_model> is unknown!')
    
    losses = dict()
    
    # Only L2 and Boundary for teacher evaluation
    if "l2" in cfg_loss.keys():
        losses["l2"] = F.mse_loss(predicted_noise, noise)
    if "boundary" in cfg_loss.keys():
        if not calc_boundary:
            parameters = cfg_loss["boundary"].get("params", {})
            calc_boundary = BoundaryLoss(parameters)
        losses["boundary"] = calc_boundary(x_start, t, T, predicted_noise, noise)

    # Combine losses with coefficients
    configured = [k for k in cfg_loss.keys() if k in losses]
    if len(configured) == 1:
        only_key = configured[0]
        loss = losses[only_key]
    else:
        loss = 0
        for l_name, l_d in cfg_loss.items():
            if l_name in losses:
                coeff = l_d.get("coefficient", l_d.get("cofficient", 1))
                loss += coeff * losses[l_name]
        losses['hybrid'] = loss

    losses = dict((k, v.item()) for k, v in losses.items())
    return loss, losses