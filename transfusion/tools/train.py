"""
TransFusion training script.

Features
--------
* Single-GPU and multi-GPU DDP training
* Automatic Mixed Precision (torch.cuda.amp)
* OneCycleLR scheduler
* Gradient clipping
* Checkpoint save / resume
* TensorBoard logging

Usage (single GPU)
------------------
python tools/train.py --config configs/nuscenes.yaml --data-root /data/nuscenes

Usage (8 GPUs)
--------------
torchrun --nproc_per_node=8 tools/train.py \
    --config configs/nuscenes.yaml --data-root /data/nuscenes
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transfusion import TransFusion, TransFusionLoss
from transfusion.data.nuscenes_dataset import NuScenesDataset, collate_fn
from transfusion.utils.common import get_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--data-root",  required=True)
    p.add_argument("--work-dir",   default="work_dirs/transfusion")
    p.add_argument("--resume",     default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--fusion-mode", type=str, default=None,
                   choices=["full", "cls_only", "dual_stream"],
                   help="Override fusion philosophy for the 3-way ablation")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


def init_dist(local_rank: int) -> Tuple[int, int]:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size()


def is_main(rank: int) -> bool:
    return rank == 0


def compute_bev_size(cfg: dict) -> Tuple[int, int]:
    """Compute bev_h, bev_w from pc_range and voxel_size, divided by out_size_factor."""
    pc = cfg["model"]["pc_range"]
    vs = cfg["model"]["voxel_size"]
    osf = cfg["model"]["out_size_factor"]
    bev_w = int(round((pc[3] - pc[0]) / vs[0] / osf))
    bev_h = int(round((pc[4] - pc[1]) / vs[1] / osf))
    return bev_h, bev_w


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    criterion: TransFusionLoss,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
    epoch:     int,
    bev_h:     int,
    bev_w:     int,
    cfg:       dict,
    logger:    logging.Logger,
    writer:    Optional[SummaryWriter],
    rank:      int,
    device:    torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_skipped = 0

    for step, batch in enumerate(loader):
        t0 = time.time()

        voxels      = batch["voxels"].to(device)
        num_points  = batch["num_points"].to(device)
        coords      = batch["coords"].to(device)
        camera_imgs = batch["camera_imgs"].to(device)
        lidar2img   = batch["lidar2img"].to(device)
        gt_labels   = [t.to(device) for t in batch["gt_labels"]]
        gt_boxes    = [t.to(device) for t in batch["gt_boxes"]]

        optimizer.zero_grad(set_to_none=True)

        amp_dtype = (torch.bfloat16
                     if cfg["training"].get("amp_dtype") == "bf16"
                     else torch.float16)
        with autocast(enabled=cfg["training"]["amp"], dtype=amp_dtype):
            outputs = model(
                camera_imgs=camera_imgs,
                voxels=voxels,
                num_points=num_points,
                coords=coords,
                lidar2img=lidar2img,
            )
            loss_dict = criterion(outputs, gt_labels, gt_boxes, bev_h, bev_w)
            loss = loss_dict["loss"]

        # Skip the optimizer step entirely on a non-finite loss. Without this
        # guard, scaler.step() would still apply an inf/nan gradient and
        # permanently corrupt every model weight it touches — once that
        # happens, EVERY subsequent step also produces nan/inf forever, since
        # nan propagates through every operation. This guard leaves the
        # weights exactly as they were before the bad batch, so training can
        # simply continue past a single unstable step instead of being
        # silently destroyed by it.
        if not torch.isfinite(loss):
            n_skipped += 1
            if is_main(rank):
                logger.warning(
                    "Non-finite loss at epoch %d step %d — skipping optimizer "
                    "step (weights left unchanged). Skipped %d step(s) so far "
                    "this epoch.", epoch, step, n_skipped,
                )
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])

        if not torch.isfinite(grad_norm):
            n_skipped += 1
            if is_main(rank):
                logger.warning(
                    "Non-finite gradient norm (%.4g) at epoch %d step %d — "
                    "skipping optimizer step. Skipped %d step(s) so far this "
                    "epoch.", float(grad_norm), epoch, step, n_skipped,
                )
            # unscale_() has already run this iteration; the scaler must be
            # advanced before the next step, or the following unscale_() call
            # raises. Skipping the optimizer step does NOT exempt us from
            # completing the scaler's update cycle.
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if is_main(rank) and step % 50 == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch [%d] Step [%d/%d]  loss=%.4f  "
                "(heat=%.3f cls=%.3f box=%.3f)  lr=%.2e  %.2fs/step",
                epoch, step, len(loader), loss.item(),
                loss_dict["loss_heatmap"].item(),
                loss_dict["loss_cls"].item(),
                loss_dict["loss_box"].item(),
                lr, elapsed,
            )
            if writer is not None:
                gs = epoch * len(loader) + step
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}", v.item(), gs)
                writer.add_scalar("train/lr", lr, gs)

    if is_main(rank) and n_skipped > 0:
        logger.warning("Epoch %d complete: %d step(s) skipped due to "
                       "non-finite loss/gradient.", epoch, n_skipped)

    return total_loss / max(len(loader) - n_skipped, 1)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    work_dir: Path, epoch: int, model: nn.Module,
    optimizer, scheduler, scaler: GradScaler, logger: logging.Logger,
) -> None:
    base = model.module if isinstance(model, DDP) else model
    ckpt = {
        "epoch":     epoch,
        "model":     base.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
    }
    path = work_dir / f"epoch_{epoch:03d}.pth"
    torch.save(ckpt, path)
    latest = work_dir / "latest.pth"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)
    logger.info("Saved checkpoint → %s", path)


def load_checkpoint(
    path: str, model: nn.Module,
    optimizer=None, scheduler=None, scaler: Optional[GradScaler] = None,
    logger: Optional[logging.Logger] = None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    base = model.module if isinstance(model, DDP) else model
    missing, unexpected = base.load_state_dict(ckpt["model"], strict=False)
    if logger:
        logger.info("Resumed %s  missing=%d unexpected=%d",
                    path, len(missing), len(unexpected))
    if optimizer  and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler  and "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    if scaler     and "scaler"    in ckpt: scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    use_dist = "RANK" in os.environ
    if use_dist:
        rank, world_size = init_dist(args.local_rank)
    else:
        rank, world_size = 0, 1

    device = torch.device(
        f"cuda:{args.local_rank}" if use_dist else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size
    if args.epochs:     cfg["training"]["epochs"]     = args.epochs
    if args.fusion_mode: cfg["model"]["fusion_mode"]  = args.fusion_mode

    # Keep each fusion-mode run in its own directory so the 3-way ablation
    # doesn't overwrite checkpoints/logs.
    work_dir = Path(args.work_dir) / cfg["model"].get("fusion_mode", "full")
    if is_main(rank):
        work_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(
        "transfusion",
        log_file=str(work_dir / "train.log") if is_main(rank) else None,
    )

    torch.manual_seed(args.seed + rank)

    bev_h, bev_w = compute_bev_size(cfg)
    if is_main(rank):
        logger.info("BEV grid: H=%d W=%d", bev_h, bev_w)

    # --- Dataset ---
    dc = cfg["data"]
    mc = cfg["model"]
    train_ds = NuScenesDataset(
        data_root=args.data_root,
        version=dc["version"],
        split="train",
        pc_range=dc["point_cloud_range"],
        voxel_size=dc["voxel_size"],
        max_points=dc["max_points_per_voxel"],
        max_voxels=dc["max_voxels_train"],
        img_size=tuple(dc["img_size"]),
        augment=True,
    )
    sampler = DistributedSampler(train_ds, shuffle=True) if use_dist else None
    loader  = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # --- Model ---
    model = TransFusion(
        bev_in_channels=mc["bev_in_channels"],
        num_cameras=mc["num_cameras"],
        num_classes=mc["num_classes"],
        num_queries=mc["num_queries"],
        d_model=mc["d_model"],
        nhead=mc["nhead"],
        num_lidar_decoder_layers=mc["num_lidar_decoder_layers"],
        num_fusion_decoder_layers=mc["num_fusion_decoder_layers"],
        dropout=mc["dropout"],
        bev_h=bev_h,
        bev_w=bev_w,
        pc_range=tuple(mc["pc_range"]),
        voxel_size=tuple(mc["voxel_size"]),
        out_size_factor=mc["out_size_factor"],
        point_feat_channels=mc.get("point_feat_channels", 4),
        fusion_mode=mc.get("fusion_mode", "full"),
        use_pillar_net=True,
    ).to(device)

    if cfg["training"].get("sync_bn") and use_dist:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if use_dist:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

    if is_main(rank):
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info("Model parameters: %.1fM", n_params)

    # --- Loss ---
    lc = cfg["loss"]
    criterion = TransFusionLoss(
        num_classes=mc["num_classes"],
        heatmap_weight=lc["heatmap_weight"],
        cls_weight=lc["cls_weight"],
        bbox_weight=lc["bbox_weight"],
        aux_weight=lc["aux_weight"],
        focal_alpha=lc["focal_alpha"],
        focal_gamma=lc["focal_gamma"],
        gaussian_overlap=lc["gaussian_overlap"],
        min_radius=lc["min_radius"],
        pc_range=tuple(mc["pc_range"]),
    ).to(device)

    # --- Optimiser + scheduler ---
    tc = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["optimizer"]["lr"],
        weight_decay=tc["optimizer"]["weight_decay"],
        betas=tc["optimizer"]["betas"],
    )
    total_steps = tc["epochs"] * len(loader)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=tc["scheduler"]["max_lr"],
        total_steps=total_steps,
        pct_start=tc["scheduler"]["pct_start"],
        div_factor=tc["scheduler"]["div_factor"],
        final_div_factor=tc["scheduler"]["final_div_factor"],
    )
    scaler = GradScaler(enabled=tc["amp"])

    # --- Resume ---
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, logger) + 1

    writer = SummaryWriter(str(work_dir / "tb")) if is_main(rank) else None

    # --- Training loop ---
    for epoch in range(start_epoch, tc["epochs"]):
        if use_dist and sampler is not None:
            sampler.set_epoch(epoch)

        avg_loss = train_one_epoch(
            model, criterion, loader, optimizer, scheduler, scaler,
            epoch, bev_h, bev_w, cfg, logger, writer, rank, device,
        )

        if is_main(rank):
            logger.info("Epoch [%d] avg_loss=%.4f", epoch, avg_loss)
            save_checkpoint(work_dir, epoch, model, optimizer, scheduler, scaler, logger)

    if writer:
        writer.close()
    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
