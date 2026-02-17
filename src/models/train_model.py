"""
Model Training
==============

End-to-end training script: patient-aware splitting, class-weighted loss,
mixed-precision training loop, and prediction export.
"""

import argparse
import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupShuffleSplit
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, _ensure_dirs, load_config
from src.data.dataset import RetinopathyDataset
from src.data.make_dataset import create_manifest, load_filtered_dataset
from src.data.preprocessing import ImagePreprocessor
from src.data.transforms import get_transforms
from src.models.model import DRModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_data(df, test_size=0.15, val_size=0.15, seed=42):
    """Patient-aware train/val/test split (no patient appears in multiple sets)."""
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df["patient_id"]))
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Adjust val_size relative to the remaining (1 - test_size) fraction
    relative_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["patient_id"]))

    return train_val_df.iloc[train_idx].copy(), train_val_df.iloc[val_idx].copy(), test_df


def compute_loss_weights(train_df, device):
    """Inverse-frequency class weights for CrossEntropyLoss."""
    class_counts = train_df["level"].value_counts().reindex(range(5), fill_value=1).sort_index().values
    weights = len(train_df) / (5 * class_counts)
    return torch.FloatTensor(weights).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Run a single training epoch and return (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        with autocast("cuda", enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({"loss": loss.item()})

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Evaluate on a data loader and return (loss, accuracy, quadratic kappa)."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            del images, labels, outputs, loss

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss, accuracy, kappa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run single batch smoke test")
    parser.add_argument(
        "--cached", action="store_true", help="Use pre-cached 512x512 images (skips heavy preprocessing)"
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers from config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size from config")
    parser.add_argument("--model", type=str, default=None, help="Override model architecture (e.g., efficientnet_b3)")
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to model weights to load (e.g., best_model.pth)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to full checkpoint to resume training (optimizer, epoch, etc.)"
    )
    parser.add_argument("--evaluate", action="store_true", help="Skip training and run inference on validation set")
    parser.add_argument(
        "--min-quality", type=float, default=0.0, help="Minimum vessel density score to include image (0.0=all)"
    )
    args = parser.parse_args()

    config = load_config()
    _ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        scaler = GradScaler("cuda")
        logger.info("Enabled CUDNN Benchmark & Mixed Precision Training")
    else:
        scaler = None

    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.model is not None:
        config["model"]["name"] = args.model

    if args.cached:
        manifest_path = PROCESSED_DATA_DIR / "manifest_cached.csv"
        if not manifest_path.exists():
            raise FileNotFoundError("Cached manifest not found. Run 'python -m src.data.cache_images' first.")
        logger.info("Using Cached Manifest.")
    else:
        manifest_path = PROCESSED_DATA_DIR / "manifest.csv"
        if not manifest_path.exists():
            logger.info("Creating manifest...")
            create_manifest(DATA_DIR, manifest_path)

    logger.info(f"Loading data with min_quality={args.min_quality}")
    df = load_filtered_dataset(manifest_path, min_quality_score=args.min_quality)
    logger.info(f"Loaded DataFrame shape: {df.shape}")

    if args.dry_run:
        logger.warning("DRY RUN: Subsampling to ~15 patients, 1 epoch")
        subset_patients = df["patient_id"].unique()[:15]
        df = df[df["patient_id"].isin(subset_patients)]
        config["train"]["epochs"] = 1

    train_df, val_df, test_df = split_data(
        df, test_size=config["train"]["test_split"], val_size=config["train"]["val_split"]
    )

    logger.info(f"Split Sizes | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    preprocessor = None if args.cached else ImagePreprocessor(target_size=tuple(config["data"]["image_size"]))
    train_transforms = get_transforms("train", tuple(config["data"]["image_size"]))
    val_transforms = get_transforms("val", tuple(config["data"]["image_size"]))

    train_dataset = RetinopathyDataset(train_df, preprocessor, train_transforms)
    val_dataset = RetinopathyDataset(val_df, preprocessor, val_transforms)

    loss_weights = compute_loss_weights(train_df, device)
    logger.info(f"Class Weights: {loss_weights.cpu().numpy()}")

    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    logger.info(f"Initializing Model: {config['model']['name']}")
    model = DRModel(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    num_epochs = config["train"]["epochs"]
    best_loss = float("inf")
    start_epoch = 0

    if args.resume:
        if Path(args.resume).exists():
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scaler and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            logger.info(f"Resumed at Epoch {start_epoch + 1}")
        else:
            logger.warning(f"Checkpoint {args.resume} not found. Starting from scratch.")

    elif args.weights:
        if Path(args.weights).exists():
            logger.info(f"Loading weights from: {args.weights}")
            model.load_state_dict(torch.load(args.weights, weights_only=True))
        else:
            logger.warning(f"Weights file {args.weights} not found. Starting with random weights.")

    if not args.evaluate:
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_kappa": [],
        }

        for epoch in range(start_epoch, num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
            logger.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

            val_loss, val_acc, val_kappa = validate(model, val_loader, criterion, device)
            logger.info(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | Kappa: {val_kappa:.4f}")

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_kappa"].append(val_kappa)

            scheduler.step(val_loss)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_loss": best_loss,
                "config": config,
            }
            torch.save(checkpoint, MODELS_DIR / "last_checkpoint.pth")

            if val_loss < best_loss:
                best_loss = val_loss
                save_path = MODELS_DIR / f"{config['model']['name']}_best.pth"
                torch.save(model.state_dict(), save_path)
                torch.save(checkpoint, MODELS_DIR / f"{config['model']['name']}_best_checkpoint.pth")
                logger.info(f"--> Best Model Saved (Loss: {best_loss:.4f})")

        logger.info("Saving training history...")
        joblib.dump(history, REPORTS_DIR / f"history_{config['model']['name']}.joblib")

        logger.info("Loading best model for inference...")
        best_model_path = MODELS_DIR / f"{config['model']['name']}_best.pth"
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

    else:
        logger.info("Evaluate-only mode: skipping training loop.")
        if not args.weights and not args.resume:
            best_model_path = MODELS_DIR / f"{config['model']['name']}_best.pth"
            if best_model_path.exists():
                logger.info(f"Loading best weights from: {best_model_path}")
                model.load_state_dict(torch.load(best_model_path, weights_only=True))
            else:
                logger.warning(f"No weights found at {best_model_path}. Using random weights.")

    # Generate predictions on the validation set
    logger.info("Generating predictions on validation set...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    val_image_names = val_df["image"].values

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final Inference"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    results_df = pd.DataFrame(
        {
            "image_name": val_image_names[: len(all_labels)],
            "true": all_labels,
            "pred": all_preds,
        }
    )
    prob_arr = np.array(all_probs)
    for i in range(prob_arr.shape[1]):
        results_df[f"prob_{i}"] = prob_arr[:, i]

    results_path = REPORTS_DIR / f"predictions_{config['model']['name']}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Predictions saved to {results_path}")


if __name__ == "__main__":
    main()
