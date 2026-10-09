import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import EXPERIMENT_SCHEMA, torch_load, write_json_atomic
from .data import (
    PolicyDataset,
    collate_policy_samples,
    move_batch,
    validate_split_payload,
)
from .encoders import CosineSchedule, DIFFUSION_STEPS
from .evaluation import evaluate_model, masked_mse
from .models import build_policy


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def initialize_ema_policy(policy, config: dict, device: torch.device):
    ema_policy = build_policy(config).to(device).eval()
    ema_policy.load_state_dict(policy.state_dict())
    for parameter in ema_policy.parameters():
        parameter.requires_grad_(False)
    return ema_policy


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train_experiment(
    manifest_path: Path,
    split_path: Path,
    config: dict,
    output: Path,
    resume: bool = True,
) -> dict:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    validate_split_payload(split)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("split_sha256") != split["split_sha256"]:
        raise ValueError("Cache was not built from the requested frozen split")
    identity = {
        "config_digest": config["config_digest"],
        "split_sha256": split["split_sha256"],
        "cache_sha256": manifest["cache_sha256"],
    }
    latest_path = output / "checkpoint_last.pt"
    best_path = output / "checkpoint_best.pt"
    result_path = output / "result.json"
    if resume and result_path.is_file() and best_path.is_file():
        previous = json.loads(result_path.read_text())
        same_run = all(previous.get(key) == value for key, value in identity.items())
        if previous.get("complete") and same_run:
            best = torch_load(best_path)
            if all(best.get(key) == value for key, value in identity.items()):
                return previous
    seed = int(config.get("training_seed", 42))
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = PolicyDataset(manifest_path, set(split["train"]), config)
    validation_dataset = PolicyDataset(manifest_path, set(split["validation"]), config)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=True,
        generator=loader_generator,
        collate_fn=collate_policy_samples,
        num_workers=int(config.get("workers", 0)),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config.get("validation_batch_size", 1)),
        shuffle=False,
        collate_fn=collate_policy_samples,
        num_workers=int(config.get("workers", 0)),
    )
    policy = build_policy(config).to(device)
    ema_policy = initialize_ema_policy(policy, config, device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    if config.get("lr_scheduler", "constant") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config.get("maximum_epochs", 100))
        )
    elif config.get("lr_scheduler", "constant") == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        raise ValueError("lr_scheduler must be 'constant' or 'cosine'")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    schedule = (
        CosineSchedule(int(config.get("diffusion_steps", DIFFUSION_STEPS)), device)
        if config["head"] == "diffusion"
        else None
    )
    start_epoch = 0
    best_score = float("inf")
    best_epoch = 0
    checks_without_improvement = 0
    history = []

    if resume and latest_path.exists():
        saved = torch_load(latest_path, map_location="cpu")
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Cannot resume with different settings or prepared data")
        policy.load_state_dict(saved["model_raw"])
        ema_policy.load_state_dict(saved["model_ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        loader_generator.set_state(saved["loader_generator_state"])
        start_epoch = int(saved["epoch"])
        best_score = float(saved["best_score"])
        best_epoch = int(saved["best_epoch"])
        checks_without_improvement = int(saved["checks_without_improvement"])
        history = list(saved["history"])
        _restore_rng_state(saved["rng_state"])

    maximum_epochs = int(config.get("maximum_epochs", 100))
    minimum_epochs = int(config.get("minimum_epochs", 10))
    patience = int(config.get("early_stopping_patience", 10))
    min_delta = float(config.get("early_stopping_min_delta", 1e-4))
    ema_decay = float(config.get("ema_decay", 0.995))
    started = time.perf_counter()
    early_stopped = False

    for epoch in range(start_epoch, maximum_epochs):
        policy.train()
        training_error_sum = 0.0
        training_valid_steps = 0.0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                if config["head"] == "diffusion":
                    timestep = torch.randint(
                        0, schedule.steps, (len(batch["actions"]),), device=device
                    )
                    noisy = schedule.add_noise(
                        batch["actions"], timestep, torch.randn_like(batch["actions"])
                    )
                    prediction = policy(batch, noisy, timestep)
                else:
                    prediction = policy(batch)
                loss = masked_mse(prediction, batch["actions"], batch["mask"])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch + 1}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema_policy.parameters(), policy.parameters()
                ):
                    ema_parameter.lerp_(parameter, 1.0 - ema_decay)
                for ema_buffer, buffer in zip(ema_policy.buffers(), policy.buffers()):
                    ema_buffer.copy_(buffer)
            valid_steps = float(batch["mask"].sum().item())
            training_error_sum += float(loss.item()) * valid_steps
            training_valid_steps += valid_steps

        scheduler.step()

        metrics = evaluate_model(ema_policy, config, validation_loader, device)
        score = metrics["val_selection_score"]
        if not np.isfinite(score):
            raise FloatingPointError(
                f"Non-finite validation score at epoch {epoch + 1}: {score}"
            )
        improved = score < best_score - min_delta
        if improved:
            best_score = score
            best_epoch = epoch + 1
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
        record = {
            "epoch": epoch + 1,
            "train_masked_normalized_mse": training_error_sum / max(training_valid_steps, 1.0),
            **metrics,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "checks_without_improvement": checks_without_improvement,
        }
        history.append(record)
        checkpoint = {
            "schema": EXPERIMENT_SCHEMA,
            "config": config,
            **identity,
            "manifest_path": str(Path(manifest_path).resolve()),
            "split_path": str(Path(split_path).resolve()),
            "evaluation_weights": "ema",
            "objective": (
                "diffusion_clean_action"
                if config["head"] == "diffusion"
                else "deterministic_action_chunk"
            ),
            "model_raw": policy.state_dict(),
            "model_ema": ema_policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "loader_generator_state": loader_generator.get_state(),
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "checks_without_improvement": checks_without_improvement,
            "history": history,
            "split": split,
            "rng_state": _rng_state(),
        }
        _atomic_torch_save(latest_path, checkpoint)
        if improved:
            _atomic_torch_save(best_path, checkpoint)
        write_json_atomic(output / "history.json", history)
        print(json.dumps(record), flush=True)
        if epoch + 1 >= minimum_epochs and checks_without_improvement >= patience:
            early_stopped = True
            break

    if not best_path.is_file():
        raise RuntimeError("Training ended without a finite best checkpoint")
    best_saved = torch_load(best_path, map_location="cpu")
    best_metrics = best_saved["history"][-1]
    result = {
        "experiment_name": config["experiment_name"],
        "config_digest": config["config_digest"],
        "checkpoint": str(best_path),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_metrics": best_metrics,
        "early_stopped": early_stopped,
        "early_stopping_counter": checks_without_improvement,
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "elapsed_minutes_this_run": (time.perf_counter() - started) / 60.0,
        "device": str(device),
        "evaluation_weights": "ema",
        "split_sha256": split["split_sha256"],
        "cache_sha256": manifest["cache_sha256"],
        "complete": True,
    }
    write_json_atomic(result_path, result)
    return result
