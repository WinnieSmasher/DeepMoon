from __future__ import annotations

import argparse
from pathlib import Path

from deepmoon.config import load_config, parse_cli_overrides
from deepmoon.data.dataset import get_dataloaders
from deepmoon.models import build_model
from deepmoon.training.trainer import train_model
from deepmoon.utils.seed import (
    get_experiment_seed,
    is_deterministic_experiment,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 DeepMoon 模型")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["attention_unet", "trans_unet"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="启用确定性实验模式，便于严格 apples-to-apples 对比",
    )
    parser.add_argument("--set", action="append", default=[], help="覆盖配置，格式: a.b=value")
    args = parser.parse_args()

    overrides = parse_cli_overrides(args.set)
    if args.model is not None:
        overrides["model.arch"] = args.model
    if args.epochs is not None:
        overrides["training.epochs"] = args.epochs
    if args.device is not None:
        overrides["training.device"] = args.device
    if args.seed is not None:
        overrides["experiment.seed"] = args.seed
    if args.deterministic:
        overrides["experiment.deterministic"] = True

    config = load_config(args.config, overrides=overrides)

    experiment_seed = get_experiment_seed(config)
    if experiment_seed is not None:
        seed_everything(
            experiment_seed,
            deterministic=is_deterministic_experiment(config),
        )

    loaders = get_dataloaders(config)
    model = build_model(config)
    history = train_model(model, loaders, config)
    print(f"训练完成，共 {len(history)} 个 epoch")
    checkpoint_path = Path(config.training.checkpoint_dir) / str(config.training.checkpoint_name)
    print(f"最佳检查点: {checkpoint_path}")


if __name__ == "__main__":
    main()
