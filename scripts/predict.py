from __future__ import annotations

import argparse

from deepmoon.config import load_config, parse_cli_overrides
from deepmoon.postprocessing.crater_extraction import extract_unique_craters


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepMoon 推理与陨石坑提取")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--prediction-path", default=None)
    parser.add_argument("--result-path", default=None)
    parser.add_argument("--llt2", type=float, default=None)
    parser.add_argument("--rt2", type=float, default=None)
    parser.add_argument("--set", action="append", default=[], help="覆盖配置，格式: a.b=value")
    args = parser.parse_args()

    overrides = parse_cli_overrides(args.set)
    if args.llt2 is not None:
        overrides["postprocessing.llt2"] = args.llt2
    if args.rt2 is not None:
        overrides["postprocessing.rt2"] = args.rt2

    config = load_config(args.config, overrides=overrides)
    model_path = args.model_path or config.prediction.model_path
    data_path = args.data_path or config.data.test_path
    prediction_path = args.prediction_path or config.prediction.prediction_path
    result_path = args.result_path or config.prediction.result_path

    unique = extract_unique_craters(
        config=config,
        model_path=model_path,
        data_path=data_path,
        prediction_path=prediction_path,
        result_path=result_path,
    )
    print(f"提取完成，共 {len(unique)} 个唯一陨石坑")


if __name__ == "__main__":
    main()
