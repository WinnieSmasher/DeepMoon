from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


CHUNK_SIZE = 1024 * 1024

ZENODO_FILES = {
    "train": {
        "filename": "train_images.hdf5",
        "url": "https://zenodo.org/records/1133969/files/train_images.hdf5?download=1",
        "md5": "500f4f86a2d12c4fa134c4d225f957dd",
        "size_label": "10.0 GB",
    },
    "dev": {
        "filename": "dev_images.hdf5",
        "url": "https://zenodo.org/records/1133969/files/dev_images.hdf5?download=1",
        "md5": "34b9846f3138de1f6f17f8aa6f82a34d",
        "size_label": "2.53 GB",
    },
    "test": {
        "filename": "test_images.hdf5",
        "url": "https://zenodo.org/records/1133969/files/test_images.hdf5?download=1",
        "md5": "264be89fd66f39f300af2572f26eb7d5",
        "size_label": "2.53 GB",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 DeepMoon 真实训练数据")
    parser.add_argument(
        "--output-dir",
        default="data/external/zenodo",
        help="下载目录，默认写入 data/external/zenodo",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "dev", "test", "all"],
        default=["all"],
        help="需要下载的数据切分，默认全部下载",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标文件已存在，则覆盖下载",
    )
    parser.add_argument(
        "--skip-md5-check",
        action="store_true",
        help="下载完成后跳过 MD5 校验",
    )
    return parser.parse_args()


def expand_splits(raw_splits: list[str]) -> list[str]:
    if "all" in raw_splits:
        return ["train", "dev", "test"]
    return raw_splits


def compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_existing_file_valid(path: Path, expected_md5: str, skip_md5_check: bool) -> bool:
    if not path.exists():
        return False
    if skip_md5_check:
        print(f"已存在，跳过校验: {path}")
        return True
    actual_md5 = compute_md5(path)
    if actual_md5 == expected_md5:
        print(f"已存在且校验通过: {path}")
        return True
    print(f"已存在但校验失败，将重新下载: {path}")
    path.unlink()
    return False


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "DeepMoon Downloader"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        downloaded = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                percent = downloaded / total_bytes * 100.0
                print(f"\r下载中 {destination.name}: {percent:6.2f}%", end="", flush=True)
        if total_bytes:
            print()


def verify_md5(path: Path, expected_md5: str) -> None:
    actual_md5 = compute_md5(path)
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"MD5 校验失败: {path.name}，期望 {expected_md5}，实际 {actual_md5}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    splits = expand_splits(args.splits)

    print("准备下载以下数据切分：")
    for split in splits:
        meta = ZENODO_FILES[split]
        print(f"- {split}: {meta['filename']} ({meta['size_label']})")

    for split in splits:
        meta = ZENODO_FILES[split]
        destination = output_dir / meta["filename"]
        if not args.overwrite and ensure_existing_file_valid(
            destination,
            expected_md5=meta["md5"],
            skip_md5_check=args.skip_md5_check,
        ):
            continue

        print(f"开始下载 {meta['filename']}")
        try:
            download_file(meta["url"], destination)
            if not args.skip_md5_check:
                verify_md5(destination, meta["md5"])
        except Exception:
            if destination.exists():
                destination.unlink()
            raise
        print(f"下载完成: {destination}")

    print("全部任务完成。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n下载已取消。", file=sys.stderr)
        raise SystemExit(130)
