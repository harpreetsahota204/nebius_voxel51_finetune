"""Fine-tunes an Ultralytics YOLO segmentation model inside a Nebius Serverless job.

This script is the payload of the Nebius job launched from
``yolo_seg_finetune_nebius.ipynb``. The notebook injects it into the container
(``--inject-file``) and runs it with the stock ``ultralytics/ultralytics`` image,
so no custom image has to be built or pushed.

Data flow (everything under ``--bucket-root`` is a mounted Object Storage bucket):

1. Extract the YOLO-format dataset archive from the bucket onto local disk.
   Bucket mounts are FUSE-backed and read-mostly, so training never touches them.
2. Point ``dataset.yaml`` at the local copy.
3. Train with Ultralytics, writing run artifacts to local disk.
4. Copy the useful artifacts (weights, metrics, plots) back to the bucket.

Entry point: ``main()`` (``python train_seg_job.py --help`` lists the options).
The same script runs locally for a smoke test by pointing ``--bucket-root`` at
any directory that contains the dataset archive.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import yaml
from ultralytics import YOLO

# Small artifacts worth keeping besides the weights: metrics (csv), curves and
# plots (png), the resolved training args (yaml), and preview images (jpg).
ARTIFACT_PATTERNS: tuple[str, ...] = ("*.csv", "*.png", "*.yaml", "*.jpg")


def parse_args() -> argparse.Namespace:
    """Parses the command-line options that control data paths and training.

    Returns:
        Namespace with the dataset location, model, and training hyperparameters.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket-root", default="/bucket", help="Mounted bucket path")
    parser.add_argument(
        "--dataset-archive",
        default="datasets/plantseg-yolo.tar.gz",
        help="Dataset .tar.gz, relative to --bucket-root",
    )
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Where artifacts are copied, relative to --bucket-root",
    )
    parser.add_argument("--run-name", default="plantseg-seg", help="Run folder name")
    parser.add_argument("--model", default="yolo26s-seg.pt", help="Pretrained weights")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scratch", default="/tmp/job", help="Local scratch directory")
    return parser.parse_args()


def extract_dataset(archive: Path, scratch: Path) -> Path:
    """Extracts the dataset archive onto local disk and localizes its YAML.

    Args:
        archive: path to the ``.tar.gz`` produced by the notebook's export step.
        scratch: local directory to extract into.

    Returns:
        Path to a ``dataset.yaml`` whose ``path`` points at the extracted folder.
    """
    # Local extraction avoids thousands of small reads over the FUSE mount
    scratch.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(scratch)

    # The archive holds exactly one top-level folder: the FiftyOne export dir
    dataset_dir = next(p for p in scratch.iterdir() if p.is_dir())
    yaml_path = dataset_dir / "dataset.yaml"

    # FiftyOne wrote an absolute path from the machine that exported the data;
    # rewrite it so it resolves inside this container
    config = yaml.safe_load(yaml_path.read_text())
    config["path"] = str(dataset_dir)
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return yaml_path


def copy_artifacts(run_dir: Path, destination: Path) -> None:
    """Copies run artifacts to the bucket using data-only file copies.

    Args:
        run_dir: local Ultralytics run directory (contains ``weights/``).
        destination: folder inside the mounted bucket to write into.
    """
    (destination / "weights").mkdir(parents=True, exist_ok=True)

    # copyfile (not copy2/copytree): the FUSE mount rejects the chmod/utime
    # calls those make, which surfaces as "Operation not permitted"
    for weights in (run_dir / "weights").glob("*.pt"):
        shutil.copyfile(weights, destination / "weights" / weights.name)
    for pattern in ARTIFACT_PATTERNS:
        for artifact in run_dir.glob(pattern):
            shutil.copyfile(artifact, destination / artifact.name)


def main() -> None:
    """Runs extract -> train -> copy-back and prints where the results landed."""
    args = parse_args()
    bucket_root = Path(args.bucket_root)
    scratch = Path(args.scratch)

    # Stage 1: get the data onto fast local disk
    data_yaml = extract_dataset(bucket_root / args.dataset_archive, scratch / "data")

    # Stage 2: fine-tune; weights are downloaded by Ultralytics on first use
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(scratch / "runs"),
        name=args.run_name,
        exist_ok=True,
    )

    # Stage 3: persist results before the container filesystem disappears
    run_dir = scratch / "runs" / args.run_name
    destination = bucket_root / args.output_dir / args.run_name
    copy_artifacts(run_dir, destination)
    print(f"Artifacts copied to {destination}")


if __name__ == "__main__":
    main()
