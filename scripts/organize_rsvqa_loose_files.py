from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


def _place(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    if mode == "link":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        shutil.move(str(source), str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize loose RSVQA files shown in the Colab file browser.")
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--mode", choices=["link", "copy", "move"], default="link")
    parser.add_argument("--extract-images", action="store_true")
    args = parser.parse_args()
    source = Path(args.source_root).resolve()
    data_root = Path(args.data_root).resolve()

    for path in source.glob("USGS_*.json"):
        _place(path, data_root / "rsvqa_hr/annotations" / path.name, args.mode)
    for path in source.glob("LR_*.json"):
        _place(path, data_root / "rsvqa_lr/annotations" / path.name, args.mode)
    for path in source.glob("all_*.json"):
        # These are normally RSVQA auxiliary files; preserve them with HR annotations.
        _place(path, data_root / "rsvqa_hr/annotations" / path.name, args.mode)

    archives = list(source.glob("Images*.tar")) + list(source.glob("Images*.tar.gz"))
    for archive in archives:
        target = data_root / "rsvqa_hr/archives" / archive.name
        _place(archive, target, args.mode)
        if args.extract_images:
            images_dir = data_root / "rsvqa_hr/images"
            images_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(target) as handle:
                root_resolved = images_dir.resolve()
                for member in handle.getmembers():
                    destination = (images_dir / member.name).resolve()
                    if root_resolved not in destination.parents and destination != root_resolved:
                        raise RuntimeError(f"Unsafe archive member: {member.name}")
                handle.extractall(images_dir)
    print("RSVQA loose-file organization complete. Run scripts/validate_datasets.py next.")


if __name__ == "__main__":
    main()
