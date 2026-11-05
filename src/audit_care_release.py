#!/usr/bin/env python
"""记录 CARE 发布版本的病例/切片数量与排序文件名清单哈希。"""

import argparse
import hashlib
import json
from pathlib import Path

from dataset import parse_case_id


def audit(data_root, output, downloaded=None):
    root = Path(data_root)
    manifest = {
        "release": "DataV6",
        "downloaded": downloaded,
        "hash_scope": "SHA-256 of sorted split-relative .npz paths; file contents are not hashed",
        "splits": {},
    }
    for split in ("train", "test"):
        split_root = root / split
        files = sorted(split_root.glob("*.npz"))
        if not files:
            files = sorted((split_root / f"{split}_npz").glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no .npz files under {split_root}")
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.relative_to(split_root).as_posix().encode("utf-8"))
            digest.update(b"\n")
        manifest["splits"][split] = {
            "n_cases": len({parse_case_id(path.name) for path in files}),
            "n_slices": len(files),
            "file_list_sha256": digest.hexdigest(),
        }
    manifest["total_cases"] = sum(x["n_cases"] for x in manifest["splits"].values())
    manifest["total_slices"] = sum(x["n_slices"] for x in manifest["splits"].values())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/CARE")
    parser.add_argument("--out", default="results/v2/care_release_manifest.json")
    parser.add_argument("--downloaded", default="2026-09-15")
    args = parser.parse_args()
    audit(args.data_root, args.out, args.downloaded)


if __name__ == "__main__":
    main()
