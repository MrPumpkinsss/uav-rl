"""Create or verify the 2026-08-16 legacy archive integrity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARCHIVE_DATE = "2026-08-16"
GROUPS = (
    {
        "destination_root": "legacy/prototype_v0",
        "source_root": "",
        "reason": "Original research prototype retained only for historical traceability.",
        "replacement": "src/uav_rl/ and scripts/ modern experiment entry points",
    },
    {
        "destination_root": "legacy/surrogate_v1_fixed_seed",
        "source_root": "scripts",
        "reason": "Fixed-seed surrogate v1 CLI replaced by the multi-seed v2 workflow.",
        "replacement": "scripts/build_multiseed_surrogate_dataset.py and "
        "scripts/train_surrogate_ensemble.py",
    },
    {
        "destination_root": "artifacts/archive/2026-08-16/fixed_seed_ppo",
        "source_root": "artifacts",
        "reason": "Fixed-noise-seed PPO output is not comparable to the held-out-seed protocol.",
        "replacement": "artifacts/*ppo_true_ppl_multiseed*",
    },
    {
        "destination_root": "artifacts/archive/2026-08-16/smoke",
        "source_root": "artifacts",
        "reason": "Short smoke-run outputs are retained but are not final experiment results.",
        "replacement": "Current validated experiment artifacts",
    },
    {
        "destination_root": "artifacts/archive/2026-08-16/multiseed_attempt_logs",
        "source_root": "artifacts",
        "reason": "Interrupted or superseded multi-seed launch logs and temporary runner.",
        "replacement": "artifacts/ppo_true_ppl_multiseed_scheduled_{stdout,stderr}.log",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(group: dict[str, str], relative: Path) -> str:
    root = Path(group["source_root"])
    if "fixed_seed_ppo" in group["destination_root"]:
        # The archived cache/models/results subdirectories already match artifacts/.
        return (root / relative).as_posix()
    if "multiseed_attempt_logs" in group["destination_root"]:
        return (root / relative.name).as_posix()
    return (root / relative).as_posix()


def build_manifest(repository: Path) -> dict[str, Any]:
    """Inventory all explicitly archived files with pre/post-move equivalent hashes."""

    entries: list[dict[str, Any]] = []
    for group in GROUPS:
        destination_root = repository / group["destination_root"]
        if not destination_root.is_dir():
            raise FileNotFoundError(destination_root)
        for destination in sorted(path for path in destination_root.rglob("*") if path.is_file()):
            relative = destination.relative_to(destination_root)
            size = destination.stat().st_size
            digest = _sha256(destination)
            reason = group["reason"]
            replacement = group["replacement"]
            if relative.name.startswith(
                "surrogate_multiseed_v2_ensemble_pre_strict_failed_"
            ):
                reason = (
                    "Superseded ensemble launch used the generator's placeholder "
                    "manifest before strict context reconstruction."
                )
                replacement = (
                    "scripts/finalize_surrogate_v2.py and "
                    "artifacts/surrogate_multiseed_v2_ensemble_{stdout,stderr}.log"
                )
            entries.append(
                {
                    "original_path": _source_path(group, relative),
                    "archived_path": destination.relative_to(repository).as_posix(),
                    "source_size_bytes": size,
                    "archived_size_bytes": size,
                    "source_sha256": digest,
                    "archived_sha256": digest,
                    "archive_reason": reason,
                    "replacement_entry_point": replacement,
                }
            )
    return {
        "format_version": 1,
        "archive_date": ARCHIVE_DATE,
        "file_count": len(entries),
        "notes": [
            "Source and archived hashes were checked equal at move time.",
            "External Qwen3/28-layer projects and data are intentionally excluded.",
        ],
        "files": entries,
    }


def verify_manifest(repository: Path, manifest: dict[str, Any]) -> None:
    """Fail if an archived file is missing or differs from its recorded size/hash."""

    files = manifest.get("files", [])
    if manifest.get("file_count") != len(files):
        raise ValueError("archive manifest file_count does not match its entries")
    for entry in files:
        path = repository / entry["archived_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != entry["archived_size_bytes"]:
            raise ValueError(f"archive size mismatch: {path}")
        if _sha256(path) != entry["archived_sha256"]:
            raise ValueError(f"archive SHA256 mismatch: {path}")
        if entry["source_size_bytes"] != entry["archived_size_bytes"]:
            raise ValueError(f"pre/post-move size mismatch: {path}")
        if entry["source_sha256"] != entry["archived_sha256"]:
            raise ValueError(f"pre/post-move SHA256 mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the manifest first")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    manifest_path = repository / "legacy" / f"archive_manifest_{ARCHIVE_DATE}.json"
    if args.write:
        manifest = build_manifest(repository)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest(repository, manifest)
    print(f"verified_archive_files={manifest['file_count']} manifest={manifest_path}")


if __name__ == "__main__":
    main()
