#!/usr/bin/env python3
"""Download and verify the exact LoCoMo dataset used by this reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "source_manifest.json"
DEFAULT_OUTPUT = HERE / ".data" / "locomo10.json"


class DatasetError(RuntimeError):
    """Raised when the pinned dataset cannot be prepared safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spec(manifest_path: Path = MANIFEST_PATH) -> dict:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["locomo"]


def verify_dataset(path: Path, spec: dict | None = None) -> None:
    spec = spec or load_spec()
    if not path.is_file():
        raise DatasetError(f"LoCoMo dataset not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != int(spec["size"]):
        raise DatasetError(
            f"LoCoMo size mismatch: expected {spec['size']}, got {actual_size}: {path}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != spec["sha256"]:
        raise DatasetError(
            f"LoCoMo SHA256 mismatch: expected {spec['sha256']}, got {actual_hash}: {path}"
        )


def download_dataset(output: Path, *, force: bool = False, retries: int = 3) -> Path:
    spec = load_spec()
    output = output.expanduser().resolve()
    if output.exists() and not force:
        verify_dataset(output, spec)
        print(f"LoCoMo dataset already verified: {output}")
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="locomo10-", suffix=".json", dir=output.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        print(f"Downloading pinned LoCoMo dataset:\n  {spec['url']}")
        last_error: Exception | None = None
        for attempt in range(1, max(1, retries) + 1):
            request = urllib.request.Request(
                spec["url"], headers={"User-Agent": "Hermes-LoCoMo-Reproduction/1.0"}
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response, temp_path.open(
                    "wb"
                ) as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                last_error = None
                break
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt < max(1, retries):
                    delay = min(2 ** (attempt - 1), 8)
                    print(
                        f"Download attempt {attempt}/{retries} failed: {exc}; "
                        f"retrying in {delay}s",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
        if last_error is not None:
            raise last_error
        verify_dataset(temp_path, spec)
        temp_path.replace(output)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(f"PASS: downloaded and verified LoCoMo dataset: {output}")
    print(f"      bytes:  {spec['size']}")
    print(f"      sha256: {spec['sha256']}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Redownload an existing file")
    parser.add_argument("--retries", type=int, default=3, help="Download attempts (default: 3)")
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify the local file without downloading"
    )
    args = parser.parse_args()
    try:
        if args.verify_only:
            verify_dataset(args.output.expanduser().resolve())
            print(f"PASS: LoCoMo dataset verified: {args.output.expanduser().resolve()}")
        else:
            download_dataset(args.output, force=args.force, retries=max(1, args.retries))
    except (DatasetError, OSError, urllib.error.URLError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
