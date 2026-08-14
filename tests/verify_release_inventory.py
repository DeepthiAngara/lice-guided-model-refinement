#!/usr/bin/env python3
from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "results" / "verification" / "release_file_inventory.csv"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

inventory = pd.read_csv(INVENTORY)
failures = []
for row in inventory.itertuples(index=False):
    path = ROOT / row.Relative_Path
    if not path.exists():
        failures.append((row.Relative_Path, "missing"))
    elif path.stat().st_size != row.Size_Bytes:
        failures.append((row.Relative_Path, "size mismatch"))
    elif sha256(path) != row.SHA256:
        failures.append((row.Relative_Path, "checksum mismatch"))
print(f"Inventory files checked: {len(inventory)}")
print(f"Inventory failures: {len(failures)}")
if failures:
    print(failures[:20])
    raise SystemExit(1)

