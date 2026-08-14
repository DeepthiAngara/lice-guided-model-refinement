#!/usr/bin/env python3
"""Create the v1.3.0 inventory, checksums, and release manifest."""
from pathlib import Path
import hashlib, json, platform, subprocess, sys
from datetime import date
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "verification"
OUT.mkdir(parents=True, exist_ok=True)
verification = subprocess.run(
    [sys.executable, str(ROOT / "tests" / "verify_all_results.py"), "--write-results"],
    cwd=ROOT,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(verification.stdout)
if verification.returncode:
    raise SystemExit("Numerical verification failed; release was not frozen.")
excluded = {
    "results/verification/release_file_inventory.csv",
    "results/verification/release_manifest.json",
}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = []
for path in sorted(p for p in ROOT.rglob("*") if p.is_file()):
    relative = str(path.relative_to(ROOT))
    if relative in excluded:
        continue
    files.append({"Relative_Path": relative, "Size_Bytes": path.stat().st_size,
                  "SHA256": sha256(path)})
inventory = pd.DataFrame(files)
inventory.to_csv(OUT / "release_file_inventory.csv", index=False)

summary = json.loads((OUT / "master_verification_summary.json").read_text())
manifest = {
    "release_version": "v1.3.0",
    "release_date": str(date.today()),
    "files_in_inventory": len(inventory),
    "inventory_total_bytes": int(inventory.Size_Bytes.sum()),
    "verification": summary,
    "workflow_based_artifact_names": True,
    "python": sys.version,
    "platform": platform.platform(),
    "git_commit": None,
}
(OUT / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
