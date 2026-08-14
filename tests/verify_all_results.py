#!/usr/bin/env python3
"""Master verification entry point for release v1.3.0."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
OUT = ROOT / "results" / "verification"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--write-results",
    action="store_true",
    help="write frozen verification logs and summaries before release inventory generation",
)
args = parser.parse_args()
programs = [
    ("core_results", TESTS / "verify_original_results_stable_names.py"),
    ("extended_analyses", TESTS / "verify_extended_analyses.py"),
    ("method_audit", TESTS / "verify_method_audit.py"),
    ("recent_comparators", TESTS / "verify_recent_comparators.py"),
]
rows = []
component_check_totals = {}
for scope, program in programs:
    completed = subprocess.run([sys.executable, str(program)], cwd=ROOT, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if args.write_results:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{scope}_verification.log").write_text(
            completed.stdout, encoding="utf-8"
        )
    component_summary = None
    for line in completed.stdout.splitlines():
        if line.startswith("VERIFICATION_SUMMARY "):
            component_summary = json.loads(line.removeprefix("VERIFICATION_SUMMARY "))
    valid_summary = component_summary is not None
    component_failed = component_summary.get("failed", 1) if valid_summary else 1
    status = "PASS" if completed.returncode == 0 and component_failed == 0 else "FAIL"
    check_count = component_summary.get("checks") if valid_summary else None
    component_check_totals[scope] = check_count
    rows.append({"Scope": scope, "Program": str(program.relative_to(ROOT)),
                 "Exit_Code": completed.returncode, "Checks": check_count,
                 "Status": status})

summary = pd.DataFrame(rows)
payload = {
    "programs": len(summary),
    "programs_passed": int((summary.Status == "PASS").sum()),
    "programs_failed": int((summary.Status == "FAIL").sum()),
    "component_check_totals": {
        **component_check_totals,
        "total": sum(value for value in component_check_totals.values() if value is not None),
    },
}
if args.write_results:
    summary.to_csv(OUT / "master_verification_program_status.csv", index=False)
    (OUT / "master_verification_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
print(summary.to_string(index=False))
print(json.dumps(payload, indent=2))
if payload["programs_failed"]:
    raise SystemExit(1)
