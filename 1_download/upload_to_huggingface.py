#!/usr/bin/env python3
"""
upload_to_huggingface.py — push benchmark data to the HuggingFace dataset.

Walks the LAYOUT table in data_locations.py, so the working-tree -> dataset
mapping is declared once and used by both readers and this uploader. Adding a
dataset section means editing LAYOUT, not this file.

  working tree                                    dataset prefix
  0_data/                                     ->  experimental/
  2_deterministic_build/outputs/gen_struct_*  ->  cdsm/<stage>/
  3_ml_prediction/outputs/<model>/            ->  predictions/<model>/
  4_scoring/results/                          ->  scores/

Scoring tables are also written as .parquet alongside the .csv, which is what
makes the HuggingFace data viewer render them in the browser.

Uploads are additive by default: nothing on the Hub is deleted unless you pass
--replace, which clears the target prefix first.

Auth (never pass a token on the command line — it lands in your shell history):
  hf auth login                  # preferred, stores in ~/.cache/huggingface
  export HF_TOKEN=...            # or the environment
(`huggingface-cli` was renamed to `hf` in huggingface_hub 1.x.)

The dataset card (dataset_card.md, beside this script) is published as the
repository's README.md. --all includes it; --card publishes it alone.

Usage:
  python upload_to_huggingface.py --dry-run              # show plan, upload nothing
  python upload_to_huggingface.py --all                  # every section + the card
  python upload_to_huggingface.py --section experimental --section scores
  python upload_to_huggingface.py --all --trajectories   # include the ~95 MB of .dcd
  python upload_to_huggingface.py --section scores --replace
  python upload_to_huggingface.py --card                 # card only
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_locations import LAYOUT, REPO_ID, REPO_TYPE, repo_root  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Never upload these, wherever they appear.
ALWAYS_IGNORE = [".DS_Store", "**/.DS_Store", "__pycache__/*", "**/__pycache__/*",
                 "*.pyc", "keep.txt"]

TRAJECTORY_GLOBS = ["**/trajectories/*"]

# Published as the dataset repo's README.md, which is what HuggingFace renders
# as the card. Kept in git beside the uploader so it is versioned with the code
# that produces the data it describes.
CARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_card.md")


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024


def measure(path: str, skip_trajectories: bool) -> tuple:
    """(file_count, total_bytes) for what would actually be uploaded."""
    count = total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        if skip_trajectories and "trajectories" in dirnames:
            dirnames.remove("trajectories")
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn == ".DS_Store" or fn.endswith(".pyc"):
                continue
            count += 1
            total += os.path.getsize(os.path.join(dirpath, fn))
    return count, total


def write_parquet_alongside_csv(results_dir: str) -> None:
    """Mirror each scores CSV as parquet so the HF data viewer can render it."""
    try:
        import pandas as pd
    except ImportError:
        log.warning("  pandas unavailable — skipping parquet conversion")
        return
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".csv"):
            continue
        csv_path = os.path.join(results_dir, fn)
        pq_path = csv_path[:-4] + ".parquet"
        pd.read_csv(csv_path).to_parquet(pq_path, index=False)
        log.info(f"  {fn} -> {os.path.basename(pq_path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", action="append", default=[],
                    help="dataset prefix to upload; repeatable. Default: none")
    ap.add_argument("--all", action="store_true",
                    help="upload every section, and the dataset card")
    ap.add_argument("--card", action="store_true",
                    help="publish dataset_card.md as the repo README.md")
    ap.add_argument("--trajectories", action="store_true",
                    help="include the ~95 MB of MD trajectories under cdsm/*/")
    ap.add_argument("--replace", action="store_true",
                    help="delete the target prefix on the Hub before uploading")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without contacting the Hub")
    args = ap.parse_args()

    send_card = args.card or args.all
    if args.all:
        sections = sorted(LAYOUT)
    elif args.section:
        unknown = [s for s in args.section if s not in LAYOUT]
        if unknown:
            sys.exit(f"unknown section(s): {unknown}\nexpected: {sorted(LAYOUT)}")
        sections = args.section
    elif args.card:
        sections = []
    else:
        sys.exit("Nothing to do: pass --all, --section <prefix>, or --card. "
                 f"Sections: {sorted(LAYOUT)}")

    if send_card and not os.path.isfile(CARD_PATH):
        sys.exit(f"Dataset card not found at {CARD_PATH}")

    skip_traj = not args.trajectories

    # ── plan ──────────────────────────────────────────────────────────────────
    plan, missing, total_files, total_bytes = [], [], 0, 0
    for prefix in sections:
        local = os.path.join(repo_root(), LAYOUT[prefix])
        if not os.path.isdir(local):
            missing.append((prefix, local))
            continue
        count, size = measure(local, skip_traj)
        plan.append((prefix, local, count, size))
        total_files += count
        total_bytes += size

    log.info(f"Target: {REPO_ID} ({REPO_TYPE})")
    log.info(f"Trajectories: {'included' if args.trajectories else 'skipped'}")
    log.info(f"Mode: {'REPLACE (deletes prefix first)' if args.replace else 'additive'}")
    log.info("Plan:")
    for prefix, local, count, size in plan:
        log.info(f"  {prefix:34s} <- {os.path.relpath(local, repo_root()):48s} "
                 f"{count:5d} files  {human(size):>9s}")
    for prefix, local in missing:
        log.warning(f"  {prefix:34s} <- MISSING {os.path.relpath(local, repo_root())}")
    log.info(f"  {'TOTAL':34s}    {total_files:5d} files  {human(total_bytes):>9s}")
    if send_card:
        log.info(f"  {'README.md (card)':34s} <- {os.path.relpath(CARD_PATH, repo_root())}")

    if args.dry_run:
        log.info("Dry run — nothing uploaded.")
        return

    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    token = os.environ.get("HF_TOKEN")  # falls back to the cached CLI login
    api = HfApi(token=token)
    try:
        who = api.whoami()["name"]
    except Exception:
        sys.exit("Not authenticated. Run `hf auth login`, or set HF_TOKEN.")
    log.info(f"Authenticated as {who}")

    ignore = list(ALWAYS_IGNORE) + (TRAJECTORY_GLOBS if skip_traj else [])

    for prefix, local, count, size in plan:
        if prefix == "scores":
            log.info("Deriving parquet from scores CSVs …")
            write_parquet_alongside_csv(local)
            count, size = measure(local, skip_traj)

        if args.replace:
            log.info(f"Clearing {prefix}/ on the Hub …")
            try:
                api.delete_folder(path_in_repo=prefix, repo_id=REPO_ID,
                                  repo_type=REPO_TYPE,
                                  commit_message=f"Clear {prefix}/ before re-upload")
            except (HfHubHTTPError, Exception) as exc:
                log.info(f"  nothing to clear ({type(exc).__name__})")

        log.info(f"Uploading {prefix}/  ({count} files, {human(size)}) …")
        api.upload_folder(
            folder_path=local,
            path_in_repo=prefix,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            ignore_patterns=ignore,
            commit_message=f"Upload {prefix} ({count} files)",
        )
        log.info(f"  {prefix}/ done.")

    if send_card:
        log.info("Publishing dataset card as README.md …")
        api.upload_file(
            path_or_fileobj=CARD_PATH,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Update dataset card",
        )
        log.info("  card done.")

    log.info(f"Done. https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
