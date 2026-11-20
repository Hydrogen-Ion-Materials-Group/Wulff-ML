#!/usr/bin/env python3
"""
Clear cached pipeline results.

Usage:
    python clear_cache.py                  # wipe the entire cache
    python clear_cache.py train_xgb        # wipe only one task's cache
    python clear_cache.py --status         # show what's cached, don't delete anything
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wulff_ml import cache


def main():
    args = sys.argv[1:]
    if args and args[0] == "--status":
        cache.status()
        return
    if args:
        cache.clear(args[0])
    else:
        confirm = input("This will delete ALL cached results (models, splits, SHAP values, "
                         "etc.) and force everything to recompute on next run. Continue? [y/N] ")
        if confirm.strip().lower() == "y":
            cache.clear()
        else:
            print("Cancelled.")


if __name__ == "__main__":
    main()
