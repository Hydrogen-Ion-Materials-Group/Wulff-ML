"""
Command-line entry points installed with the package.

    wulff-predict   predict facet surface energies and Wulff shapes from
                    Materials Project IDs
    wulff-key       enter / show / delete the saved Materials Project API key
                    (implemented in wulff_ml.api_key)

Examples:
    wulff-predict                              # asks for the Materials Project ID(s)
    wulff-predict mp-1077900 mp-21119          # IDs given directly
    wulff-predict --file ids.txt               # IDs from a text file
    wulff-predict mp-149 --no-wulff            # surface energies only
    wulff-predict mp-149 --out my_results      # choose the output folder

Find the ID of a compound on https://next-gen.materialsproject.org/materials
(it looks like mp-149). The first run asks for your free Materials Project
API key and trains the model once; later runs start immediately.
"""
import argparse
import sys

from wulff_ml.api_key import (
    get_mp_api_key, MissingAPIKeyError, looks_like_auth_error, AUTH_HINT,
)
from wulff_ml import config
from wulff_ml.prediction import parse_mpids

MP_SEARCH_URL = "https://next-gen.materialsproject.org/materials"


def _ask_for_ids():
    print()
    print("Enter the Materials Project ID(s) of the compound(s) to predict,")
    print("for example:  mp-1077900 mp-21119   (separate several with spaces or commas)")
    print(f"You can look IDs up at {MP_SEARCH_URL}")
    while True:
        try:
            raw = input("mp-id(s): ")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)
        if not raw.strip():
            continue
        ids, bad = parse_mpids(raw)
        if bad:
            print(f"  Not a valid ID: {', '.join(bad)}  (expected something like mp-149)")
            continue
        return ids


def predict_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="wulff-predict",
        description="Predict facet surface energies and Wulff shapes from Materials Project IDs.")
    ap.add_argument("mpids", nargs="*", help="Materials Project IDs, e.g. mp-1077900 mp-21119")
    ap.add_argument("--file", help="text file with IDs (separated by spaces, commas or lines)")
    ap.add_argument("--no-wulff", action="store_true",
                    help="only predict surface energies, skip the Wulff shapes")
    ap.add_argument("--out", help="output folder (default: results/predict/<date_time>/)")
    args = ap.parse_args(argv)

    try:
        config.check_installation()
    except RuntimeError as e:
        sys.exit(str(e))

    raw = list(args.mpids)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            raw.append(f.read())

    if raw:
        ids, bad = parse_mpids(raw)
        if bad:
            sys.exit(f"Not a valid Materials Project ID: {', '.join(bad)}  "
                     "(expected something like mp-149)")
    elif sys.stdin.isatty():
        ids = _ask_for_ids()
    else:
        ap.error("no IDs given. Example: wulff-predict mp-1077900 mp-21119")

    if not ids:
        ap.error("no IDs found")

    # Ask for the API key first (once), so the user is not kept waiting
    # for a prompt after the model has been loaded or trained.
    try:
        api_key = get_mp_api_key()
    except MissingAPIKeyError as e:
        sys.exit(str(e))

    from wulff_ml.prediction import predict_compounds
    try:
        predict_compounds(ids, make_wulff=not args.no_wulff, out_dir=args.out, api_key=api_key)
    except ValueError as e:
        sys.exit(str(e))
    except Exception as e:  # noqa: BLE001
        if looks_like_auth_error(e):
            sys.exit(AUTH_HINT)
        raise


if __name__ == "__main__":
    predict_main()
