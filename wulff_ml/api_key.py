"""
Materials Project API key handling.

The key is looked up in this order:

  1. the environment variable ``MP_API_KEY``
  2. the key saved by a previous run (``~/.wulff_ml/mp_api_key``)
  3. an interactive prompt in the terminal (only when running in a terminal)

When the key is typed into the prompt it is checked against the Materials
Project, saved to ``~/.wulff_ml/mp_api_key`` (readable only by the current
user, and outside the repository so it can never be committed by accident)
and used automatically from then on.

Command line (installed with the package; ``python -m wulff_ml.api_key`` works too):

    wulff-key             # enter (or replace) the saved key
    wulff-key --show      # show where the key comes from (masked)
    wulff-key --forget    # delete the saved key

Set ``WULFF_ML_HOME`` to store the key somewhere else than ``~/.wulff_ml``.
"""
import getpass
import os
import re
import sys
from pathlib import Path
from typing import Optional

ENV_VAR = "MP_API_KEY"
KEY_URL = "https://next-gen.materialsproject.org/api"
MAX_ATTEMPTS = 3


class MissingAPIKeyError(RuntimeError):
    """Raised when a Materials Project API key is needed but none is available."""


# ==============================================================================
# Storage
# ==============================================================================
def key_file() -> Path:
    home = os.environ.get("WULFF_ML_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".wulff_ml"
    return base / "mp_api_key"


def _read_saved_key() -> Optional[str]:
    path = key_file()
    try:
        key = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    return key or None


def _save_key(key: str) -> Optional[Path]:
    """Write the key to disk, readable only by the current user.
    Returns the path, or None if the file could not be written."""
    path = key_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create the file with 0600 permissions from the start.
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(key + "\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass  # e.g. Windows: permissions are managed differently
        return path
    except OSError as e:
        print(f"  Could not save the key ({e}). It will be used for this run only.")
        return None


def forget_key() -> bool:
    """Delete the saved key. Returns True if a key file was removed."""
    path = key_file()
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


# ==============================================================================
# Validation
# ==============================================================================
def _mask(key: str) -> str:
    return key[:4] + "…" + key[-4:] if len(key) > 8 else "…"


_AUTH_TOKENS = ("403", "401", "forbidden", "unauthorized", "invalid api key", "api key")


def looks_like_auth_error(exc: Exception) -> bool:
    """True if an exception from the Materials Project means 'bad API key'."""
    msg = str(exc).lower()
    return any(token in msg for token in _AUTH_TOKENS)


AUTH_HINT = ("The Materials Project did not accept the stored API key.\n"
             "Enter a new one with:  wulff-key")


def _check_with_mp(key: str):
    """Ask the Materials Project whether the key works.

    Returns True (accepted), False (rejected) or None (could not check, for
    example because there is no internet connection)."""
    try:
        from mp_api.client import MPRester

        with MPRester(key) as mpr:
            mpr.materials.summary.search(material_ids=["mp-149"],
                                         fields=["material_id"])
        return True
    except Exception as e:  # noqa: BLE001 - classify by message below
        return False if looks_like_auth_error(e) else None


# ==============================================================================
# Main entry point
# ==============================================================================
def get_mp_api_key(required: bool = True, allow_skip: bool = False) -> Optional[str]:
    """Return the Materials Project API key, asking for it if necessary.

    Parameters
    ----------
    required   : raise MissingAPIKeyError if no key can be obtained
                 (otherwise return None).
    allow_skip : in the prompt, an empty answer cancels instead of asking
                 again; returns None (or raises, if ``required``).
    """
    key = os.environ.get(ENV_VAR, "").strip()
    if key:
        return key

    key = _read_saved_key()
    if key:
        os.environ[ENV_VAR] = key
        return key

    if sys.stdin is not None and sys.stdin.isatty():
        key = _prompt_for_key(allow_skip=allow_skip)
        if key:
            os.environ[ENV_VAR] = key
            return key
    if required:
        raise MissingAPIKeyError(
            "No Materials Project API key found.\n"
            f"  1. Get a free key at {KEY_URL}\n"
            "  2. Then either run   wulff-key   to store it,\n"
            f"     or set the environment variable {ENV_VAR}."
        )
    return None


def _prompt_for_key(allow_skip: bool = False) -> Optional[str]:
    print()
    print("-" * 70)
    print(" Materials Project API key needed")
    print("-" * 70)
    print(" This step downloads data from the Materials Project, which needs a")
    print(" free API key:")
    print(f"   1. Log in (or register) at {KEY_URL}")
    print("   2. Copy the key shown on that page")
    print(" Paste it below. Nothing is displayed while you type or paste; that")
    print(" is normal. The key is saved on this computer so you only need to")
    print(" do this once.")
    if allow_skip:
        print(" Press Enter without typing anything to skip this part.")
    print("-" * 70)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            key = getpass.getpass("API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not key:
            if allow_skip:
                return None
            print("  No key entered.")
            continue
        if re.search(r"\s", key):
            print("  A key never contains spaces. Please paste it again.")
            continue

        print("  Checking the key with the Materials Project ...")
        verdict = _check_with_mp(key)
        if verdict is False:
            print("  The Materials Project rejected this key. Please check it and try again.")
            continue
        if verdict is None:
            print("  Could not verify the key (no connection?). Using it for this run only;")
            print("  it is not saved. Run again with a working connection to store it.")
            print()
            return key
        print("  Key accepted.")

        path = _save_key(key)
        if path:
            print(f"  Saved to {path}")
            print("  It will be used automatically from now on.")
            print("  To remove it again: wulff-key --forget")
        print()
        return key

    print(f"  No valid key after {MAX_ATTEMPTS} attempts.")
    return None


# ==============================================================================
# wulff-key
# ==============================================================================
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if "--forget" in argv:
        if forget_key():
            print(f"Deleted {key_file()}")
        else:
            print("No saved key found.")
        return 0

    if "--show" in argv:
        env = os.environ.get(ENV_VAR, "").strip()
        saved = _read_saved_key()
        if env:
            print(f"Using the key from the environment variable {ENV_VAR}: {_mask(env)}")
        elif saved:
            print(f"Using the saved key in {key_file()}: {_mask(saved)}")
        else:
            print("No key set. Run: wulff-key")
        return 0

    if not (sys.stdin is not None and sys.stdin.isatty()):
        print("This command needs a terminal. Alternatively set the "
              f"environment variable {ENV_VAR}.")
        return 1

    key = _prompt_for_key(allow_skip=True)
    if not key:
        print("No key saved.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
