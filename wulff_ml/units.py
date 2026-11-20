"""
Unit conversion for surface energy values.

The DFT pipeline stores surface energies internally in eV/Å²
(the raw Materials Project / VASP convention).  All plots and printed
tables display values in J/m², which is the standard SI unit used in
the experimental and engineering literature.

Conversion:  1 eV/Å² = 16.0218 J/m²

Usage
-----
    from wulff_ml.units import to_jm2, EV_ANG2_TO_JM2, UNIT_LABEL, UNIT_LABEL_EV

    y_jm2  = to_jm2(y_ev_ang2)           # convert a scalar, array or Series
    ax.set_ylabel(UNIT_LABEL)             # "Surface Energy γ (J/m²)"
"""

EV_ANG2_TO_JM2: float = 16.0218
"""Exact conversion factor: 1 eV/Å² → J/m²."""

# Human-readable unit strings for axis labels, table headers, etc.
UNIT_LABEL     = "Surface Energy γ (J/m²)"
UNIT_LABEL_EV  = "Surface Energy γ (eV/Å²)"   # kept as reference / secondary axis
UNIT_SHORT     = "J/m²"
UNIT_SHORT_EV  = "eV/Å²"


def to_jm2(value):
    """Convert surface energy from eV/Å² to J/m².

    Accepts a scalar, a numpy array, or a pandas Series.
    Returns the same type multiplied by EV_ANG2_TO_JM2.
    """
    return value * EV_ANG2_TO_JM2


def to_ev_ang2(value):
    """Convert surface energy from J/m² back to eV/Å² (e.g. for reference)."""
    return value / EV_ANG2_TO_JM2
