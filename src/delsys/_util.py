"""Small utility helpers used across the delsys package."""

from typing import List, Set


def _mod_to_attr(name: str) -> str:
    """Convert a modality tag to the :class:`Sensor` attribute name that holds it.

    Maps every EMG variant (``'EMG'``, ``'EMGS'``, ``'EMGD'``, ``'EMGQ'``) to
    ``'emg'`` so EMG-bundle access is uniform regardless of sensor flavor.
    Link devices have longer attribute names than their modality tags
    (``'VO2'`` → ``'vo2master'``, ``'HR'`` → ``'hrstrap'``); everything else
    is just lower-cased.

    Args:
        name: Modality tag — case-insensitive.

    Returns:
        The lower-case attribute name used on :class:`Sensor` instances.
    """
    if name.upper().startswith("EMG"):
        return "emg"
    overrides = {"vo2": "vo2master", "hr": "hrstrap"}
    return overrides.get(name.lower(), name.lower())


def _modset_to_strlist(modset: Set[str]) -> List[str]:
    """Flatten a set of modality tags into the strings legacy lookup compares against.

    EMG variants collapse to ``'EMG'``. Each entry is included in both its
    original case and lower-cased form so substring matches are
    case-insensitive.

    Args:
        modset: Set of modality tags (e.g. ``{'EMGS', 'ACC', 'GYRO'}``).

    Returns:
        Flattened list including the canonical (case-folded EMG) labels and
        their lower-case equivalents.
    """
    modality_strings = ["EMG" if x.startswith("EMG") else x for x in modset]
    return modality_strings + [x.lower() for x in modality_strings]
