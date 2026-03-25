DEFAULT_VERTEBRA_CLASSES: dict[str, int] = {
    "vertebrae_L1": 31,
    "vertebrae_L2": 30,
    "vertebrae_L3": 29,
    "vertebrae_L4": 28,
    "vertebrae_L5": 27,
    "vertebrae_S1": 26,
}

DEFAULT_TISSUE_CLASSES: dict[str, int] = {
    "muscle": 1,
    "sat": 2,
    "vat": 3,
    "imat": 4,
}

DEFAULT_TISSUE_CLASSES_INV: dict[int, str] = {
    1: "muscle",
    2: "sat",
    3: "vat",
    4: "imat",
}

TISSUE_HU_RANGES: dict[str, tuple[int, int]] = {
    "muscle": (-29, 150),
    "imat": (-190, -30),
    "vat": (-205, -51),
}
