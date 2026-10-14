"""Label vocabulary shared by detectors, de-duplication and the composer.

Two labels are *compatible* when they can describe the same physical object:
identical, synonyms, one is a broader term for the other ("fruit" / "apple"),
or both are person-like. Compatibility is what lets detections from different
engines be merged instead of shown twice.
"""
from __future__ import annotations

# Person-like labels. On illustrations they are unified to "cartoon character";
# the gendered ones are never used as an automatic label (see detection docs).
PERSON_LIKE = {"person", "child", "boy", "girl", "man", "woman", "doll", "cartoon character", "cartoon",
               "cartoon boy", "cartoon girl", "kid", "baby", "people"}
GENDERED = {"boy", "girl", "man", "woman", "cartoon boy", "cartoon girl"}
CHARACTER_LABEL = "cartoon character"

# broader term -> more specific terms
HYPERNYMS = {
    "fruit": {"apple", "orange", "banana", "strawberry", "pear", "grape", "lemon", "watermelon", "cherry"},
    "animal": {"dog", "cat", "bird", "fish", "rabbit", "bear", "horse", "sheep", "cow", "elephant",
               "zebra", "giraffe"},
    "vehicle": {"car", "bus", "truck", "bicycle", "motorcycle", "train", "boat", "airplane"},
    "plant": {"tree", "flower", "potted plant"},
}
SYNONYMS = {"sofa": "couch", "motorbike": "motorcycle", "aeroplane": "airplane", "tv": "tv monitor",
            "teddy": "teddy bear"}

# Curated vocabulary for the illustration model (YOLOE prompt-free subset).
# "heart" and "balloon" were deliberately left out: in evaluation they fired on
# almost every round shape (red circle -> "heart" 0.85).
ILLUSTRATION_VOCAB = [
    "cartoon character", "child", "person", "doll",
    "apple", "orange", "fruit", "banana", "strawberry", "pear", "grape", "lemon", "watermelon", "cherry",
    "dog", "cat", "bird", "fish", "rabbit", "bear",
    "car", "tree", "flower", "house",
]
GENDER_VOCAB = ["boy", "girl"]


def canonical(label: str) -> str:
    label = (label or "").strip().lower()
    return SYNONYMS.get(label, label)


def group_of(label: str) -> str:
    """The broad group a label belongs to (used for compatibility)."""
    label = canonical(label)
    if label in PERSON_LIKE:
        return "person"
    for broad, narrow in HYPERNYMS.items():
        if label == broad or label in narrow:
            return broad
    return label


def compatible(a: str, b: str) -> bool:
    a, b = canonical(a), canonical(b)
    if a == b:
        return True
    ga, gb = group_of(a), group_of(b)
    if ga != gb:
        return False
    # Within a group, only a broad term matches a specific one ("fruit" ~ "apple"),
    # never two different specific terms ("apple" !~ "orange"). People always match.
    return ga == "person" or a in HYPERNYMS or b in HYPERNYMS


def specificity(label: str) -> int:
    """Higher = more specific. Used to pick the label kept after a merge."""
    label = canonical(label)
    if label in HYPERNYMS or label in {"cartoon", "person", "people"}:
        return 0
    if label in {"cartoon character", "child", "doll", "kid"}:
        return 1
    return 2


def is_person_like(label: str) -> bool:
    return canonical(label) in PERSON_LIKE
