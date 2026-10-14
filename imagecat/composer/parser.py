"""Rule-based parser for composition instructions.

Example: "Create an image containing one red apple and one orange on a blue background"
-> items [(1, red, apple), (1, None, orange)], background "blue".

Spatial phrases are understood per item:
  * absolute positions: "on the left", "in the center", "at the top right", ...
  * relations to another item: "to the left of the apple", "above the orange",
    "behind the boy", "in front of the girl", "next to the apple"
  * size words: "big/large/huge", "small/little/tiny"
  * depth: "in the foreground", "in the background"

It is deliberately simple and transparent (no LLM): what it understood is
always shown back to the user before anything is generated.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ..colors import COLOR_NAMES, COLOR_RGB

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "pair": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
EXTRA_COLORS = {"grey": "gray", "violet": "purple", "magenta": "pink", "tan": "beige", "gold": "yellow"}
COLOR_WORDS = set(COLOR_NAMES) | set(EXTRA_COLORS)
COLOR_MODIFIERS = {"light", "dark", "bright", "pale", "deep"}
STOPWORDS = {"the", "some", "of", "with", "also", "image", "picture", "containing", "showing", "having"}
IRREGULAR_PLURALS = {"people": "person", "children": "child", "men": "man", "women": "woman",
                     "mice": "mouse", "geese": "goose", "teeth": "tooth", "feet": "foot"}
NO_SINGULAR = {"glasses", "scissors", "grass", "bus", "class", "dress", "cactus", "series"}

LAYOUTS = ("row", "overlap", "grid", "diagonal", "spatial")
SIZE_WORDS = {"big": 1.4, "large": 1.4, "huge": 1.7, "giant": 1.7, "small": 0.65, "little": 0.65,
              "tiny": 0.45}
H_POSITIONS = {"left": "left", "right": "right", "center": "center", "centre": "center", "middle": "center"}
V_POSITIONS = {"top": "top", "upper": "top", "bottom": "bottom", "lower": "bottom"}
_RELATION = re.compile(
    r"\b(?P<rel>(?:to the |on the )?left of|(?:to the |on the )?right of|above|over|on top of|below|under|"
    r"underneath|beneath|behind|in front of|next to|beside)\s+(?:the |an? )?(?P<ref>[a-z][a-z ]*?)\s*$")
_RELATION_NAMES = {"left of": "left_of", "right of": "right_of", "above": "above", "over": "above",
                   "on top of": "above", "below": "below", "under": "below", "underneath": "below",
                   "beneath": "below", "behind": "behind", "in front of": "in_front_of", "next to": "next_to",
                   "beside": "next_to"}
_POSITION = re.compile(
    r"\b(?:on|at|in|to|towards?)\s+(?:the\s+)?(?:(?P<v>top|bottom|upper|lower)(?:[- ]|\s+(?:of\s+the\s+)?)?)?"
    r"(?P<h>left|right|center|centre|middle)?(?:\s+(?:side|corner|edge|part))?(?:\s+of\s+the\s+"
    r"(?:image|picture|canvas|scene))?\b")
_DEPTH = re.compile(r"\bin the (?P<d>foreground|background)\b")
_LAYOUT_PATTERNS = [
    (r"\boverlapp?(ing|ed)?\b|\bon top of each other\b|\bstacked\b", "overlap"),
    (r"\bside[- ]by[- ]side\b|\bin a row\b|\bnext to each other\b", "row"),
    (r"\bin a grid\b|\bgrid\b", "grid"),
    (r"\bdiagonal(ly)?\b", "diagonal"),
]
_PREFIX = re.compile(
    r"^\s*(please\s+)?(can you\s+)?(create|make|generate|compose|build|draw|render|produce|give me|show|"
    r"place|put|add|position|arrange|insert)?\s*"
    r"(me\s+)?((an?|one|the)\s+)?(new\s+)?(composite\s+)?(image|picture|composition|scene|photo|collage)?\s*"
    r"(that\s+)?(containing|contains|with|showing|shows|having|has|of|featuring|including)?\s*",
)
_BACKGROUND = re.compile(r"\b(on|with|against|over)\s+(an?\s+)?(\w+)\s+background\b")


@dataclass
class RequestedItem:
    count: int
    label: str
    color: str | None = None
    text: str = ""
    position: str | None = None        # e.g. "left", "center", "top-right"
    relation: dict | None = None       # e.g. {"type": "left_of", "ref": "apple"}
    size: float | None = None          # multiplier from "big"/"small"
    depth: str | None = None           # "foreground" | "background"

    @property
    def is_spatial(self) -> bool:
        return bool(self.position or self.relation or self.depth)


@dataclass
class ParsedInstruction:
    raw: str
    items: list[RequestedItem] = field(default_factory=list)
    background: str | None = None
    layout: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def singularize(word: str) -> str:
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if word in NO_SINGULAR or len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if re.search(r"(ches|shes|xes|sses|zes)$", word):
        return word[:-2]
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _canonical_color(word: str) -> str:
    return EXTRA_COLORS.get(word, word)


def _extract_spatial(chunk: str) -> tuple[str, dict]:
    """Remove spatial phrases from a chunk and return them separately."""
    info: dict = {}
    text = " " + chunk.lower().strip() + " "
    m = _DEPTH.search(text)
    if m:
        info["depth"] = m.group("d")
        text = text[:m.start()] + " " + text[m.end():]
    m = _RELATION.search(text.strip())
    if m:
        rel = re.sub(r"^(to|on) the ", "", m.group("rel"))
        info["relation"] = {"type": _RELATION_NAMES[rel], "ref": m.group("ref").strip()}
        text = text.strip()[:m.start()]
    for m in _POSITION.finditer(text):
        v, h = m.group("v"), m.group("h")
        if not (v or h):
            continue
        parts = [V_POSITIONS[v]] if v else []
        if h:
            parts.append(H_POSITIONS[h])
        info["position"] = "-".join(parts) if parts != ["center"] else "center"
        text = text[:m.start()] + " " + text[m.end():]
        break
    return text, info


def parse_item(chunk: str) -> RequestedItem | None:
    chunk, spatial = _extract_spatial(chunk)
    size = None
    words = re.findall(r"[a-z0-9]+", chunk.lower())
    for w in words:
        if w in SIZE_WORDS:
            size = SIZE_WORDS[w]
    tokens = [t for t in words if t not in STOPWORDS and t not in SIZE_WORDS]
    if not tokens:
        return None
    count = 1
    if tokens[0].isdigit():
        count = max(1, int(tokens[0]))
        tokens = tokens[1:]
    elif tokens[0] in NUMBER_WORDS:
        count = NUMBER_WORDS[tokens[0]]
        tokens = tokens[1:]
    if tokens and tokens[0] == "of":  # "a pair of ..."
        tokens = tokens[1:]

    color = None
    rest: list[str] = []
    for i, tok in enumerate(tokens):
        is_last = i == len(tokens) - 1
        # A colour word is an adjective unless it is the final noun
        # ("one orange" is the fruit, "one orange ball" is a colour).
        if tok in COLOR_WORDS and not is_last and color is None:
            color = _canonical_color(tok)
        elif tok in COLOR_MODIFIERS and not is_last and i + 1 < len(tokens) and tokens[i + 1] in COLOR_WORDS:
            continue
        else:
            rest.append(tok)
    if not rest:
        return None
    rest[-1] = singularize(rest[-1]) if (count > 1 or rest[-1].endswith("s")) else rest[-1]
    label = " ".join(rest)
    if spatial.get("relation"):
        spatial["relation"]["ref"] = " ".join(singularize(w) if i == len(spatial["relation"]["ref"].split()) - 1
                                              else w for i, w in enumerate(spatial["relation"]["ref"].split()))
    return RequestedItem(count=count, label=label, color=color, text=chunk.strip(), size=size, **spatial)


def parse_instruction(text: str) -> ParsedInstruction:
    parsed = ParsedInstruction(raw=text)
    s = " " + (text or "").lower().strip() + " "
    s = re.sub(r"[.!?;:]+", " ", s)

    bg = _BACKGROUND.search(s)
    if bg:
        word = _canonical_color(bg.group(3))
        if word in COLOR_RGB:
            parsed.background = word
        else:
            parsed.warnings.append(f"Unknown background colour '{bg.group(3)}' ignored.")
        s = s[: bg.start()] + " " + s[bg.end():]

    for pattern, layout in _LAYOUT_PATTERNS:
        if re.search(pattern, s):
            parsed.layout = parsed.layout or layout
            s = re.sub(pattern, " ", s)
    s = re.sub(r"\b(arranged|placed|positioned|together|each other)\b", " ", s)

    s = _PREFIX.sub("", s.strip(), count=1)
    chunks = [c for c in re.split(r",|\band\b|\bplus\b|&|\balong with\b", s) if c.strip()]
    for chunk in chunks:
        item = parse_item(chunk)
        if item is not None:
            parsed.items.append(item)
        else:
            parsed.warnings.append(f"Could not understand '{chunk.strip()}'.")
    if not parsed.items:
        parsed.warnings.append("No objects found in the instruction.")
    elif parsed.layout is None and any(i.is_spatial for i in parsed.items):
        parsed.layout = "spatial"
    labels = {i.label for i in parsed.items}
    for item in parsed.items:
        ref = (item.relation or {}).get("ref")
        if ref and ref not in labels and not any(lbl.endswith(" " + ref) for lbl in labels):
            parsed.warnings.append(f"'{item.label}' is placed relative to '{ref}', which is not in the request.")
    return parsed
