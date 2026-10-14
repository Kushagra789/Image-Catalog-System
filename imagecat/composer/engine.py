"""Composite image generation from real catalog objects.

Pipeline: parse instruction -> find matching catalog objects -> cut them out ->
lay them out on a canvas -> render -> save as a new catalog entry with full
provenance.

Nothing is ever generated or hallucinated: every pixel of an object layer comes
from a catalog image. Requested objects that do not exist in the catalog are
reported as missing; saving with missing items requires ``allow_partial=True``.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..colors import COLOR_RGB
from ..models import ImageRecord, ObjectRecord, SOURCE_COMPOSITE, STATUS_SKIPPED
from ..service import CatalogService
from .cutout import CutoutError, extract_cutout
from .parser import EXTRA_COLORS, LAYOUTS, ParsedInstruction, RequestedItem, parse_instruction, singularize

GENERATOR = "imagecat rule-based compositor v1 (cut-and-paste of catalog objects; no generative AI)"


class CompositionError(Exception):
    pass


class MissingAssetsError(CompositionError):
    def __init__(self, missing: list["MissingItem"]):
        self.missing = missing
        names = ", ".join(f"{m.needed - m.found}x {m.describe()}" for m in missing)
        super().__init__(f"Missing catalog assets: {names}.")


@dataclass
class MissingItem:
    label: str
    color: str | None
    needed: int
    found: int
    reason: str
    suggestions: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return f"{self.color + ' ' if self.color else ''}{self.label}"


@dataclass
class Layer:
    label: str
    color: str | None
    requested: str
    source_image_id: int
    source_object_id: int
    match_type: str
    mask_method: str = "auto"
    mask_used: str = ""
    mask_note: str = ""
    x: int = 0
    y: int = 0
    scale: float = 1.0
    z: int = 0
    width: int = 0   # cutout size before scaling
    height: int = 0
    # Spatial request for this layer (from the instruction) and how it was honoured.
    request_label: str = ""
    position: str | None = None
    relation: dict | None = None
    size: float | None = None
    depth: str | None = None
    placement: str = ""
    # What the source looked like when the composite was made (provenance that
    # survives later edits or deletion of the source).
    source_snapshot: dict = field(default_factory=dict)


@dataclass
class CompositionPlan:
    instruction: str
    parsed: ParsedInstruction
    canvas_width: int
    canvas_height: int
    background: str
    layout: str
    layers: list[Layer]
    missing: list[MissingItem]
    cutouts: dict[int, Image.Image] = field(default_factory=dict, repr=False)  # index -> RGBA
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return not self.missing and bool(self.layers)

    def metadata(self) -> dict[str, Any]:
        return {
            "generator": GENERATOR,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "instruction": self.instruction,
            "parsed": self.parsed.to_dict(),
            "canvas": {"width": self.canvas_width, "height": self.canvas_height,
                       "background": self.background},
            "layout": self.layout,
            "options": self.options,
            "layers": [asdict(l) for l in self.layers],
            "missing": [asdict(m) for m in self.missing],
            "complete": self.is_complete,
        }


def _norm_color(c: str | None) -> str | None:
    if not c:
        return None
    c = c.strip().lower()
    return EXTRA_COLORS.get(c, c)


def _norm_label(label: str) -> str:
    words = re.findall(r"[a-z0-9]+", label.lower())
    if words:
        words[-1] = singularize(words[-1])
    return " ".join(words)


def label_match(requested: str, label: str) -> str | None:
    """'exact', 'specific' (catalog label is a more specific form, e.g.
    'boy' matches 'cartoon boy'), or None. No synonyms: an apple is never
    substituted by a different object."""
    r, l = _norm_label(requested), _norm_label(label)
    if r == l:
        return "exact"
    if l.endswith(" " + r):
        return "specific"
    return None


def parse_background(value: str | None) -> tuple[int, int, int]:
    if not value:
        return (255, 255, 255)
    v = value.strip().lower()
    if re.fullmatch(r"#?[0-9a-f]{6}", v):
        v = v.lstrip("#")
        return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    v = EXTRA_COLORS.get(v, v)
    if v in COLOR_RGB:
        return COLOR_RGB[v] if v != "white" else (255, 255, 255)
    raise CompositionError(f"Unknown background colour '{value}'.")


class Composer:
    def __init__(self, service: CatalogService):
        self.service = service
        self.db = service.db

    # ---------------------------------------------------------- asset search
    def _rank_key(self, obj: ObjectRecord, sizes: dict[int, tuple[int, int]]):
        verified = obj.is_correction  # a human looked at / created it
        w, h = sizes.get(obj.image_id, (0, 0))
        # Objects cut off by the image frame, or partly hidden behind another
        # object, make poor cutouts (missing pieces), so fully visible ones win.
        truncated = obj.x <= 1 or obj.y <= 1 or obj.x + obj.width >= w - 1 or obj.y + obj.height >= h - 1
        covered = float(obj.attributes.get("occluded_fraction") or 0.0)
        occluded = bool(obj.attributes.get("occluded")) or covered > 0.015
        conf = obj.confidence if obj.confidence is not None else -1.0
        return (not verified, truncated, occluded, covered > 0.002, -conf, -obj.bbox.area)

    def find_assets(self, items: list[RequestedItem], strict_color: bool = True,
                    include_composites: bool = False) -> tuple[list[tuple[RequestedItem, ObjectRecord, str]],
                                                               list[MissingItem]]:
        objects = self.db.all_active_objects(include_composites=include_composites)
        sizes = {r.id: (r.width, r.height) for r in self.db.list_images()}
        used: set[int] = set()
        matches: list[tuple[RequestedItem, ObjectRecord, str]] = []
        missing: list[MissingItem] = []
        for item in items:
            want_color = _norm_color(item.color)
            all_matches = [(o, m) for o, m in ((o, label_match(item.label, o.label)) for o in objects) if m]
            by_label = [(o, m) for o, m in all_matches if o.id not in used]
            exact_color = [(o, m) for o, m in by_label if not want_color or _norm_color(o.color) == want_color]
            pool = exact_color
            match_suffix = ""
            if want_color and len(exact_color) < item.count and not strict_color:
                others = [(o, m) for o, m in by_label if (o, m) not in exact_color]
                pool = exact_color + others
                match_suffix = "+color_ignored"
            pool.sort(key=lambda om: (om[1] != "exact",) + self._rank_key(om[0], sizes))
            chosen = pool[: item.count]
            for o, m in chosen:
                used.add(o.id)
                tag = m
                if want_color and _norm_color(o.color) != want_color:
                    tag += match_suffix
                matches.append((item, o, tag))
            if len(chosen) < item.count:
                missing.append(self._missing(item, len(chosen), all_matches, used, objects))
        return matches, missing

    @staticmethod
    def _missing(item: RequestedItem, found: int, all_matches, used: set[int], objects) -> MissingItem:
        """Explain a shortfall using *all* matching objects, including ones already
        assigned to earlier items of the same request."""
        want_color = _norm_color(item.color)
        suggestions: list[str] = []
        in_color = [o for o, _ in all_matches if not want_color or _norm_color(o.color) == want_color]
        by_label = all_matches
        if all_matches and want_color and not in_color:
            colors = sorted({o.color or "unknown" for o, _ in all_matches})
            reason = (f"'{item.label}' exists in the catalog but not in colour '{want_color}' "
                      f"(available colours: {', '.join(colors)}).")
            suggestions.append("Untick 'strict colour matching' to use another colour, or correct the colour "
                               "in the Manual Editor if it was mis-detected.")
        elif in_color:
            already = sum(1 for o in in_color if o.id in used)
            reason = f"Only {len(in_color)} distinct '{item.label}' object(s) in the catalog"
            reason += f" ({already} already used by earlier items in this request)." if already else "."
        else:
            reason = f"No object labelled '{item.label}' in the catalog."
        if not by_label and want_color:
            same_color = [o for o in objects if _norm_color(o.color) == want_color][:5]
            for o in same_color:
                suggestions.append(
                    f"Object #{o.id} in image #{o.image_id} is a {want_color} '{o.label}'. If it really is a "
                    f"{item.label}, relabel it in the Manual Editor.")
        if not suggestions:
            suggestions.append(f"Upload an image containing a {item.label} or add it manually in the Manual Editor.")
        return MissingItem(item.label, want_color, item.count, found, reason, suggestions)

    # ------------------------------------------------------------- planning
    def plan(self, instruction: str, canvas_size: tuple[int, int] = (800, 600),
             background: str | None = None, layout: str | None = None, mask_method: str = "auto",
             strict_color: bool = True, include_composites: bool = False) -> CompositionPlan:
        parsed = parse_instruction(instruction)
        if not parsed.items:
            raise CompositionError("; ".join(parsed.warnings) or "No objects requested.")
        w, h = int(canvas_size[0]), int(canvas_size[1])
        if not (64 <= w <= 4096 and 64 <= h <= 4096):
            raise CompositionError("Canvas size must be between 64 and 4096 pixels.")
        bg = background or parsed.background or "white"
        parse_background(bg)  # validate
        lay = layout or parsed.layout or "row"
        if lay not in LAYOUTS:
            raise CompositionError(f"Unknown layout '{lay}'.")
        if layout and layout != "spatial" and any(i.is_spatial for i in parsed.items):
            parsed.warnings.append(f"Positions in the instruction are ignored because the '{layout}' layout "
                                   "was chosen. Pick 'spatial' (or 'from instruction') to use them.")

        matches, missing = self.find_assets(parsed.items, strict_color, include_composites)
        layers: list[Layer] = []
        cutouts: dict[int, Image.Image] = {}
        for item, obj, match_type in matches:
            record = self.db.get_image(obj.image_id)
            if record is None:
                continue
            path = self.service.catalog.path_for(record)
            try:
                cut, used, note = extract_cutout(path, obj, mask_method)
            except (CutoutError, OSError) as exc:
                missing.append(MissingItem(item.label, _norm_color(item.color), 1, 0,
                                           f"Object #{obj.id} could not be cut out: {exc}"))
                continue
            cutouts[len(layers)] = cut
            layers.append(Layer(
                label=obj.label, color=obj.color, requested=item.text or item.label,
                source_image_id=obj.image_id, source_object_id=obj.id, match_type=match_type,
                mask_method=mask_method, mask_used=used, mask_note=note,
                width=cut.width, height=cut.height,
                request_label=item.label, position=item.position, relation=item.relation,
                size=item.size, depth=item.depth, source_snapshot=_snapshot(record, obj),
            ))
        plan = CompositionPlan(
            instruction=instruction, parsed=parsed, canvas_width=w, canvas_height=h,
            background=bg, layout=lay, layers=layers, missing=missing, cutouts=cutouts,
            options={"mask_method": mask_method, "strict_color": strict_color,
                     "include_composites": include_composites},
        )
        auto_layout(plan)
        return plan

    # ------------------------------------------------------------ rendering
    @staticmethod
    def render(plan: CompositionPlan) -> tuple[Image.Image, list[tuple[int, int, int, int] | None]]:
        """Render the plan. Returns the image and each layer's visible bbox."""
        canvas = Image.new("RGBA", (plan.canvas_width, plan.canvas_height), parse_background(plan.background) + (255,))
        boxes: list[tuple[int, int, int, int] | None] = [None] * len(plan.layers)
        order = sorted(range(len(plan.layers)), key=lambda i: plan.layers[i].z)
        for i in order:
            layer = plan.layers[i]
            cut = plan.cutouts.get(i)
            if cut is None:
                continue
            sw = max(1, round(cut.width * layer.scale))
            sh = max(1, round(cut.height * layer.scale))
            scaled = cut.resize((sw, sh), Image.LANCZOS) if (sw, sh) != cut.size else cut
            layer_img = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            layer_img.paste(scaled, (int(layer.x), int(layer.y)))
            canvas = Image.alpha_composite(canvas, layer_img)
            alpha = np.asarray(layer_img)[:, :, 3]
            ys, xs = np.nonzero(alpha > 10)
            if len(xs):
                boxes[i] = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        return canvas.convert("RGB"), boxes

    # --------------------------------------------------------------- saving
    def save(self, plan: CompositionPlan, display_name: str | None = None,
             allow_partial: bool = False) -> ImageRecord:
        if not plan.layers:
            raise CompositionError("Nothing to save: none of the requested objects were found in the catalog.")
        if plan.missing and not allow_partial:
            raise MissingAssetsError(plan.missing)
        image, boxes = self.render(plan)
        meta = plan.metadata()
        meta["layer_boxes"] = boxes
        source_ids = {
            "image_ids": sorted({l.source_image_id for l in plan.layers}),
            "object_ids": [l.source_object_id for l in plan.layers],
        }
        name = "composite_" + "_".join(re.findall(r"[a-z]+", " ".join(l.label for l in plan.layers)))[:60]
        record = self.service.catalog.add_pil_image(
            image, name, display_name or f"Composite: {plan.instruction[:60]}",
            detection_status=STATUS_SKIPPED,
            detection_message=("Composite image: objects recorded from composition metadata (exact placement), "
                               "not from detection."),
            is_composite=True, source_asset_ids=source_ids, composition_metadata=meta,
        )
        for i, (layer, box) in enumerate(zip(plan.layers, boxes)):
            if box is None:
                continue  # layer was placed completely off-canvas
            polygon = _layer_polygon(plan, i)
            self.db.insert_object(
                image_id=record.id, label=layer.label, color=layer.color, confidence=None,
                bbox=box, source=SOURCE_COMPOSITE,
                attributes={
                    "source_image_id": layer.source_image_id,
                    "source_object_id": layer.source_object_id,
                    "z": layer.z, "mask_used": layer.mask_used, "scale": layer.scale,
                    "polygon": polygon,
                    "note": "Placed by the compositor; no confidence because nothing was detected.",
                },
            )
        return self.db.get_image(record.id)  # type: ignore[return-value]


def _layer_polygon(plan: CompositionPlan, i: int) -> list[list[int]] | None:
    layer, cut = plan.layers[i], plan.cutouts.get(i)
    if cut is None:
        return None
    sw, sh = max(1, round(cut.width * layer.scale)), max(1, round(cut.height * layer.scale))
    alpha = np.asarray(cut.resize((sw, sh)))[:, :, 3]
    contours, _ = cv2.findContours((alpha > 10).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True)
    return [[int(p[0][0] + layer.x), int(p[0][1] + layer.y)] for p in c]


def auto_layout(plan: CompositionPlan) -> None:
    """Assign x, y, scale and z to every layer according to plan.layout."""
    n = len(plan.layers)
    if n == 0:
        return
    W, H = plan.canvas_width, plan.canvas_height
    layers = plan.layers
    target_h = 0.6 * H

    if plan.layout in ("row", "overlap", "diagonal"):
        step_frac = {"row": 1.0, "overlap": 0.6, "diagonal": 0.75}[plan.layout]
        scales = [min(target_h / l.height, (0.9 * W / n) / l.width if plan.layout == "row" else 10.0)
                  for l in layers]
        widths = [l.width * s for l, s in zip(layers, scales)]
        gap = 0.04 * W if plan.layout == "row" else 0.0
        total = sum(widths[:-1]) * step_frac + widths[-1] + gap * (n - 1)
        shrink = min(1.0, 0.95 * W / total) if total > 0 else 1.0
        scales = [s * shrink for s in scales]
        widths = [w * shrink for w in widths]
        total *= shrink
        x = (W - total) / 2
        for i, (l, s, w) in enumerate(zip(layers, scales, widths)):
            h = l.height * s
            if plan.layout == "diagonal" and n > 1:
                y = 0.05 * H + (0.9 * H - h) * i / (n - 1)
            elif plan.layout == "overlap":
                y = (H - h) / 2 + (0.06 * H if i % 2 else -0.06 * H) * (n > 1)
            else:
                y = (H - h) / 2
            l.x, l.y, l.scale, l.z = int(round(x)), int(round(max(0, min(H - h, y)))), round(s, 4), i
            x += w * step_frac + gap
    elif plan.layout == "spatial":
        _spatial_layout(plan)
    elif plan.layout == "grid":
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        cw, ch = W / cols, H / rows
        for i, l in enumerate(layers):
            r, c = divmod(i, cols)
            s = min(0.85 * cw / l.width, 0.85 * ch / l.height)
            l.scale = round(s, 4)
            l.x = int(round(c * cw + (cw - l.width * s) / 2))
            l.y = int(round(r * ch + (ch - l.height * s) / 2))
            l.z = i
    else:
        raise CompositionError(f"Unknown layout '{plan.layout}'.")


H_ANCHOR = {"left": 1 / 6, "center": 0.5, "right": 5 / 6}
V_ANCHOR = {"top": 1 / 6, "middle": 0.5, "bottom": 5 / 6}
DEPTH_SCALE = {"foreground": 1.15, "background": 0.85}


def _anchor(position: str) -> tuple[str, str]:
    parts = position.split("-")
    v = parts[0] if parts[0] in V_ANCHOR else "middle"
    h = parts[-1] if parts[-1] in H_ANCHOR else "center"
    return h, v


def _find_ref(layers: list[Layer], ref: str, me: int) -> int | None:
    for j, other in enumerate(layers):
        if j != me and (label_match(ref, other.request_label) or label_match(ref, other.label)):
            return j
    return None


def _spatial_layout(plan: CompositionPlan) -> None:
    """Place layers on a 3x3 grid of anchors and resolve relations.

    Absolute positions use the centres of the canvas thirds. Items without a
    position fill free columns of the middle row. Several items sharing a cell
    split it. Relations ("left of the apple", "behind the girl") are placed
    next to / around the referenced layer.
    """
    W, H = plan.canvas_width, plan.canvas_height
    layers = plan.layers
    uses_rows = any(l.position and _anchor(l.position)[1] != "middle" for l in layers) or any(
        (l.relation or {}).get("type") in ("above", "below") for l in layers)
    cell_w, cell_h = W / 3, (H / 3 if uses_rows else 0.75 * H)

    def base_scale(l: Layer, share: int = 1) -> float:
        s = min(0.85 * cell_h / l.height, 0.85 * (cell_w / share) / l.width)
        return s * (l.size or 1.0) * DEPTH_SCALE.get(l.depth or "", 1.0)

    cells: dict[tuple[str, str], list[int]] = {}
    free, relational = [], []
    for i, l in enumerate(layers):
        if l.relation:
            relational.append(i)
        elif l.position:
            cells.setdefault(_anchor(l.position), []).append(i)
        else:
            free.append(i)
    if free and not cells:
        # Nothing has an absolute position: spread the free items evenly.
        spread = {1: ["center"], 2: ["left", "right"]}.get(len(free), ["left", "center", "right"])
        for k, i in enumerate(free):
            cell = (spread[k % len(spread)], "middle")
            cells.setdefault(cell, []).append(i)
            layers[i].placement = f"{cell[1]}-{cell[0]} (automatic: no position requested)"
        free = []
    for i in free:
        options = [(h, "middle") for h in ("left", "center", "right")]
        cell = min(options, key=lambda c: (len(cells.get(c, [])), options.index(c)))
        cells.setdefault(cell, []).append(i)
        layers[i].placement = f"{cell[1]}-{cell[0]} (automatic: no position requested)"

    for (h, v), idxs in cells.items():
        n = len(idxs)
        for k, i in enumerate(idxs):
            l = layers[i]
            s = base_scale(l, n)
            cx = H_ANCHOR[h] * W + (k - (n - 1) / 2) * (cell_w / n)
            cy = V_ANCHOR[v] * H
            l.scale = round(s, 4)
            l.x, l.y = int(round(cx - l.width * s / 2)), int(round(cy - l.height * s / 2))
            if l.position:
                l.placement = f"{l.position} (from instruction)"
            l.z = i

    placed = set(i for idxs in cells.values() for i in idxs)
    pending = list(relational)
    for _ in range(len(pending) + 1):  # resolve chains such as A left of B, B above C
        still = []
        for i in pending:
            l = layers[i]
            ref = _find_ref(layers, l.relation["ref"], i)
            if ref is None:
                plan.parsed.warnings.append(
                    f"'{l.label}': reference '{l.relation['ref']}' was not found; placed automatically.")
                l.relation = None
                l.placement = "automatic (reference missing)"
                l.scale = round(base_scale(l), 4)
                l.x, l.y, l.z = int((W - l.width * l.scale) / 2), int((H - l.height * l.scale) / 2), i
                placed.add(i)
                continue
            if ref not in placed:
                still.append(i)
                continue
            _place_relative(l, layers[ref], W, base_scale(l))
            l.placement = f"{l.relation['type'].replace('_', ' ')} '{layers[ref].label}' (from instruction)"
            placed.add(i)
        pending = still
        if not pending:
            break
    for i in pending:  # circular references
        layers[i].placement = "automatic (circular reference)"

    for l in layers:
        if l.depth == "background":
            l.z -= 100
        elif l.depth == "foreground":
            l.z += 100
        # Keep every layer fully on the canvas.
        w, h = l.width * l.scale, l.height * l.scale
        if w > W or h > H:
            l.scale = round(l.scale * min(W / w, H / h), 4)
            w, h = l.width * l.scale, l.height * l.scale
        l.x = int(min(max(0, l.x), W - w))
        l.y = int(min(max(0, l.y), H - h))


def _place_relative(l: Layer, ref: Layer, W: int, scale: float) -> None:
    rw, rh = ref.width * ref.scale, ref.height * ref.scale
    # Without an explicit size word, a relational item is never taller than its
    # reference, so the pair looks balanced.
    l.scale = round(scale if l.size is not None else min(scale, rh / l.height), 4)
    w, h = l.width * l.scale, l.height * l.scale
    rcx, rcy = ref.x + rw / 2, ref.y + rh / 2
    gap = 0.03 * W
    kind = l.relation["type"]
    if kind == "left_of":
        l.x, l.y = ref.x - w - gap, rcy - h / 2
    elif kind in ("right_of", "next_to"):
        l.x, l.y = ref.x + rw + gap, rcy - h / 2
    elif kind == "above":
        l.x, l.y = rcx - w / 2, ref.y - h - gap
    elif kind == "below":
        l.x, l.y = rcx - w / 2, ref.y + rh + gap
    elif kind == "behind":
        l.x, l.y = rcx - w / 2 + 0.35 * rw, rcy - h / 2 - 0.08 * rh
    elif kind == "in_front_of":
        l.x, l.y = rcx - w / 2 - 0.3 * rw, rcy - h / 2 + 0.08 * rh
    l.x, l.y = int(round(l.x)), int(round(l.y))
    l.z = ref.z - 1 if kind == "behind" else ref.z + 1


def _snapshot(record: ImageRecord, obj: ObjectRecord) -> dict:
    return {
        "image": {"id": record.id, "display_name": record.display_name, "filename": record.filename,
                  "sha256": record.file_hash, "width": record.width, "height": record.height},
        "object": {"id": obj.id, "label": obj.label, "color": obj.color, "confidence": obj.confidence,
                   "bbox": list(obj.bbox.as_tuple()), "source": obj.source, "origin": obj.origin,
                   "manually_corrected": obj.is_correction},
    }
