# Project Report: Image Catalog, Object Detection, Metadata Management and Composite Image Generation System

_Report date: 2026-10-14 · Platform: Windows 11, Python 3.12.10, Streamlit 1.65.0 · Repository: `C:\Users\kusha\Image-Catalog-System`_

---

## 1. Summary

A local web application that

1. keeps a **catalog** of images (files on disk, metadata in SQLite),
2. **detects objects** automatically on upload, using three complementary engines (a COCO YOLO model for
   photographs, an open-vocabulary YOLOE model restricted to an illustration vocabulary, and classical OpenCV
   analysis for shapes, colour regions and cartoon figures),
3. lets users **correct** every detection (label, colour, confidence, bounding box, drawn and edited with the
   mouse). Corrections are stored durably and are never overwritten without explicit permission,
4. builds **composite images** from natural-language instructions ("Place an apple on the left, an orange on
   the right, and a cartoon boy in the center") by cutting the *actual* objects out of catalog images. It never
   generates or approximates missing content, and every composite records full provenance.

Size: ~5,200 lines of application code (Python + one JavaScript component) and ~1,700 lines of tests
(final suite: see §8).

---

## 2. Requirements traceability

| # | Requirement | Implementation | Evidence |
|---|---|---|---|
| 1 | Upload, preview, list, delete; local files + SQLite; ID, name, display name, filename, width, height, timestamp, status | `catalog.py`, `db.py`, pages *Upload Images*, *Image Catalog* | `test_catalog.py`, UI smoke tests |
| 2 | Automatic detection; photos and cartoons; YOLO + OpenCV; names, colours, confidences, boxes, source; report failure/empty; no fabricated confidence | `detection/` (engine, yolo, yoloe, opencv, merge, labels) | `test_detection.py`, `test_hybrid_detection.py`, `test_fixtures_and_yolo.py` |
| 3 | Add/edit/remove objects; persist; authoritative; confirmation before replacing | `service.py` (correction policy), *Manual Editor*, mouse editor | `test_corrections.py`, `test_regressions.py`, `test_bbox_editor_*.py` |
| 4 | NL instructions; real catalog objects; crop/mask/resize/position/overlap; no silent AI replacement; report missing; save with source IDs + metadata | `composer/` (parser, engine, cutout) | `test_composer.py`, `test_spatial_composition.py`, `test_cutout_rembg.py` |
| 5 | Test data: apple, orange, cartoon boy/girl, shapes, colors, apple_orange, overlapping_shapes; scenario tests; real uploads; reproducible fixtures | `fixtures.py` (+ blank), `test_data/real_cartoons/` (6 licensed real images) | `test_fixtures_and_yolo.py` |
| 6 | Pages: Dashboard, Upload, Catalog, Detection, Manual Editor, Composite Generator, Metadata Viewer | `app.py`, `imagecat/ui/` | `test_ui_smoke.py` (each page with and without data) |
| 7 | Modular code, requirements, README, .gitignore, pytest, Windows venv, graceful missing deps, honest testing, limitations | whole repository | this report, `README.md` |

---

## 3. Architecture

```
                 ┌───────────────────────── Streamlit UI (app.py, imagecat/ui/) ─────────────────────────┐
                 │ Dashboard · Upload · Catalog · Object Detection · Manual Editor (+ mouse box editor)   │
                 │ Composite Generator · Metadata Viewer                                                   │
                 └───────────────┬───────────────────────────────┬──────────────────────────┬─────────────┘
                                 │                               │                          │
                     CatalogService (service.py)          Composer (composer/engine.py)     │
        upload · run_detection (correction policy,      parse → find assets → cut out →     │
        transactional) · add/update/delete/restore      layout → render → save + provenance │
        objects · health_report                                  │                          │
             │                │                                  │                          │
     ImageCatalog       DetectionEngine (detection/engine.py)    cutout.py (alpha / polygon /│
     (catalog.py)       ├─ OpenCVDetector  (classical)            rembg / key / GrabCut)     │
       files in         ├─ YoloDetector    (COCO, photos+all)                                │
       data/images      ├─ IllustrationDetector (YOLOE subset, illustrations)                │
             │          └─ merge.py + labels.py (de-duplication)                             │
             └──────────────────────────────┬──────────────────────────────────────────────┘
                                     Database (db.py) → data/catalog.db (SQLite)
```

**Main workflows**

1. *Upload*: validate the file → EXIF-rotate → SHA-256 duplicate check → store the file → insert the image row →
   detect → in one transaction: insert objects, log the run, set the status.
2. *Detect*: classify the image (blank / flat illustration / photo) → OpenCV (flat only) → YOLO → YOLOE (flat
   only) → unify person-like labels → merge duplicates → attach unverified boy/girl suggestions → status.
3. *Correct*: every edit is validated (box inside the image, confidence in 0–1 or empty, non-empty label) and
   stored immediately. Edited auto detections keep their original values and an edit history.
4. *Re-detect*: "keep corrections" (default) replaces only untouched automatic objects. "Replace everything"
   requires an explicit confirmation. Both run in a write-locked transaction that re-reads corrections.
5. *Compose*: parse → match catalog objects (no substitution) → report missing → cut out → lay out → user
   adjusts → render → save as a new catalog image with provenance.

**Module map**: see `README.md` → *Project structure*.

---

## 4. Technologies

| Technology | Version | Role |
|---|---|---|
| Python | 3.12.10 | language (3.12 chosen over the installed 3.14 for PyTorch/Ultralytics compatibility) |
| Streamlit | 1.65.0 | web UI; Components v2 for the mouse box editor |
| SQLite (sqlite3) | 3.49.1 | metadata store with foreign keys and CHECK constraints |
| OpenCV | 5.0.0 | colour segmentation, contours, shape fitting, GrabCut |
| Pillow | 12.3.0 | image I/O, EXIF rotation, rendering |
| Ultralytics | 8.4.176 | `yolo11n-seg` (COCO) and `yoloe-26s-seg-pf` (open vocabulary) |
| PyTorch (CPU) | 2.14.1 | inference backend for Ultralytics |
| rembg + onnxruntime | 2.0.85 / 1.31.0 | U2Net salient-object matting for cut-outs |
| NumPy / pandas | 2.5.4 / 3.0.6 | arrays, tables in the UI |
| pytest / Node.js | 9.1.1 / 24.18 | Python tests; JavaScript geometry tests |

No paid services and no cloud APIs: everything runs locally. Models are free downloads (~6 MB, ~40 MB, ~176 MB).

---

## 5. Algorithms

### 5.1 Image type classification (`opencv_detector.classify_image`)
Downscale to ≤256 px, quantise each channel to 16 levels.
- **Blank:** grey std < 3 and channel range < 25.
- **Flat illustration:** the 16 most frequent colours cover ≥ 80 % of all pixels **and** ≥ 68 % of the
  *foreground* (pixels more than ΔE 20 from the median border colour).
- **Photo:** everything else.

The foreground condition was added after a real product photo on a white background was misclassified as
flat (overall 0.88, foreground 0.60). Measured foreground coverage: real photos 0.35–0.60, real flat cartoons
0.745–0.90, synthetic fixtures ≥ 0.94.

### 5.2 Classical analysis of flat images
1. **Palette.** Up to 12 colours from quantised histogram bins, merging shades within ΔE 15 in CIELAB.
   The background is the palette colour that matches ≥ 50 % of the border.
2. **Region extraction.** Each pixel goes to its nearest palette colour (ΔE ≤ 25, else unassigned). Each colour
   is cleaned with a morphological opening and split into connected regions (≥ 0.3 % of the image).
3. **Shape recognition by hypothesis fitting.** For each region, a full ellipse, a minimum-area rectangle and a
   minimum enclosing triangle are fitted. A hypothesis is accepted if it contains the region and everything
   inside it is either the region or another (occluding) foreground colour; the fit score must be ≥ 0.93.
   - If fitting to the whole outline fails, it is repeated using only the outline points that touch the
     background, which "completes" shapes partly hidden by another shape. These are flagged `occluded` when
     more than 1.5 % of the completed shape is covered.
   - Pentagons and hexagons use polygon approximation (solidity ≥ 0.95), and stars use concavity.
   - Irregular regions become colour regions.
4. **Foreground blobs.** Connected multi-colour components (≥ 1 % of the image, ≥ 2 colours each covering
   ≥ 5 %) are reported with an outline polygon.
5. **Face-like heuristic (cartoon character).** A blob becomes `cartoon character` if two compact dark
   components (outer contour filled, fill ≥ 0.5) lie side by side in its upper half, at similar height and size.
   The rule deliberately does not use skin colour, which would be biased and would miss non-human characters.

**Colour naming:** each pixel is named by HSV rules (13 basic colours, including brown and beige), and the most
frequent name over the object's mask is used.

### 5.3 Learned detectors
- **YOLO `yolo11n-seg`** (COCO, 80 classes) runs on all images and provides boxes, the model's own confidence
  and mask outlines. On flat images, its results are flagged "verify (illustration)".
- **YOLOE `yoloe-26s-seg-pf`** (prompt-free, about 4,585 classes) runs on flat images only.
  - **Vocabulary restriction.** The classification head is a linear layer with one row per class, and scores
    are independent sigmoids. The app therefore keeps only the rows of a curated 24-class vocabulary
    (cartoon character, child, person, doll, fruits, animals, a few objects). The kept scores are unchanged.
  - **Why it is needed.** Used raw, the model labelled a cartoon girl's head "joker" (0.95) and an apple photo
    "fig" (0.92). "heart" and "balloon" were excluded because they fired on almost every round shape.
- **Boy/girl suggestions.** A second copy of YOLOE restricted to {boy, girl} produces a suggestion only, and
  it is never applied automatically. In evaluation it called **both** children in a boy-and-girl cartoon "girl".

### 5.4 Duplicate merging (`detection/merge.py`, `labels.py`)
Labels are *compatible* when they are equal, synonyms, a broad/specific pair (fruit ⊃ apple), or both
person-like.
- **Model vs model.**
  - Compatible labels and IoU ≥ 0.5 → one object, with the most specific label.
  - Compatible part (≥ 85 % contained) → the larger box is kept.
  - Unrelated labels with IoU ≥ 0.8 → the domain-appropriate model wins (YOLOE on illustrations, YOLO on photos).
- **OpenCV vs model.**
  - A foreground blob with IoU ≥ 0.5 is the same object. The model provides the name; OpenCV provides the
    outline and the full extent (e.g. an apple's stem and leaf).
  - A shape or region with IoU ≥ 0.8 is merged, but the model's label wins only at confidence ≥ 0.5.
  - Shapes inside a whole object stay as parts (`part_of`).
- **Nothing is discarded.** Merged detections are kept in `merged_from` and `alternative_labels`, and confidence
  always belongs to the kept label (never averaged).

### 5.5 Correction policy (`service.py`)
*Corrections* are manual objects, edited detections and deleted detections (kept as tombstones). Composite
layers are also protected.
- Detection runs automatically only on upload.
- **Keep corrections** (default): deletes only unprotected automatic objects, and skips new detections that
  overlap a protected object (IoU ≥ 0.5). This is why a deleted false positive never returns.
- **Replace everything**: raises `ConfirmationRequired` unless `confirm_replace=True`. The UI requires a
  confirmation tick that states how many corrections will be lost.
- The write happens inside `BEGIN IMMEDIATE`. Corrections are **re-read after detection**, so an edit made while
  detection was running is still protected, and any failure rolls everything back. A failed detection never
  deletes existing objects.

### 5.6 Instruction parsing and layout (`composer/parser.py`, `engine.py`)
- **Parsing.** Rule-based; nothing is sent to an external model. It extracts counts (words and digits),
  colours (an adjective unless it is the final noun, so "one orange" is the fruit), labels (singularised),
  background colour, layout words and, per item, positions, relations (left/right of, above, below, behind,
  in front of, next to), size words and depth.
- **Matching.** An exact label or a more specific catalog label ("boy" → "cartoon boy"); no synonyms. Colour
  matching is strict by default. Ranking: verified objects first, then not truncated by the image border, then
  not occluded, then confidence, then size.
- **Spatial layout.** A 3×3 anchor grid. Items without a position spread evenly, items sharing a cell split it,
  and relations are placed around the referenced layer and adjust the z-order. Everything is kept on the canvas.

### 5.7 Cut-outs (`composer/cutout.py`)
`auto` order:
1. PNG alpha,
2. an exact OpenCV contour (pixel-perfect on flat art),
3. **rembg/U2Net**, limited to the box and a slightly dilated known outline (used only if the model is already
   downloaded, so `auto` never triggers a download),
4. the stored outline,
5. colour keying (border-connected background),
6. GrabCut,
7. a plain rectangle.

### 5.8 Composite provenance
Each composite stores:
- `source_asset_ids` (image IDs and object IDs),
- `composition_metadata`: instruction, parse, canvas, layout, every layer's placement, mask method and a
  **snapshot of the source at composition time** (image filename + SHA-256; object label, colour, confidence,
  box, source, correction flag), and missing items,
- one `composite` object per layer with its exact placed box and outline.

Composites are excluded as sources by default, to avoid copies of copies.

---

## 6. Database schema (`data/catalog.db`)

```sql
images(id INTEGER PK, name, display_name, filename UNIQUE, width, height, created_at,
       detection_status DEFAULT 'pending', detection_message, detected_at, file_hash,
       is_composite DEFAULT 0, source_asset_ids JSON, composition_metadata JSON, dominant_colors JSON)

objects(id INTEGER PK, image_id → images(id) ON DELETE CASCADE, label, color,
        confidence REAL CHECK (NULL or 0..1), x, y, width CHECK > 0, height CHECK > 0,
        source, origin DEFAULT 'auto', is_edited, is_deleted, attributes JSON, created_at, updated_at)

detection_runs(id INTEGER PK, image_id → images(id) ON DELETE CASCADE, run_at, mode, status,
               message, num_objects, details JSON)
```

- **Indexes:** `objects(image_id)`, `objects(label)`.
- **Foreign keys:** enforced (`PRAGMA foreign_keys = ON` on every connection).
- **Status values:** `pending | success | empty | failed | skipped`.
- **Source values:** `yolo | yoloe | opencv_shape | opencv_color_region | opencv_foreground | manual | composite`.
- **`attributes` JSON keys:**
  - `polygon`, `polygon_source`, shape metrics (`shape`, `solidity`, `occluded`, …),
  - `original` and `edit_history` (corrections),
  - `merged_from`, `alternative_labels`, `suggested_labels`, `part_of`, `label_mapping`, `character_evidence`,
    `domain_warning`.
- **Schema stability:** the schema has not changed since the first version. All later features live in the JSON
  columns, so catalogs created earlier keep working (verified on the live catalog).

**Health check** (Dashboard, read-only) covers: `PRAGMA integrity_check`, `foreign_key_check`, missing image
files, orphan files, boxes outside their image, out-of-range confidences, and composites with deleted sources.

---

## 7. User interface

| Page | Main functions |
|---|---|
| Dashboard | Counts, engine status (YOLO / YOLOE / OpenCV / rembg), status and source breakdown, recent images, **database health check**, known limitations |
| Upload Images | Multi-file upload with options; synthetic fixtures in one click; per-file result with boxes and engine reports |
| Image Catalog | Search by name or object label, filters, pagination (24 per page), cached thumbnails, rename, delete with warnings (corrections lost, composites using the image) |
| Object Detection | Overlay with source colours, "hide parts of larger objects", object table with flags, engine reports, correction-safe re-run, run history |
| Manual Editor | Mouse box editor (select / move / resize / draw) synced with numeric inputs, colour and confidence editors, alternative labels and unverified suggestions (one-click apply), remove/restore |
| Composite Generator | Instruction, canvas and options → parsed request (positions/relations) → assets + missing report → layer table (x, y, scale, z, how placed) → preview → save |
| Metadata Viewer | All fields, object attributes, detection runs, provenance with source thumbnails, JSON export (one image or the whole catalog) |

---

## 8. Testing report

### 8.1 Automated tests
Command: `python -m pytest` (Windows PowerShell, venv). **Result of the final run: see §8.4.**

| File | Tests | What is verified |
|---|---|---|
| `test_detection.py` | 26 | colour naming; image-type classification incl. product photo regression; shapes, colour regions, overlap/occlusion vs ground truth (IoU > 0.9); blank; failure (no engine, corrupt file, YOLO unavailable) |
| `test_hybrid_detection.py` | 31 | label compatibility; every merge rule; confidence never averaged; character unification; gender only as suggestion; face heuristic (no false positives on 5 non-character fixtures); real YOLOE vocabulary restriction |
| `test_spatial_composition.py` | 22 | parsing of positions/relations/size/depth; left/centre/right, top/bottom, relations, z-order, on-canvas, saved placement metadata |
| `test_corrections.py` | 21 | add/edit/delete/restore; validation; restart persistence; keep-corrections re-run; replace-all confirmation |
| `test_composer.py` | 19 | example instruction end to end; real pixels; missing assets block saving; no colour substitution; ranking; overlap; cut-out methods |
| `test_ui_smoke.py` | 19 | all 7 pages with and without data; adding fixtures; manual add; confirmation gating; composite plan/save; missing report |
| `test_fixtures_and_yolo.py` | 14 | fixtures deterministic; YOLO on a real photo with real confidences and masks |
| `test_parser.py` | 13 | instruction parsing basics |
| `test_regressions.py` | 10 | correction made *during* detection survives; failure rolls back; provenance snapshot; deleted sources; health check; fast queries; old records |
| `test_review_fixes.py` | 11 | each defect from the independent code review (§8.3, items 12-20) |
| `test_catalog.py` | 8 | upload, duplicates, invalid files, EXIF rotation, alpha, delete cascade, long names |
| `test_bbox_editor_ui.py` | 6 | Manual Editor with simulated browser gestures: payload, move → form → save, gesture applied once, click-to-select, draw → add, clamping |
| `test_cutout_rembg.py` | 5 | rembg removes background; respects known outline; `auto` never downloads; exact contours preferred |
| `test_visualization.py` | 2 | label tags never overlap |
| `test_bbox_editor_js.py` | 1 (11 checks) | JavaScript hit-testing, move/resize clamping, drawing, run with Node.js |

Tests use temporary folders and never touch `data/`. Model-dependent and Node-dependent tests skip themselves if
their dependency is missing. On this machine none were skipped.

### 8.2 Evaluation on real images (manual measurements, not a benchmark)

| Image | Result |
|---|---|
| Real apple photo (user) | apple 0.85 (YOLO), class *photo* |
| Real clementine photo (user) | orange 0.92 (YOLO). Before the 5.1 fix it was misclassified as flat (8 meaningless regions); after the fix it is classified *photo* |
| `bus.jpg`, `zidane.jpg` (Ultralytics) | bus 0.90, persons 0.82–0.91 |
| Cartoon boy & girl (Wikimedia, CC BY-SA) | **2 × cartoon character** (0.71 / 0.63). Boy/girl suggestion "girl" for both (wrong for the boy) |
| Glossy clip-art apple (CC0) | fruit 0.60 (YOLO's "kite" 0.62 merged as an alternative) |
| Green apples (CC0) | apple 0.20 + "sports ball" 0.54 (wrong) |
| Cartoon orange character (CC BY-SA) | only foreground blobs, no semantic label |
| Line-art boy, 1913 poster | nothing detected (classified *photo*) |
| Synthetic fixtures (9) | every ground-truth item matched, IoU > 0.9, labels/colours correct |

Six real cartoons are **far too few to estimate accuracy**. They illustrate behaviour and failure modes only.

**Cut-out comparison:**
- *Real apple photo:* the YOLO outline left a white halo, GrabCut lost the stem area (14 s), rembg was clean (2 s).
- *Cartoon boy drawn with only a box:* GrabCut cut off the legs and made the eye whites transparent; rembg kept
  the figure (plus its ground shadow).

### 8.3 Defects found by testing and fixed
1. Overlapping label tags hid each other on the overlay → collision-aware placement.
2. Composites picked partly hidden or border-truncated objects → ranking prefers fully visible ones.
3. A raw YOLOE vocabulary gave absurd labels ("joker", "fig") → vocabulary restriction.
4. Merge order buried a whole apple inside its "circle" body part → foreground blobs are merged first and
   become whole objects.
5. Duplicate "kite" + "fruit" boxes on one apple → same-region rule with domain preference.
6. The mouse editor was not registered in fresh Streamlit runtimes → registration on every render (an
   identical definition is a no-op).
7. rembg model path changed between rembg versions → the location is asked from rembg itself.
8. A stalled 176 MB download → resumed with curl and verified against the published MD5.
9. A product photo on white was classified as an illustration → foreground coverage criterion (§5.1).
10. A correction made during a running detection could be deleted (stale list) → re-read inside a
    write-locked transaction (§5.5).
11. The image status update ran outside the detection transaction → moved inside (full rollback on failure).

Found by an independent final code review (each with a regression test in `test_review_fixes.py`; the two UI
tests were additionally checked to fail when their fix is disabled):

12. **(High)** After *Apply/Use* changed a label, the edit form still held the old label and confidence, and the next
    *Save changes* silently reverted the correction → form state is cleared after every saved change.
13. A detection that the user had *moved* re-appeared at its original place on re-detection → edited objects also
    block their original box.
14. The mouse-editor canvas jumped back one gesture → rerun after applying a gesture.
15. Clearing a cell in the composer's layer table crashed the page (NaN) → cleared cells keep their value.
16. Cut-outs of large photos used ~0.9 GB of temporary memory → no float copies; opaque images use a view.
17. Pages showed a stale image status right after a detection run → the selector returns the fresh record.
18. 16-bit greyscale PNGs were stored almost entirely white → scaled to 8 bit.
19. Misleading "No object labelled 'X'" when the only X was already used earlier in the request → accurate reason.
20. Deleting an image whose file is locked (Windows) left a half-deleted state; `%`/`_` in search were wildcards →
    file removed first (nothing deleted on failure); literal search.

Also improved: cached thumbnails, single-query object counts, pagination (24 per page), display-sized previews of
large photos, spinners for detection/planning/saving, friendly error messages instead of tracebacks.

### 8.4 Final automated test run
`python -m pytest` on 2026-10-14 (Windows 11, Python 3.12.10, existing `.venv`): **208 passed, 0 skipped, 0 failed**
(~95 s). Additionally verified on this machine:
- **Windows PowerShell:** `.\.venv\Scripts\Activate.ps1` works under the CurrentUser `RemoteSigned` policy and
  all packages import.
- **Real-data render:** all 7 pages render without errors against a *copy* of the real catalog (including a
  3836×2557 photo and an earlier composite).
- **Live catalog:** the health check reports "No problems found".

### 8.5 Not verified automatically
Real pointer interaction with the mouse editor, touch input, accuracy on a larger set of real cartoons, and
visual cut-out quality on your own images. See `MANUAL_TESTING.md`.

---

## 9. Limitations

- **Cartoon semantics.**
  - Line art, vintage drawings and stylised characters are often missed.
  - Labels on illustrations must be verified ("verify (illustration)" flag).
  - The face heuristic needs two visible, dark, compact eyes.
- **Gender.** Appearance-based boy/girl is unreliable and ethically sensitive, so it is offered only as a
  suggestion that the user must confirm.
- **Photos.** Only the 80 COCO classes are detected automatically; OpenCV analysis is skipped on photos.
- **Colour names** are HSV rules and can be shifted by lighting.
- **Cut-outs.**
  - rembg can keep ground shadows.
  - Fine structures (hair, fur) and heavily overlapping objects remain hard.
  - Occluded shapes are cut out with only their visible part.
- **Instructions.** Rule-based: free-form descriptions and exact distances are not understood.
- **Mouse editor.** The geometry and the Python side are tested, but real pointer interaction in a browser
  must be checked by hand. Touch devices are untested.
- **Scale.** Designed for a single local user. SQLite with one connection per operation is fine for hundreds
  of images; detection runs on the CPU (about 1–4 s per image after a ~20 s first model load).

---

## 10. Ethics, licensing and data

- **No generative AI** is used for images. Composites contain only catalog pixels, and missing objects are
  reported, never invented.
- **Confidence values are never fabricated.** Heuristics store NULL.
- **Gender** is never assigned automatically (see §5.3).
- **Local data.** All data stays local: there are no cloud services, and the Ultralytics settings and rembg
  models are kept inside the project folder.
- **Third-party test images** in `test_data/real_cartoons/` are CC0 / CC BY-SA 4.0 / public domain, with
  attribution in `ATTRIBUTION.md`. They are excluded from git.
- **Model licences.** Ultralytics models are AGPL-3.0 (fine for coursework; commercial use needs a licence),
  and U2Net (rembg) is Apache-2.0.

---

## 11. Future work
1. A proper evaluation set (≥ 100 labelled cartoons) to measure precision/recall and tune thresholds.
2. A text-prompted open-vocabulary model (YOLOE with a text encoder, or a CLIP re-ranker) for arbitrary user
   labels.
3. Interactive segmentation (click-to-mask) for difficult cut-outs.
4. Undo history for corrections; multi-user locking if ever deployed as a server.

---

## 12. How to run
See `README.md` → *Quick start*. In short:

```powershell
cd C:\Users\kusha\Image-Catalog-System
.\.venv\Scripts\Activate.ps1
python -m pytest
streamlit run app.py
```
