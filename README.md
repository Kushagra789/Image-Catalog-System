# Image Catalog System

A local web application for cataloguing images, detecting objects in photographs **and cartoons**, correcting
detections by hand, and building **composite images from natural-language instructions** using only real
objects cut out of catalog images.

| Document | Contents |
|---|---|
| `README.md` (this file) | installation, running, features, architecture, schema, testing summary, limitations |
| `PROJECT_REPORT.md` | full project report: requirements traceability, algorithms, testing report, evaluation, ethics |
| `MANUAL_TESTING.md` | checklist of what must still be verified by hand in a browser |
| `PROGRESS.md` | final status and how to resume |
| `test_data/real_cartoons/ATTRIBUTION.md` | licences of the third-party test images |

---

## 1. Installation (Windows PowerShell)

Requirements: Windows 10/11, **Python 3.12** (3.14 is not used: PyTorch/Ultralytics support for it was uncertain),
internet access once (for packages and model downloads), ~2 GB of disk space. No WSL, no paid services.

```powershell
cd C:\Users\kusha\Image-Catalog-System

# 1. Create and activate the virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
#    If activation is blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 2. Install dependencies (PyTorch CPU comes with ultralytics, ~1 GB)
python -m pip install --upgrade pip
pip install -r requirements-lock.txt     # exact versions verified with the test suite
#   (or: pip install -r requirements.txt  for the minimum-version list)
```

The project folder already contains an installed `.venv` and the downloaded models, so on this machine only
step 1's `Activate.ps1` line is needed.

**Models** are stored in `models\` and downloaded automatically on first use if missing:

| Model | Size | Used for |
|---|---|---|
| `yolo11n-seg.pt` | ~6 MB | COCO objects with masks (photos; flagged on illustrations) |
| `yoloe-26s-seg-pf.pt` | ~40 MB | illustration vocabulary (cartoons, clip-art) |
| `rembg\models\u2net\u2net.onnx` | ~176 MB | learned background removal for cut-outs |

Without them, or without `ultralytics` / `rembg`, the app still runs with OpenCV analysis, classical cut-outs
and manual labelling. The Dashboard shows what is unavailable and why. The `auto` cut-out mode never triggers
the 176 MB download.

## 2. Running

```powershell
cd C:\Users\kusha\Image-Catalog-System
.\.venv\Scripts\Activate.ps1
streamlit run app.py            # opens http://localhost:8501 ; stop with Ctrl+C
python -m pytest                # full automated test suite (~1.5 min)
```

The server only listens on `localhost` (see `.streamlit/config.toml`). The first detection after start loads
the models (~20 s); later images take ~1–4 s each on CPU.

| Environment variable | Effect |
|---|---|
| `IMAGECAT_DATA_DIR` | folder for `catalog.db` and `images\` (default `.\data`) |
| `IMAGECAT_DISABLE_YOLO=1` / `IMAGECAT_DISABLE_ILLUSTRATION_MODEL=1` | run without YOLO / YOLOE |
| `IMAGECAT_YOLO_MODEL`, `IMAGECAT_ILLUSTRATION_MODEL` | other Ultralytics weights (e.g. `yolo11s-seg.pt`, `yoloe-26n-seg-pf.pt`) |
| `IMAGECAT_REMBG_MODEL` | other rembg model (e.g. `isnet-anime`; separate download) |

```powershell
# Back up the catalog before risky work
Copy-Item data\catalog.db "data\backups\catalog_$(Get-Date -Format yyyyMMdd_HHmmss).db"
# Export the synthetic fixtures and ground truth to test_data\synthetic\
python scripts\generate_test_images.py
```

## 3. Features

**Image catalog.** Upload (multi-file), preview, search by name or object label, filter, paginate, rename,
delete with warnings (manual corrections that would be lost, composites that used the image). Files are stored
in `data\images\`, metadata in SQLite. EXIF rotation is applied, duplicates are detected by SHA-256, and 16-bit
images are scaled correctly.

**Automatic object detection** (on upload), a hybrid of three engines:
- **YOLO** `yolo11n-seg`: 80 COCO classes, the model's own confidence, segmentation outlines.
- **YOLOE** `yoloe-26s-seg-pf`: runs on illustrations only, restricted to a curated vocabulary
  (cartoon character, child, fruits, animals, …).
- **OpenCV**: on flat images only, shapes (also partly hidden ones), colour regions, multi-colour foreground
  blobs, and a face-like heuristic for cartoon characters.

Detections are merged across engines; the other labels are kept as alternatives. Each object stores its label,
colour, confidence (**only when a model produced one, never invented**), box (x, y, width, height), source and
outline. Results are always reported as `success`, `empty` (e.g. a blank image) or `failed` (with the reason).

**Cartoon characters** are labelled `cartoon character`. **Boy/girl is never assigned automatically**: the
model's guess is shown only as an unverified suggestion, because it is unreliable (in testing it called a cartoon
boy "girl").

**Manual corrections.**
- Add, edit, remove and restore objects: label, colour, confidence and box.
- Boxes can be **drawn, moved and resized with the mouse**, or typed exactly.
- Every change is saved to SQLite immediately and survives restarts.
- Corrections are **authoritative**. Re-running detection keeps them by default; replacing them requires an
  explicit confirmation.

**Composite generation** from instructions such as
*"Place an apple on the left, an orange on the right, and a cartoon boy in the center."*
- Understands counts, colours, objects, background colour, layouts, positions, relations
  (left/right of, above, below, behind, in front of, next to), size and depth.
- Uses **only real catalog objects**. Missing ones are reported with suggestions and never replaced by
  look-alikes or AI-generated content.
- Cut-outs use alpha / exact outline / rembg / colour key / GrabCut. Layers can be positioned, scaled and
  re-stacked.
- Results are saved as new catalog images with source image and object IDs, a snapshot of each source, and the
  full composition metadata.

**Database health check** (Dashboard, read-only): integrity, foreign keys, missing or orphaned files, invalid
boxes, composites whose sources were deleted.

### Pages

| Page | Purpose |
|---|---|
| Dashboard | Counts, engine and model status, status/source/label summaries, recent images, health check, known limitations |
| Upload Images | Upload with detection options; one-click synthetic test images |
| Image Catalog | Search/filter/paginate, rename, delete (with warnings) |
| Object Detection | Annotated image, hide parts of larger objects, object table, engine reports, correction-safe re-run, run history |
| Manual Editor | Mouse box editor + numeric inputs, colour/confidence editing, alternative labels and suggestions, remove/restore |
| Composite Generator | Instruction → understood request → assets and missing report → layer table → preview → save |
| Metadata Viewer | All fields, attributes, detection runs, provenance, JSON export |

## 4. Architecture and workflow

```
Streamlit UI (app.py, imagecat/ui/*)  ──►  CatalogService (service.py)  ──►  Database (db.py) ─► data/catalog.db
        │                                   ├─ ImageCatalog (catalog.py) ─► data/images/
        │                                   └─ DetectionEngine (detection/engine.py)
        │                                        ├─ OpenCVDetector · YoloDetector · IllustrationDetector
        │                                        └─ merge.py + labels.py
        └──────────────►  Composer (composer/engine.py) ─ parser.py · cutout.py
```

1. **Upload:** validate → EXIF-rotate → hash → store file → insert record → detect → save objects, the run log
   and the status in one transaction.
2. **Detect:** image type (blank / flat / photo) → OpenCV (flat) → YOLO → YOLOE (flat) → unify character
   labels → merge duplicates → attach suggestions.
3. **Correct:** validated edits are stored immediately; edited detections keep their original values and an
   edit history.
4. **Re-detect:** write-locked transaction. Corrections are re-read *after* detection, and any failure rolls
   everything back.
5. **Compose:** parse → match (no substitution) → report missing → cut out → lay out → adjust → save with
   provenance.

```
app.py                       Streamlit entry point (st.navigation)
imagecat/
  config.py                  paths/settings; keeps Ultralytics + rembg files inside the project
  models.py  db.py           dataclasses; SQLite schema and queries
  catalog.py  service.py     file ingest/delete; detection orchestration, correction policy, health check
  colors.py  imaging.py      HSV colour naming; image I/O
  detection/                 engine, opencv_detector, yolo_detector, yoloe_detector, merge, labels
  composer/                  parser, engine (matching, layouts, rendering, saving), cutout
  fixtures.py                deterministic synthetic test images + exact ground truth
  visualization.py           box/label drawing
  ui/                        one module per page; bbox_editor.py + static/bbox_editor.js (mouse editor)
scripts/generate_test_images.py
tests/                       pytest suite (17 test modules, incl. a real-browser test)
test_data/real_cartoons/     6 openly licensed real test images (not in git)
```

## 5. Technologies and algorithms

| Technology | Version | Role |
|---|---|---|
| Python | 3.12.10 | language |
| Streamlit | 1.65.0 | UI; Components v2 for the mouse editor |
| SQLite | 3.49 | metadata (foreign keys, CHECK constraints, transactions) |
| OpenCV | 5.0 | segmentation, contours, shape fitting, GrabCut |
| Pillow · NumPy · pandas | 12.3 · 2.5 · 3.0 | image I/O and rendering; arrays; tables |
| Ultralytics (+ PyTorch CPU) | 8.4.176 (2.14) | YOLO `yolo11n-seg`, YOLOE `yoloe-26s-seg-pf` |
| rembg (+ onnxruntime) | 2.0.85 (1.31) | U2Net background removal |
| pytest · Node.js | 9.1 · 24 | tests (Python; JavaScript geometry) |

Key algorithms (details in `PROJECT_REPORT.md` §5):
- **Flat vs photo:** top-16 colour coverage overall (≥ 80 %) *and* on the foreground (≥ 68 %).
- **Palette segmentation** in CIELAB.
- **Shape recognition** by fitting complete ellipse, rectangle and triangle hypotheses, which recognises
  occluded shapes.
- **Face-like heuristic:** two compact dark blobs side by side in the upper half of a figure; no skin colour used.
- **Vocabulary restriction:** YOLOE's per-class head rows are kept only for a curated subset.
- **Label-aware de-duplication:** IoU / containment combined with synonym, hypernym and person-like groups.
- **Correction policy:** tombstones, protection of edited boxes at their original place, write-locked
  re-read on re-detection.
- **Rule-based instruction parser** with a 3×3 spatial layout.
- **Asset ranking:** verified, then complete, then unoccluded, then confidence, then size.
- **Cut-out cascade:** alpha → exact contour → rembg (constrained) → outline → key → GrabCut.

## 6. Database schema (`data/catalog.db`)

| Table | Columns |
|---|---|
| `images` | `id` PK, `name`, `display_name`, `filename` (unique), `width`, `height`, `created_at`, `detection_status` (`pending/success/empty/failed/skipped`), `detection_message`, `detected_at`, `file_hash` (SHA-256), `is_composite`, `source_asset_ids` JSON, `composition_metadata` JSON, `dominant_colors` JSON |
| `objects` | `id` PK, `image_id` → images ON DELETE CASCADE, `label`, `color`, `confidence` (NULL or 0–1, CHECK), `x`, `y`, `width` > 0, `height` > 0, `source` (`yolo/yoloe/opencv_shape/opencv_color_region/opencv_foreground/manual/composite`), `origin` (`auto/manual`), `is_edited`, `is_deleted` (tombstone), `attributes` JSON, `created_at`, `updated_at` |
| `detection_runs` | `id` PK, `image_id` → images ON DELETE CASCADE, `run_at`, `mode`, `status`, `message`, `num_objects`, `details` JSON (per-engine reports) |

`objects.attributes` JSON keys:
- `polygon` (outline), shape metrics, `occluded`,
- `original` and `edit_history` (for corrections),
- `merged_from`, `alternative_labels`, `suggested_labels`, `part_of`, `label_mapping`, `character_evidence`,
  `domain_warning`.

The table structure has not changed since the first version, so earlier catalogs work unchanged.

## 7. Testing summary

`python -m pytest` → **218 passed, 0 skipped** (2026-10-16, Windows 11, Python 3.12.10). Tests use temporary
folders and never touch `data\`. Model-, Node- and Playwright-dependent tests skip themselves when their
dependency is missing.

| Area | Files |
|---|---|
| Catalog (upload, duplicates, invalid/16-bit files, EXIF, delete) | `test_catalog.py` |
| Detection on fixtures and real photos; failure/empty; flat-vs-photo | `test_detection.py`, `test_fixtures_and_yolo.py` |
| Hybrid cartoon detection, merging, gender-only-as-suggestion | `test_hybrid_detection.py` |
| Corrections, persistence, re-run policy, concurrency, rollback | `test_corrections.py`, `test_regressions.py` |
| Composites, provenance, missing assets, spatial instructions, cut-outs | `test_composer.py`, `test_spatial_composition.py`, `test_cutout_rembg.py`, `test_parser.py` |
| All 7 pages, mouse editor (simulated gestures + JS geometry with Node) | `test_ui_smoke.py`, `test_bbox_editor_ui.py`, `test_bbox_editor_js.py`, `test_visualization.py` |
| Fixes from the final independent code review | `test_review_fixes.py` |
| Fixes from the real-browser acceptance run; **real Chromium** end-to-end editor test | `test_browser_run_fixes.py`, `test_browser_e2e.py` |

**Real-browser acceptance run (2026-10-16).** The manual checklist was executed with Playwright against a separate
app instance using a copy of the catalog: **37 checks passed** after two defects were found and fixed. Results and
evidence are in `MANUAL_TESTING.md`; screenshots are in the git-ignored `test_evidence\` folder.

Optional browser tooling (used for that run, ~0.7 GB of browser files):

```powershell
pip install playwright
$env:PLAYWRIGHT_BROWSERS_PATH = "$PWD\.playwright"     # keep the browsers inside the project (git-ignored)
python -m playwright install chromium
python -m pytest tests\test_browser_e2e.py               # real-Chromium editor test
```

The full testing report, real-image evaluation and list of defects found and fixed are in `PROJECT_REPORT.md` §8.
**The few items that still need a person** (touch input, your own cartoons, layer-table cell editing) are listed
at the end of `MANUAL_TESTING.md`.

## 8. Limitations

- **Cartoons:** line art, vintage drawings and stylised characters are often missed. Labels on illustrations
  must be verified. 6 real cartoons are too few to measure accuracy.
- **Boy/girl** from appearance is unreliable and is never applied automatically.
- **Photos:** only the 80 COCO classes are detected automatically.
- **Colour names** are rule-based and can be shifted by lighting.
- **Cut-outs:** masks are limited to the detected box, so a slightly tight box clips edges (enlarge the box in
  the Manual Editor); rembg may keep ground shadows; hair, fur and heavy overlaps remain hard.
- **Instructions** are rule-based: no free-form descriptions or exact distances.
- **Mouse editor:** verified with real mouse events in Chromium; touch input is untested.
- **Single-user local application** on CPU.
