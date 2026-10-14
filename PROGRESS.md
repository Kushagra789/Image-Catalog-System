# Progress Log: Image Catalog System

_Last updated: 2026-10-14. **Final status: feature-complete; 208/208 automated tests passing; ready for final
manual verification (see `MANUAL_TESTING.md`).**_

Documentation: `README.md` (install, run, architecture, schema, testing summary) · `PROJECT_REPORT.md` (full report)
· `MANUAL_TESTING.md` (browser checklist) · this file (status and history).

---

## 1. Commands

```powershell
cd C:\Users\kusha\Image-Catalog-System
.\.venv\Scripts\Activate.ps1                 # Python 3.12.10 venv (blocked? Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
python -m pytest                             # expect: 208 passed (~1.5 min)
streamlit run app.py                         # http://localhost:8501 ; Ctrl+C to stop
```

```powershell
python -m pytest -k compos                   # subset
python -m pytest -rs                         # show reasons for skips (none expected here)
python scripts\generate_test_images.py       # export synthetic fixtures + ground truth
pip install -r requirements-lock.txt         # recreate the exact tested environment
Copy-Item data\catalog.db "data\backups\catalog_$(Get-Date -Format yyyyMMdd_HHmmss).db"   # backup
```

**After this update:** the Streamlit server you started at 18:27 on 2026-10-14 still runs the *previous* code
(its cached service objects predate today's fixes). Stop it with Ctrl+C in its terminal and run
`streamlit run app.py` again before manual testing.

---

## 2. Completed features

### Original requirements: all complete
- [x] Image catalog: upload, preview, list/search/filter/paginate, rename, delete (with warnings); files in
      `data\images\`, metadata in SQLite.
- [x] Automatic detection on upload; labels, colours, real confidences only, boxes, source; success/empty/failed reporting.
- [x] Manual corrections: add/edit/remove/restore; persistent; authoritative; confirmation before replacement.
- [x] Composite generation from natural language using real catalog objects only; missing assets reported;
      saved with source IDs, per-layer source snapshot and composition metadata.
- [x] Test data: 9 deterministic synthetic fixtures with ground truth plus 6 licensed real cartoon images.
- [x] Seven Streamlit pages.

### Improvements (sessions 3–4)
- [x] Hybrid cartoon detection (YOLO + vocabulary-restricted YOLOE + OpenCV face-like heuristic); boy/girl only as
      an unverified suggestion.
- [x] Label-aware duplicate merging; alternatives kept; parts of larger objects marked and hideable.
- [x] Spatial instructions (positions, relations, size, depth).
- [x] Mouse bounding-box editor (select, move, resize, draw) synced with exact numeric inputs.
- [x] rembg (U2Net) cut-outs, constrained by box and outline.
- [x] Database health check (read-only) on the Dashboard.
- [x] Correction safety: re-detection runs in a write-locked transaction that re-reads corrections; moved
      detections are protected at their original place; form state is cleared after saves.
- [x] Performance: cached thumbnails, single-query counts/search, pagination, display-sized previews, low-memory
      cut-outs.
- [x] Error handling and spinners on all long operations.
- [x] Photo-vs-illustration fix for product photos on white backgrounds.

---

## 3. Test results

**2026-10-14: `python -m pytest` → 208 passed, 0 skipped, 0 failed** (Windows 11, Python 3.12.10, existing `.venv`).

| File | Tests | File | Tests |
|---|---|---|---|
| test_hybrid_detection.py | 31 | test_parser.py | 13 |
| test_detection.py | 26 | test_review_fixes.py | 11 |
| test_spatial_composition.py | 22 | test_regressions.py | 10 |
| test_corrections.py | 21 | test_catalog.py | 8 |
| test_composer.py | 19 | test_bbox_editor_ui.py | 6 |
| test_ui_smoke.py | 19 | test_cutout_rembg.py | 5 |
| test_fixtures_and_yolo.py | 14 | test_visualization.py | 2 |
| | | test_bbox_editor_js.py | 1 (11 JS checks) |

**Also verified:**
- `Activate.ps1` and all imports work in a fresh PowerShell under the `RemoteSigned` policy.
- All 7 pages render without errors on a copy of the real catalog.
- The live catalog passes the health check.

**History:** 117 tests (session 1) → 121 → 186 (session 3) → 208 (session 4).

---

## 4. Session 4 (2026-10-14): what changed

- **Bugs found on real data:** a product photo on a white background was treated as an illustration →
  foreground-coverage rule.
- **Bugs found by me:** a correction made *during* a running detection could be deleted; the status update ran
  outside the transaction.
- **Independent code review:** 9 confirmed defects plus 3 uncertain ones. All fixed, with regression tests. The
  most serious was a relabel being silently reverted by the next "Save changes".
- **New:** health check, composite source snapshots, delete warnings, pagination, caching, spinners, error handling.
- **Docs:** README rewritten; new `PROJECT_REPORT.md` and `MANUAL_TESTING.md`; this file.
- **No schema change; no data deleted.** Backups before the work: `data\backups\catalog_20261014_182725.db`,
  `catalog_20261014_184352_before_finalize.db`.

---

## 5. Remaining limitations / open items

**Needs manual verification:** see `MANUAL_TESTING.md`. The key items are:
- [ ] the mouse editor with a real pointer,
- [ ] real cartoon and photo behaviour in the UI,
- [ ] composite quality.

**Known limitations:**
- Cartoon semantics are limited (line art and stylised characters are missed); gender is never inferred
  automatically.
- Only the 80 COCO classes are detected on photos.
- Colour names are rule-based.
- rembg may keep shadows.
- The instruction parser is rule-based.
- Single-user, CPU-only.

**Your data:**
- Image #2 (clementine) still holds 8 OpenCV regions from before the photo fix. It has no corrections, so
  Object Detection → *Keep my corrections* → Run detection refreshes it safely. I did not change your data
  automatically.
- The existing apple record keeps its ~200-character internal name; the 80-character cap applies to new
  uploads only.

**Open decisions (yours):**
- [ ] Delete `C:\Users\kusha\AppData\Roaming\Ultralytics\settings.json` (created outside the project by an
      early version check in session 1; harmless).
- [ ] `git init` and a first commit (`.gitignore` already excludes `data\`, `models\`, `.venv\` and third-party
      test images).

---

## 6. Saved state (2026-10-14)

| Item | State |
|---|---|
| Source code | `app.py`, `imagecat\`, `tests\` (15 files), `scripts\` |
| Catalog | `data\catalog.db`: 3 images, 12 objects; integrity ok; health check clean |
| Images | `data\images\` (3 files, all present) |
| Backups | `data\backups\` (3 database backups) |
| Models | `models\yolo11n-seg.pt`, `models\yoloe-26s-seg-pf.pt`, `models\rembg\models\u2net\u2net.onnx` (MD5 verified) |
| Environment | `.venv\` (Python 3.12.10); exact versions in `requirements-lock.txt` |
