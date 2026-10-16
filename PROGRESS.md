# Progress Log: Image Catalog System

_Last updated: 2026-10-16. **Status: feature-complete; 218/218 automated tests passing; 37 real-browser acceptance
checks passed (Playwright). Four small items remain for a person (see §5).**_

Documentation: `README.md` (install, run, architecture, schema, testing summary) · `PROJECT_REPORT.md` (full report)
· `MANUAL_TESTING.md` (checklist with actual results) · this file (status and history).

---

## 1. Commands

```powershell
cd Image-Catalog-System                      # the project folder
.\.venv\Scripts\Activate.ps1                 # Python 3.12.10 venv (blocked? Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
python -m pytest                             # expect: 218 passed (~1.5–2 min)
streamlit run app.py                         # http://localhost:8501 ; Ctrl+C to stop
```

```powershell
python -m pytest -k compos                   # subset
python -m pytest -rs                         # show reasons for skips (none expected here)
python -m pytest tests\test_browser_e2e.py   # real-Chromium editor test (needs Playwright; README §7)
python scripts\generate_test_images.py       # export synthetic fixtures + ground truth
pip install -r requirements-lock.txt         # recreate the exact tested environment
Copy-Item data\catalog.db "data\backups\catalog_$(Get-Date -Format yyyyMMdd_HHmmss).db"   # backup
```

**Restart the app before using it:** the copy started at 21:00 on 2026-10-16 runs code from before the two fixes
of 21:29 and 22:05 (§4). Press Ctrl+C in its terminal, then run `streamlit run app.py` again.

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

### Improvements (sessions 3–5)
- [x] Hybrid cartoon detection (YOLO + vocabulary-restricted YOLOE + OpenCV face-like heuristic); boy/girl only as
      an unverified suggestion.
- [x] Label-aware duplicate merging; alternatives kept; parts of larger objects marked and hideable.
- [x] Spatial instructions (positions, relations, size, depth).
- [x] Mouse bounding-box editor (select, move, resize, draw) synced with exact numeric inputs; **verified with real
      mouse events in Chromium**.
- [x] rembg (U2Net) cut-outs, constrained by box and outline.
- [x] Database health check (read-only) on the Dashboard.
- [x] Correction safety:
      - re-detection runs in a write-locked transaction that re-reads corrections;
      - moved detections are protected at their original place;
      - edit-form and selection state are safe against stale browser widget values.
- [x] Performance: cached thumbnails, single-query counts/search, pagination, display-sized previews, low-memory
      cut-outs.
- [x] Error handling and spinners on all long operations.
- [x] Photo-vs-illustration fix for product photos on white backgrounds.

---

## 3. Test results

**Automated (2026-10-16): `python -m pytest` → 218 passed, 0 skipped, 0 failed** (Windows 11, Python 3.12.10,
existing `.venv`, Playwright 1.64 installed).

| File | Tests | File | Tests |
|---|---|---|---|
| test_hybrid_detection.py | 31 | test_review_fixes.py | 11 |
| test_detection.py | 26 | test_regressions.py | 10 |
| test_spatial_composition.py | 22 | test_catalog.py | 8 |
| test_corrections.py | 21 | test_bbox_editor_ui.py | 7 |
| test_composer.py | 19 | test_browser_run_fixes.py | 7 |
| test_ui_smoke.py | 19 | test_cutout_rembg.py | 5 |
| test_fixtures_and_yolo.py | 14 | test_browser_e2e.py | 2 |
| test_parser.py | 13 | test_visualization.py | 2 |
| | | test_bbox_editor_js.py | 1 (11 JS checks) |

**Real-browser acceptance run (2026-10-16), Playwright + Chromium, on a copy of the catalog: 37 checks PASS**
- **Covered:** A1–A9, B1–B3, C1–C4, D1, E1–E3, E5–E6, F1–F4, detection on every fixture, and the apple + orange
  composite from your real photos.
- **Full results:** every result with its evidence is in `MANUAL_TESTING.md`.
- **Evidence:** screenshots, composites and `results.json` are in `test_evidence\2026-10-16_playwright\`
  (git-ignored).

**History:** 117 tests (session 1) → 121 → 186 (session 3) → 208 (session 4) → 218 (session 5).

---

## 4. Session 5 (2026-10-16): what changed

- **Manual results corrected.** Results A1–A5 reported in chat earlier that day were withdrawn: the database
  showed the steps had not reached the app. The Chrome extension could not be authorised, so the checklist was
  run with **Playwright** (installed with your approval). The browser files take 719 MB in `.playwright\`
  (git-ignored), not the ~150 MB first estimated.
- **Defect 21 (fixed with approval, 21:29).** On illustrations, model labels with confidence ≥ 0.5 replaced
  correct OpenCV shapes and colour swatches ("kite" 0.57 for the orange swatch, "orange" 0.59 for the orange
  ellipse). Threshold raised to 0.65 in `detection/merge.py`; 7 regression tests added.
- **Defect 22 (fixed with approval, 22:05; High).**
  - *Symptom:* in a real browser, a label changed with Use/Apply was reverted by the next Save, and a drag made
    right after a canvas selection could be dropped.
  - *Cause:* stale widget values held by the browser.
  - *Fix:* revision-based form keys plus a non-widget selection state in `ui/editor.py`.
  - *Tests:* 3 regression tests, including a real-Chromium end-to-end test.
- **Docs and requirements:** `MANUAL_TESTING.md` now records actual results; `PROJECT_REPORT.md` §8.3–8.5 and §9,
  `README.md` §7–8, `requirements.txt` (Playwright as optional) and `requirements-lock.txt` were updated.
- **Data:** your live catalog was not modified; it is byte-identical to
  `data\backups\catalog_20261016_210223_before_manual_tests.db`.

---

## 5. What remains

**For a person (details at the end of `MANUAL_TESTING.md`):**
- [ ] A10: mouse editor on a touch device (optional).
- [ ] C5: upload a few of your own cartoons and note the results.
- [ ] E4: edit x/y/scale/z in the Composite Generator's layer table, including clearing one cell.

**Your decisions:**
- [ ] **Image #2 (clementine).** It still holds 8 OpenCV regions from before the photo fix, and its orange
      cut-out is noticeably worse than a refreshed record's. It has no corrections, so Object Detection →
      *Keep my corrections* → Run detection refreshes it safely. Not done.
- [ ] **Stray settings file.** Delete `%APPDATA%\Ultralytics\settings.json` (created
      outside the project in session 1; harmless).
- [ ] **Connect GitHub and push.** Deferred; ask first (§7).
- [ ] **Remove the browser files (optional).** To free 719 MB after testing: `pip uninstall playwright` and delete
      `.playwright\`. `tests/test_browser_e2e.py` then skips itself.

**Known limitations** (`README.md` §8):
- Cartoon semantics are limited; gender is never inferred automatically.
- Only COCO classes are detected on photos.
- Cut-outs are clipped to the detected box (slightly tight boxes clip edges).
- Colour names are rule-based; the instruction parser is rule-based.
- Touch input is untested; single-user, CPU-only.

---

## 6. Saved state (2026-10-16)

| Item | State |
|---|---|
| Source code | `app.py`, `imagecat\`, `tests\` (17 test modules + conftest.py), `scripts\` |
| Catalog | `data\catalog.db`: 3 images, 12 objects, 0 corrections; integrity ok; unchanged by testing |
| Images | `data\images\` (3 files, all present) |
| Backups | `data\backups\` (4 database backups, newest `catalog_20261016_210223_before_manual_tests.db`) |
| Models | `models\yolo11n-seg.pt`, `models\yoloe-26s-seg-pf.pt`, `models\rembg\models\u2net\u2net.onnx` (MD5 verified) |
| Test tooling | `.playwright\` (Chromium for Playwright, 719 MB, git-ignored) |
| Test evidence | `test_evidence\2026-10-16_playwright\` (git-ignored; contains copies of your photos) |
| Environment | `.venv\` (Python 3.12.10); exact versions in `requirements-lock.txt` |

---

## 7. Git status (2026-10-16)

| Item | Status |
|---|---|
| Repository | local only (`.git` in the project folder), branch `main`; **no remote, nothing pushed** |
| Commits on `main` | `f4f9df9` (complete project), `4ab7191` (progress and Git status), `57a4ce7` (browser-found fixes and acceptance results), plus the commit that records this section |
| Author identity | `Kushagra Pandey <259154468+Kushagra789@users.noreply.github.com>` (GitHub no-reply address), set **for this repository only** (`git config --local`); the global `.gitconfig` is unchanged |
| History rewrite | On 2026-10-16, with your approval, the 3 commits were rewritten so that the personal Gmail address no longer appears (author/committer fields and one line of `PROGRESS.md`). Dates, messages and all other content are unchanged; the old IDs were `f6ee8ff`, `b23ebe3`, `a9b221e` |
| Local-only backups of the old history | branch `backup/before-email-rewrite` and `refs/original/refs/heads/main`. They still contain the Gmail address. `git push -u origin main` does **not** upload them; never use `git push --all` or `--mirror`. Delete them once the new history is confirmed (only with your permission) |
| Ignored (never committed) | `data\` (catalog, images, backups), `models\`, `.venv\`, `.playwright\`, `test_data\`, `test_evidence\`, caches |
| Licence | **AGPL-3.0-only** (`LICENSE`, official GNU text), chosen for public release because Ultralytics is AGPL-3.0; obligations and open questions are in `README.md` §9 and `PROJECT_REPORT.md` §10 |
| Visibility decision | **public** repository under `Kushagra789` (decided 2026-10-16); not yet created or pushed |

**Publish on GitHub** (account `Kushagra789`, **public** repository; nothing below has been run):

1. On github.com:
   - **Settings → Emails:** tick **Keep my email addresses private** and
     **Block command line pushes that expose my email**.
   - **New repository:** name `Image-Catalog-System`, **Public**, and **no** README, .gitignore or licence
     (all three are already in the project).
2. In PowerShell:

   ```powershell
   cd Image-Catalog-System
   git status                                   # must say: nothing to commit, working tree clean
   git remote add origin https://github.com/Kushagra789/Image-Catalog-System.git
   git push -u origin main                      # Git Credential Manager opens a browser sign-in the first time
   ```
3. Check on github.com: 5 commits, about 64 files (including `LICENSE`), no `data\`, `models\` or `.venv\`
   folders.
