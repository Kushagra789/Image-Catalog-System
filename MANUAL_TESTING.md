# Manual / Browser Testing: Checklist and Results

_Last run: 2026-10-16. Status per item: **PASS**, **FAIL→FIXED** (failed, fixed with approval, then passed) or
**NOT TESTED** (needs a person or material that is not available)._

## How the items were tested

The Claude Chrome extension could not be authorised for this account, so the browser items were run with
**Playwright** (Chromium 156, real mouse/keyboard events, real file uploads, page reloads, window resizing,
screenshots). Details:

- **Isolated app.** A separate app instance on port 8502 used a **copy** of the catalog (`IMAGECAT_DATA_DIR`
  pointing to a scratch folder). The live catalog in `data\` was **not touched**; it was byte-identical to the
  pre-test backup `data\backups\catalog_20261016_210223_before_manual_tests.db` afterwards.
- **Evidence for every item:**
  - what the page showed (field values, alerts, dropdown values),
  - for the mouse editor, the **pixels of the canvas** (box outline drawn at the new position),
  - the database row afterwards (on the copy).
- **Stored evidence:** screenshots, composites and `results.json` are in
  `test_evidence\2026-10-16_playwright\` (git-ignored: it contains copies of your photos).
- **Not covered by automation:** a human's visual judgement (I reviewed the screenshots and composites myself;
  notes below) and touch input.

Earlier on 2026-10-16, A1–A5 were reported as passed in chat, but the database showed the steps had not reached
the app. Those reports were withdrawn; every result below comes from the Playwright run.

## Summary

| Section | PASS | FAIL→FIXED | NOT TESTED |
|---|---|---|---|
| A. Mouse box editor | 9 | 0 | 1 (touch) |
| B. Correction safety | 2 | 1 (B1) | 0 |
| C. Cartoon detection | 4 | 0 | 1 (your own cartoons) |
| D. Real photos | 1 | 0 | 1 (refreshing image #2 is your decision, not a test) |
| E. Composites | 5 | 0 | 1 (layer-table cell editing in the browser) |
| F. General | 4 | 0 | 0 |
| Extra: detection on all fixtures | 7 | 1 (shapes/colours) | 0 |

**Defects found and fixed during this run** (each fixed with your approval, plus regression tests; see
`PROJECT_REPORT.md` §8.3, items 21–22):
1. On illustrations, a model label replaced a correct OpenCV shape or colour region: YOLO "kite" 0.57 replaced
   the orange colour swatch, and YOLOE "orange" 0.59 replaced the orange ellipse. Threshold raised to 0.65.
2. In the real browser, a label changed with **Use/Apply** was reverted by the next **Save changes**, and a drag
   right after a canvas selection could be dropped. Both were caused by the browser sending back stale widget
   values. Fixed with revision-based form keys and a non-widget selection state. This was invisible to the
   headless test runner, so a real-browser end-to-end test now guards it.

---

## A. Mouse bounding-box editor (`shapes` fixture)
- [x] **A1 PASS**: Clicking a box selects it. Clicking the blue square selected `#38 square (blue)`; fields
      (260, 60, 141, 141); outline drawn on the canvas at that box.
- [x] **A2 PASS**: Dragging moves the box and it stays put. Two drags: (260,60) → (320,60) → (320,100); the
      canvas outline was at the new place after each; width/height unchanged.
- [x] **A3 PASS**: Three resizes (bottom-right corner, left edge, top-right corner): (320,100,141,141) →
      (320,100,171,161) → (340,100,151,161) → (340,90,166,171); the outline followed every time.
- [x] **A4 PASS**: The box can't leave the image. Dragged past the right edge: x + w = 640. Resized past the
      bottom-right corner: y + h = 480 (the resize really happened: height 171 → 390).
- [x] **A5 PASS**: Save, then reload. Fields before Save (247, 25, 166, 390) = SQLite row = values after a page
      reload (`is_edited = 1`, ✎ shown); success message shown.
- [x] **A6 PASS**: Reset box. An unsaved drag to x = 147 returned to the saved (247, 25, 166, 390); the database
      was unchanged.
- [x] **A7 PASS**: Draw in *Add object*. A drag on an empty area gave (10, 412, 140, 58); saved as a `manual`
      object; all other boxes unchanged.
- [x] **A8 PASS**: Mouse editing off. No canvas; static preview shown; numeric x = 123 accepted.
- [x] **A9 PASS**: Smaller window (900 px). The canvas refit to 256 px wide; a click still selected the right
      object (`#35 circle`).
- [ ] **A10 NOT TESTED**: Touch screen / tablet. Playwright can only emulate taps, not real touch drags; needs a
      touch device.

## B. Correction safety
- [x] **B1 FAIL→FIXED→PASS**: Relabel with **Use**, move the box, **Save**: label and box are both kept.
      - *First run:* the stored label reverted to `circle`.
      - *After the fix:* stored label `apple`, x 50 → 65.
      - *Now guarded by:* `tests/test_browser_e2e.py` (real Chromium).
- [x] **B2 PASS**: *Keep my corrections* re-run: the 3 corrections were identical before and after; the moved
      square was not re-created at its old place.
- [x] **B3 PASS**: *Replace everything*: **Run detection** was disabled until the confirmation box was ticked,
      then enabled (never clicked).

## C. Cartoon detection with real images (uploaded through the real file dialog)
- [x] **C1 PASS**: `boy_and_girl.png`: two `cartoon character` objects (YOLOE 0.71, YOLO 0.63); both carry only
      the unverified suggestion "cartoon girl" (wrong for the boy, as documented); no automatic boy/girl label.
- [x] **C2 PASS**: `apple_cc0.png` → `fruit` 0.60. `green_apples.png` → `apple` 0.20 plus `sports ball` 0.54
      (wrong, flagged).
- [x] **C3 PASS** (known misses confirmed): `orange_retro.png` → only foreground objects; `boy_braces.jpg` and
      `boy_girl_1913.jpg` → *nothing detected*.
- [x] **C4 PASS**: *Hide parts of larger objects* removed the head/shirt/leg shapes from the overlay (checked by
      eye); the database still holds all 20 objects.
- [ ] **C5 NOT TESTED**: Your own cartoon images (not available to me). Upload a few and note what is found and missed.

## D. Real photos
- [x] **D1 PASS**: Your Clementine photo, uploaded again as a new image, is classified as a **photo**: one object
      `orange` 0.92 (YOLO), no OpenCV regions, no "verify (illustration)" flag.
- [ ] **D2 YOUR DECISION** (not a test): Your existing image #2 still has the 8 regions from before the photo fix.
      Refreshing it (Object Detection → *Keep my corrections* → **Run detection**; #2 has no corrections)
      measurably improves its cut-out. The old record gives a notch near the leaves and a white shadow blob;
      a refreshed record gives a clean rembg cut-out. See `test_evidence\…\composites\orange_cutout_compare.png`.
      **Not done; your catalog was not modified.**

## E. Composite generation (through the real browser, Save clicked, result checked in the database)
- [x] **E (apple + orange) PASS**: "Create an image containing one red apple and one orange."
      - *Sources:* apple from your photo **#1** (rembg cut-out), orange from your Clementine photo **#2**;
        the loading spinner was shown.
      - *Ranking note:* in an earlier attempt, the red circle that test B1 had relabelled "apple" was chosen
        instead, because user-verified labels rank first by design. The label was reset for this check.
- [x] **E1 PASS**: "Place an apple on the left, an orange on the right, and a cartoon boy in the center." →
      x-centres 133 / 400 / 666 on an 800 px canvas; every layer placed "from instruction".
- [x] **E2 PASS**: "a cartoon girl in the foreground and a cartoon boy behind the girl" → z: boy −1, girl 100;
      the layers overlap (the boy is drawn behind).
- [x] **E (cartoon boy, girl and coloured shapes) PASS**: "a cartoon boy, a cartoon girl, a green triangle and a
      yellow rectangle on a blue background" → all 4 layers, blue background; cartoons cut with rembg, shapes
      with exact outlines. (Setup on the copy: the two real cartoon children were labelled boy/girl first.)
- [x] **E3 PASS**: "one banana and a cartoon boy" → "Missing: 1 × banana"; Save was disabled until
      *save anyway* was ticked (then enabled; not clicked; no composite created).
- [ ] **E4 NOT TESTED in the browser**: Editing x/y/scale/z in the layer table. That table is drawn on a canvas
      grid that Playwright can't fill reliably. The logic, including "a cleared cell does not crash", is covered
      by automated tests (`test_review_fixes.py`). Please try it once by hand.
- [x] **E5 PASS against the stated criteria** (no white rim on photo objects; flat art cut exactly), checked by
      eye, with two observations:
      - **Clipping to the detection box:** cut-outs are limited to the object's detection box. YOLO's box for your
        apple photo is slightly tight, so the left edge and the stem tip are clipped; the orange's leaves are
        clipped where they leave the box.
      - **Old image #2:** gives a poorer orange cut-out (see D2).
- [x] **E6 PASS**: Metadata Viewer on a saved composite: provenance section with source thumbnails, source
      object IDs, `source_snapshot` on every layer.

## F. General
- [x] **F1 PASS**: Dashboard → **Run health check** → "No problems found (24 image(s), SQLite integrity: ok)".
- [x] **F2 PASS**: Image Catalog search "orange" → "10 of 24 image(s)". The delete dialog warns about manual
      corrections (`shapes`) and about composites that use your apple photo (#1). Both were cancelled; nothing
      was deleted.
- [x] **F3 PASS**: Invalid upload (`not_an_image.png`): a clear error; image count unchanged (19 → 19).
- [x] **F4 PASS**: The loading spinner was observed during composite planning; no Python traceback on any of the
      7 pages.

## Extra: automatic detection on every fixture (through the real upload button)
- [x] shapes 6/6 (after fix), colour swatches 6/6 (after fix), overlapping shapes with correct occlusion flags,
      blank → "empty, the image appears blank", cartoon boy/girl fixtures → `cartoon character` with no
      automatic gender.

---

## Still to do by hand
1. **A10:** try the mouse editor on a touch device (optional).
2. **C5:** upload a few of your own cartoons and note the results.
3. **E4:** in the Composite Generator, change x/y/scale/z in the layer table and clear one cell.
4. **D2:** decide whether to refresh image #2 (recommended for better cut-outs).

How to run the app for these: `.\.venv\Scripts\Activate.ps1` then `streamlit run app.py` (restart it first if it
was started before 2026-10-16 22:05, the time of the last code fix).
