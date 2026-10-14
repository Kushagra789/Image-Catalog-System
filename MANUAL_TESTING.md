# Manual Testing Checklist

Automated tests (208) cover the logic, the database and every page headlessly. The items below can only be
confirmed by a person using a real browser, or with real images that are not in the repository.
Tick each box and note anything unexpected.

**Before you start**

```powershell
cd C:\Users\kusha\Image-Catalog-System
.\.venv\Scripts\Activate.ps1
Copy-Item data\catalog.db "data\backups\catalog_$(Get-Date -Format yyyyMMdd_HHmmss).db"   # safety backup
streamlit run app.py      # if it was already running before the latest code changes, stop it (Ctrl+C) and start again
```

Open <http://localhost:8501>. Real cartoon test images: `test_data\real_cartoons\` (licences in `ATTRIBUTION.md`).

---

## A. Mouse bounding-box editor (highest priority: never tested with a real pointer)
Manual Editor → choose an image with several objects.
- [ ] Clicking a box selects it (handles appear, the object list switches to it).
- [ ] Dragging inside the selected box moves it; the x/y fields update; the box stays where you dropped it.
- [ ] Dragging a corner or edge resizes it; width/height update; a second and third drag also stay in place.
- [ ] A box cannot be dragged or resized outside the image.
- [ ] **Save changes** → success message; F5 → the new box is still there; the table shows `edited`.
- [ ] **Reset box** discards unsaved box changes.
- [ ] *Add object* → dragging on an empty area draws a new red box; existing boxes do not move; **Add object** saves it.
- [ ] Turning *Mouse editing* off shows a static preview, and the numeric inputs still work.
- [ ] Works after resizing the browser window (the canvas fits, clicks still hit the right box).
- [ ] (Optional) Touch screen / tablet: untested, so note the behaviour.

## B. Correction safety
- [ ] Relabel an object with **Use** (alternative label) or **Apply** (suggestion), then change its box and **Save
      changes** → the new label is kept (regression of a fixed bug).
- [ ] Object Detection → *Keep my corrections* re-run → your edits, additions and removals are unchanged; removed
      detections do not come back; a moved box does not reappear at its old place.
- [ ] *Replace everything* → the button stays disabled until the confirmation box is ticked.

## C. Cartoon detection with real images
Upload all files from `test_data\real_cartoons\` (expected values from the automated dry run):
- [ ] `boy_and_girl.png`: two `cartoon character` objects (~0.71 / ~0.63); both show the unverified suggestion
      "cartoon girl" (wrong for the boy, as documented); nothing is labelled boy/girl automatically.
- [ ] `apple_cc0.png`: `fruit` (~0.60). `green_apples.png`: `apple` (~0.20) and `sports ball` (wrong, flagged).
- [ ] `orange_retro.png`: only foreground objects. `boy_braces.jpg`, `boy_girl_1913.jpg`: *nothing detected* (known misses).
- [ ] Object Detection → *Hide parts of larger objects* removes head/shirt/leg shapes from the overlay only.
- [ ] Your own cartoons (not in the repository): record what was found and missed.

## D. Real photos
- [ ] A photo of a single object on a plain white background (like your clementine) is reported as a photo: no
      shape/colour-region objects, no "verify (illustration)" flag on YOLO results.
- [ ] Your existing image #2 (clementine) still shows 8 regions from before the fix. To refresh it: Object
      Detection → *Keep my corrections* → **Run detection** (it has no corrections, so nothing of yours is lost).

## E. Composite generation
Relabel first (Manual Editor): the boy → `cartoon boy`, the girl → `cartoon girl`, the clip-art `fruit` → `apple` (red).
- [ ] `Place an apple on the left, an orange on the right, and a cartoon boy in the center.` → layers in the left/centre/
      right thirds; the *placed* column says "from instruction"; **Save composite** works.
- [ ] `a cartoon girl in the foreground and a cartoon boy behind the girl` → the boy is drawn behind the girl.
- [ ] `one banana and a cartoon boy` → banana reported missing with a suggestion; Save is disabled until
      "save anyway" is ticked.
- [ ] Editing x/y/scale/z in the layer table updates the preview; clearing a cell does not crash the page.
- [ ] Cut-out quality with `auto`: no white rim around photo objects (rembg); flat art cut exactly.
- [ ] Metadata Viewer on the saved composite shows source thumbnails, source IDs and per-layer `source_snapshot`.

## F. General
- [ ] Dashboard → **Run health check** → "No problems found".
- [ ] Image Catalog: search by an object label (e.g. `orange`), filters, rename; the delete dialog warns about
      corrections and about composites that use the image (cancel without deleting).
- [ ] Upload an invalid file (e.g. a renamed .txt) → a clear error, nothing added.
- [ ] Spinners appear during detection, planning and saving; no page shows a raw Python traceback.
