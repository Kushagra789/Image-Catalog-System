// Interactive bounding-box editor (Streamlit Components v2).
//
// Data in  : { image, width, height, boxes: [{id, x, y, width, height, label, color}],
//              selected, mode: "edit" | "draw", token }
// State out: "box" = { id, x, y, width, height, action: "move"|"resize"|"draw"|"select", seq }
//            (image pixel coordinates; seq increases with every finished gesture)
//
// The geometry helpers are pure functions and are unit-tested with Node
// (tests/test_bbox_editor_js.py).

export const MIN_SIZE = 2;
export const HANDLE_PX = 7;

export function clampBox(b, W, H) {
  let w = Math.max(MIN_SIZE, Math.round(Math.min(b.width, W)));
  let h = Math.max(MIN_SIZE, Math.round(Math.min(b.height, H)));
  let x = Math.round(Math.min(Math.max(0, b.x), W - w));
  let y = Math.round(Math.min(Math.max(0, b.y), H - h));
  return { ...b, x, y, width: w, height: h };
}

export function rectFromPoints(x0, y0, x1, y1, W, H) {
  const x = Math.max(0, Math.min(x0, x1));
  const y = Math.max(0, Math.min(y0, y1));
  const r = Math.min(W, Math.max(x0, x1));
  const b = Math.min(H, Math.max(y0, y1));
  return { x: Math.round(x), y: Math.round(y), width: Math.max(MIN_SIZE, Math.round(r - x)),
           height: Math.max(MIN_SIZE, Math.round(b - y)) };
}

// Which part of a box is under point (px, py) in image coordinates?
// `tol` is the handle tolerance in image pixels. Returns "move", a handle
// name ("nw", "n", "ne", "e", "se", "s", "sw", "w") or null.
export function hitPart(b, px, py, tol) {
  const l = b.x, t = b.y, r = b.x + b.width, bo = b.y + b.height;
  const nearL = Math.abs(px - l) <= tol, nearR = Math.abs(px - r) <= tol;
  const nearT = Math.abs(py - t) <= tol, nearB = Math.abs(py - bo) <= tol;
  const inX = px >= l - tol && px <= r + tol, inY = py >= t - tol && py <= bo + tol;
  if (!inX || !inY) return null;
  if (nearT && nearL) return "nw";
  if (nearT && nearR) return "ne";
  if (nearB && nearL) return "sw";
  if (nearB && nearR) return "se";
  if (nearT) return "n";
  if (nearB) return "s";
  if (nearL) return "w";
  if (nearR) return "e";
  return px >= l && px <= r && py >= t && py <= bo ? "move" : null;
}

// Hit-test several boxes: the selected box wins (its handles must stay
// reachable); otherwise the smallest box containing the point.
export function hitTest(boxes, px, py, tol, selectedId) {
  const sel = boxes.find((b) => b.id === selectedId);
  if (sel) {
    const part = hitPart(sel, px, py, tol);
    if (part) return { id: sel.id, part };
  }
  let best = null;
  for (const b of boxes) {
    if (b.id === selectedId) continue;
    const part = hitPart(b, px, py, 0);
    if (part && (!best || b.width * b.height < best.area)) best = { id: b.id, part: "move", area: b.width * b.height };
  }
  return best ? { id: best.id, part: best.part } : null;
}

// Apply a drag of (dx, dy) image pixels to `start` for the given part.
export function applyDrag(start, part, dx, dy, W, H) {
  if (part === "move") {
    return clampBox({ ...start, x: start.x + dx, y: start.y + dy }, W, H);
  }
  let l = start.x, t = start.y, r = start.x + start.width, b = start.y + start.height;
  if (part.includes("w")) l = Math.min(r - MIN_SIZE, Math.max(0, l + dx));
  if (part.includes("e")) r = Math.max(l + MIN_SIZE, Math.min(W, r + dx));
  if (part.includes("n")) t = Math.min(b - MIN_SIZE, Math.max(0, t + dy));
  if (part.includes("s")) b = Math.max(t + MIN_SIZE, Math.min(H, b + dy));
  return { ...start, x: Math.round(l), y: Math.round(t), width: Math.round(r - l), height: Math.round(b - t) };
}

export const CURSORS = { move: "move", n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize",
                         nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize" };

// ------------------------------------------------------------- component
export default function (component) {
  const { data, parentElement, setStateValue } = component;
  if (!data || !data.image) return;

  let root = parentElement.querySelector(".bbe-root");
  if (!root) {
    root = document.createElement("div");
    root.className = "bbe-root";
    root.innerHTML = '<canvas class="bbe-canvas"></canvas><div class="bbe-hint"></div>';
    parentElement.appendChild(root);
    root.__st = { seq: 0, token: null, img: null, src: null };
  }
  const st = root.__st;
  const canvas = root.querySelector("canvas");
  const hint = root.querySelector(".bbe-hint");
  const ctx = canvas.getContext("2d");
  const W = data.width, H = data.height;

  if (st.token !== data.token) {  // new data from Python replaces local edits
    st.token = data.token;
    st.boxes = (data.boxes || []).map((b) => ({ ...b }));
    st.selected = data.selected;
    st.drag = null;
  }
  st.mode = data.mode || "edit";
  hint.textContent = st.mode === "draw"
    ? "Drag on the image to draw a new box; drag its edges or corners to resize, its inside to move."
    : "Click a box to select it. Drag inside to move, drag edges/corners to resize.";

  const scale = () => canvas.width / W;
  // In draw mode existing objects are context only; just the new box is interactive.
  const interactive = () => (st.mode === "draw" ? st.boxes.filter((b) => b.id === "new") : st.boxes);

  function draw() {
    const s = scale();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (st.img && st.img.complete) ctx.drawImage(st.img, 0, 0, canvas.width, canvas.height);
    for (const b of st.boxes) {
      const sel = b.id === st.selected;
      ctx.lineWidth = sel ? 3 : 1.5;
      ctx.strokeStyle = b.color || "#e6003c";
      ctx.globalAlpha = sel || st.selected == null ? 1 : 0.55;
      ctx.strokeRect(b.x * s, b.y * s, b.width * s, b.height * s);
      if (b.label) {
        ctx.font = "12px sans-serif";
        const tw = ctx.measureText(b.label).width + 8;
        ctx.fillStyle = b.color || "#e6003c";
        ctx.fillRect(b.x * s, Math.max(0, b.y * s - 16), tw, 16);
        ctx.fillStyle = "#fff";
        ctx.fillText(b.label, b.x * s + 4, Math.max(12, b.y * s - 4));
      }
      if (sel) {
        ctx.fillStyle = "#fff";
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1;
        const xs = [b.x, b.x + b.width / 2, b.x + b.width], ys = [b.y, b.y + b.height / 2, b.y + b.height];
        for (const hx of xs) for (const hy of ys) {
          if (hx === xs[1] && hy === ys[1]) continue;
          ctx.fillRect(hx * s - 4, hy * s - 4, 8, 8);
          ctx.strokeRect(hx * s - 4, hy * s - 4, 8, 8);
        }
      }
      ctx.globalAlpha = 1;
    }
  }

  function fit() {
    const avail = Math.max(50, root.clientWidth || parentElement.clientWidth || W);
    const w = Math.min(avail, W);
    canvas.width = Math.round(w);
    canvas.height = Math.round((w * H) / W);
    draw();
  }

  if (st.src !== data.image) {
    st.src = data.image;
    st.img = new Image();
    st.img.onload = fit;
    st.img.src = data.image;
  }

  // Pointer position in image pixels, using the size the canvas is displayed at.
  const toImage = (ev) => {
    const r = canvas.getBoundingClientRect();
    return { x: ((ev.clientX - r.left) / (r.width || 1)) * W, y: ((ev.clientY - r.top) / (r.height || 1)) * H };
  };
  const tolerance = () => HANDLE_PX / ((canvas.getBoundingClientRect().width || canvas.width) / W);

  function send(box, action) {
    st.seq += 1;
    setStateValue("box", { id: box.id, x: box.x, y: box.y, width: box.width, height: box.height, action,
                           seq: Date.now() * 1000 + st.seq });
  }

  function onDown(ev) {
    const p = toImage(ev);
    const hit = hitTest(interactive(), p.x, p.y, tolerance(), st.selected);
    canvas.setPointerCapture(ev.pointerId);
    if (hit) {
      const box = st.boxes.find((b) => b.id === hit.id);
      if (st.mode === "edit" && hit.id !== st.selected) {
        st.selected = hit.id;  // just select; editing starts on the next drag
        draw();
        send(box, "select");
        st.drag = null;
        return;
      }
      st.selected = hit.id;
      st.drag = { id: hit.id, part: hit.part, x0: p.x, y0: p.y, start: { ...box }, moved: false };
    } else if (st.mode === "draw") {
      st.boxes = st.boxes.filter((b) => b.id !== "new");
      const nb = { id: "new", x: Math.round(p.x), y: Math.round(p.y), width: MIN_SIZE, height: MIN_SIZE,
                   label: "new", color: "#e6003c" };
      st.boxes.push(nb);
      st.selected = "new";
      st.drag = { id: "new", part: "draw", x0: p.x, y0: p.y, start: { ...nb }, moved: false };
    }
    draw();
  }

  function onMove(ev) {
    const p = toImage(ev);
    if (!st.drag) {
      const hit = hitTest(interactive(), p.x, p.y, tolerance(), st.selected);
      canvas.style.cursor = hit ? (hit.id === st.selected ? CURSORS[hit.part] : "pointer")
                                : (st.mode === "draw" ? "crosshair" : "default");
      return;
    }
    const d = st.drag;
    const idx = st.boxes.findIndex((b) => b.id === d.id);
    if (idx < 0) return;
    const next = d.part === "draw"
      ? { ...st.boxes[idx], ...rectFromPoints(d.x0, d.y0, p.x, p.y, W, H) }
      : applyDrag(d.start, d.part, p.x - d.x0, p.y - d.y0, W, H);
    st.boxes[idx] = { ...st.boxes[idx], ...next };
    d.moved = true;
    draw();
  }

  function onUp() {
    const d = st.drag;
    st.drag = null;
    if (!d || !d.moved) return;
    const box = st.boxes.find((b) => b.id === d.id);
    if (box) send(box, d.part === "draw" ? "draw" : d.part === "move" ? "move" : "resize");
  }

  canvas.onpointerdown = onDown;
  canvas.onpointermove = onMove;
  canvas.onpointerup = onUp;
  canvas.onpointercancel = onUp;

  if (!st.observer && typeof ResizeObserver !== "undefined") {
    st.observer = new ResizeObserver(() => fit());
    st.observer.observe(root);
  }
  if (st.img && st.img.complete) fit();
}
