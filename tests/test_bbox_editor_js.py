"""Unit tests for the editor's JavaScript geometry, executed with Node.js."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parent.parent / "imagecat" / "ui" / "static" / "bbox_editor.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")

TEST_SCRIPT = r"""
import assert from "node:assert/strict";
const m = await import(process.argv[2]);
const W = 640, H = 480;
const box = { id: 1, x: 100, y: 100, width: 200, height: 100 };
const results = {};
function check(name, fn) { fn(); results[name] = "ok"; }

check("hit inside is move", () => assert.equal(m.hitPart(box, 150, 150, 5), "move"));
check("hit corners", () => {
  assert.equal(m.hitPart(box, 100, 100, 5), "nw");
  assert.equal(m.hitPart(box, 300, 200, 5), "se");
  assert.equal(m.hitPart(box, 302, 98, 5), "ne");
  assert.equal(m.hitPart(box, 98, 203, 5), "sw");
});
check("hit edges", () => {
  assert.equal(m.hitPart(box, 200, 100, 5), "n");
  assert.equal(m.hitPart(box, 200, 201, 5), "s");
  assert.equal(m.hitPart(box, 101, 150, 5), "w");
  assert.equal(m.hitPart(box, 299, 150, 5), "e");
});
check("miss outside", () => assert.equal(m.hitPart(box, 10, 10, 5), null));
check("move clamps to image", () => {
  assert.deepEqual(m.applyDrag(box, "move", 30, -20, W, H), { ...box, x: 130, y: 80 });
  const far = m.applyDrag(box, "move", 1000, 1000, W, H);
  assert.equal(far.x, W - box.width); assert.equal(far.y, H - box.height);
  const neg = m.applyDrag(box, "move", -500, -500, W, H);
  assert.equal(neg.x, 0); assert.equal(neg.y, 0);
});
check("resize corner", () => {
  assert.deepEqual(m.applyDrag(box, "se", 50, 20, W, H), { ...box, width: 250, height: 120 });
  assert.deepEqual(m.applyDrag(box, "nw", -10, -10, W, H), { ...box, x: 90, y: 90, width: 210, height: 110 });
});
check("resize never inverts or leaves image", () => {
  const r = m.applyDrag(box, "e", -1000, 0, W, H);
  assert.equal(r.x, 100); assert.equal(r.width, m.MIN_SIZE);
  const l = m.applyDrag(box, "w", -1000, 0, W, H);
  assert.equal(l.x, 0); assert.equal(l.width, 300);
  const s = m.applyDrag(box, "s", 0, 10000, W, H);
  assert.equal(s.y + s.height, H);
});
check("draw rectangle from any corner", () => {
  assert.deepEqual(m.rectFromPoints(300, 200, 100, 50, W, H), { x: 100, y: 50, width: 200, height: 150 });
  assert.deepEqual(m.rectFromPoints(-20, -20, 9999, 9999, W, H), { x: 0, y: 0, width: W, height: H });
  assert.equal(m.rectFromPoints(10, 10, 10, 10, W, H).width, m.MIN_SIZE);
});
check("selected box wins hit test, else smallest", () => {
  const big = { id: "big", x: 0, y: 0, width: 400, height: 400 };
  const small = { id: "small", x: 50, y: 50, width: 50, height: 50 };
  assert.deepEqual(m.hitTest([big, small], 60, 60, 5, null), { id: "small", part: "move" });
  assert.deepEqual(m.hitTest([big, small], 60, 60, 5, "big"), { id: "big", part: "move" });
  assert.deepEqual(m.hitTest([big, small], 400, 400, 5, "big"), { id: "big", part: "se" });
  assert.equal(m.hitTest([big, small], 500, 500, 5, null), null);
});
check("clampBox", () => {
  assert.deepEqual(m.clampBox({ x: -5, y: 470, width: 1000, height: 50 }, W, H),
                   { x: 0, y: 430, width: 640, height: 50 });
});
check("component is the default export", () => assert.equal(typeof m.default, "function"));
console.log(JSON.stringify(results));
"""


def test_geometry_with_node(tmp_path):
    # Copy to .mjs so Node treats it as an ES module regardless of package settings.
    module = tmp_path / "bbox_editor.mjs"
    module.write_text(JS.read_text(encoding="utf-8"), encoding="utf-8")
    runner = tmp_path / "run.mjs"
    runner.write_text(TEST_SCRIPT, encoding="utf-8")
    proc = subprocess.run([NODE, str(runner), module.as_uri()], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    results = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(results) == 11 and set(results.values()) == {"ok"}
