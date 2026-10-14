"""Write the synthetic test fixtures (and their ground truth) to test_data/synthetic/.

Usage (PowerShell, from the project folder):
    .\\.venv\\Scripts\\python.exe scripts\\generate_test_images.py
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from imagecat import fixtures  # noqa: E402


def main() -> None:
    out = ROOT / "test_data" / "synthetic"
    paths = fixtures.write_all(out)
    truth = {name: [asdict(g) for g in fixtures.generate(name)[1]] for name in fixtures.FIXTURES}
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    for p in paths:
        print(p.relative_to(ROOT))
    print(f"ground truth: {(out / 'ground_truth.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
