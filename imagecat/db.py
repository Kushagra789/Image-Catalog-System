"""SQLite persistence layer.

A new connection is opened per operation, which keeps the class safe to use
from Streamlit's re-running script threads.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import (
    ORIGIN_AUTO,
    ImageRecord,
    ObjectRecord,
    STATUS_PENDING,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT NOT NULL,
    display_name         TEXT NOT NULL,
    filename             TEXT NOT NULL UNIQUE,
    width                INTEGER NOT NULL,
    height               INTEGER NOT NULL,
    created_at           TEXT NOT NULL,
    detection_status     TEXT NOT NULL DEFAULT 'pending',
    detection_message    TEXT,
    detected_at          TEXT,
    file_hash            TEXT,
    is_composite         INTEGER NOT NULL DEFAULT 0,
    source_asset_ids     TEXT,
    composition_metadata TEXT,
    dominant_colors      TEXT
);

CREATE TABLE IF NOT EXISTS objects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    color       TEXT,
    confidence  REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    x           INTEGER NOT NULL,
    y           INTEGER NOT NULL,
    width       INTEGER NOT NULL CHECK (width > 0),
    height      INTEGER NOT NULL CHECK (height > 0),
    source      TEXT NOT NULL,
    origin      TEXT NOT NULL DEFAULT 'auto',
    is_edited   INTEGER NOT NULL DEFAULT 0,
    is_deleted  INTEGER NOT NULL DEFAULT 0,
    attributes  TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_objects_image ON objects(image_id);
CREATE INDEX IF NOT EXISTS idx_objects_label ON objects(label);

CREATE TABLE IF NOT EXISTS detection_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    run_at      TEXT NOT NULL,
    mode        TEXT NOT NULL,
    status      TEXT NOT NULL,
    message     TEXT,
    num_objects INTEGER NOT NULL DEFAULT 0,
    details     TEXT NOT NULL DEFAULT '{}'
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value)


class Database:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ images
    @staticmethod
    def _row_to_image(row: sqlite3.Row) -> ImageRecord:
        return ImageRecord(
            id=row["id"],
            name=row["name"],
            display_name=row["display_name"],
            filename=row["filename"],
            width=row["width"],
            height=row["height"],
            created_at=row["created_at"],
            detection_status=row["detection_status"],
            detection_message=row["detection_message"],
            detected_at=row["detected_at"],
            file_hash=row["file_hash"],
            is_composite=bool(row["is_composite"]),
            source_asset_ids=_loads(row["source_asset_ids"]),
            composition_metadata=_loads(row["composition_metadata"]),
            dominant_colors=_loads(row["dominant_colors"]),
        )

    def insert_image(
        self,
        *,
        name: str,
        display_name: str,
        filename: str,
        width: int,
        height: int,
        file_hash: str | None = None,
        detection_status: str = STATUS_PENDING,
        detection_message: str | None = None,
        is_composite: bool = False,
        source_asset_ids: dict | None = None,
        composition_metadata: dict | None = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """INSERT INTO images (name, display_name, filename, width, height, created_at,
                       detection_status, detection_message, file_hash, is_composite,
                       source_asset_ids, composition_metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    name, display_name, filename, width, height, utcnow(),
                    detection_status, detection_message, file_hash, int(is_composite),
                    _dumps(source_asset_ids), _dumps(composition_metadata),
                ),
            )
            return int(cur.lastrowid)

    def get_image(self, image_id: int) -> ImageRecord | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return self._row_to_image(row) if row else None

    def list_images(self) -> list[ImageRecord]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM images ORDER BY id DESC").fetchall()
        return [self._row_to_image(r) for r in rows]

    def find_image_by_hash(self, file_hash: str) -> ImageRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM images WHERE file_hash = ? ORDER BY id LIMIT 1", (file_hash,)
            ).fetchone()
        return self._row_to_image(row) if row else None

    def update_image(self, image_id: int, conn: sqlite3.Connection | None = None, **fields: Any) -> None:
        allowed = {
            "display_name", "detection_status", "detection_message", "detected_at",
            "dominant_colors", "composition_metadata", "source_asset_ids",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Cannot update image fields: {sorted(unknown)}")
        json_fields = {"dominant_colors", "composition_metadata", "source_asset_ids"}
        sets, values = [], []
        for key, value in fields.items():
            sets.append(f"{key} = ?")
            values.append(_dumps(value) if key in json_fields else value)
        if not sets:
            return
        sql = f"UPDATE images SET {', '.join(sets)} WHERE id = ?"
        if conn is not None:
            conn.execute(sql, (*values, image_id))
            return
        with self.connect() as c:
            c.execute(sql, (*values, image_id))

    def delete_image(self, image_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM images WHERE id = ?", (image_id,))

    # ----------------------------------------------------------------- objects
    @staticmethod
    def _row_to_object(row: sqlite3.Row) -> ObjectRecord:
        return ObjectRecord(
            id=row["id"],
            image_id=row["image_id"],
            label=row["label"],
            color=row["color"],
            confidence=row["confidence"],
            x=row["x"],
            y=row["y"],
            width=row["width"],
            height=row["height"],
            source=row["source"],
            origin=row["origin"],
            is_edited=bool(row["is_edited"]),
            is_deleted=bool(row["is_deleted"]),
            attributes=_loads(row["attributes"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def insert_object(
        self,
        *,
        image_id: int,
        label: str,
        color: str | None,
        confidence: float | None,
        bbox: tuple[int, int, int, int],
        source: str,
        origin: str = ORIGIN_AUTO,
        attributes: dict | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        x, y, w, h = (int(v) for v in bbox)
        now = utcnow()
        params = (
            image_id, label, color, confidence, x, y, w, h, source, origin,
            json.dumps(attributes or {}), now, now,
        )
        sql = """INSERT INTO objects (image_id, label, color, confidence, x, y, width, height,
                     source, origin, attributes, created_at, updated_at)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"""
        if conn is not None:
            return int(conn.execute(sql, params).lastrowid)
        with self.connect() as c:
            return int(c.execute(sql, params).lastrowid)

    def get_object(self, object_id: int) -> ObjectRecord | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM objects WHERE id = ?", (object_id,)).fetchone()
        return self._row_to_object(row) if row else None

    def list_objects(self, image_id: int, include_deleted: bool = False,
                     conn: sqlite3.Connection | None = None) -> list[ObjectRecord]:
        sql = "SELECT * FROM objects WHERE image_id = ?"
        if not include_deleted:
            sql += " AND is_deleted = 0"
        if conn is not None:
            rows = conn.execute(sql + " ORDER BY id", (image_id,)).fetchall()
        else:
            with self.connect() as c:
                rows = c.execute(sql + " ORDER BY id", (image_id,)).fetchall()
        return [self._row_to_object(r) for r in rows]

    def all_active_objects(self, include_composites: bool = False) -> list[ObjectRecord]:
        sql = """SELECT o.* FROM objects o JOIN images i ON i.id = o.image_id
                 WHERE o.is_deleted = 0"""
        if not include_composites:
            sql += " AND i.is_composite = 0"
        with self.connect() as conn:
            rows = conn.execute(sql + " ORDER BY o.id", ()).fetchall()
        return [self._row_to_object(r) for r in rows]

    def update_object(self, object_id: int, **fields: Any) -> None:
        allowed = {
            "label", "color", "confidence", "x", "y", "width", "height",
            "is_edited", "is_deleted", "attributes",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Cannot update object fields: {sorted(unknown)}")
        sets, values = [], []
        for key, value in fields.items():
            sets.append(f"{key} = ?")
            if key == "attributes":
                value = json.dumps(value or {})
            elif key in ("is_edited", "is_deleted"):
                value = int(bool(value))
            values.append(value)
        sets.append("updated_at = ?")
        values.append(utcnow())
        with self.connect() as conn:
            conn.execute(f"UPDATE objects SET {', '.join(sets)} WHERE id = ?", (*values, object_id))

    def hard_delete_objects(self, object_ids: list[int], conn: sqlite3.Connection) -> None:
        if object_ids:
            marks = ",".join("?" * len(object_ids))
            conn.execute(f"DELETE FROM objects WHERE id IN ({marks})", object_ids)

    # ----------------------------------------------------------- detection log
    def log_detection_run(
        self,
        image_id: int,
        mode: str,
        status: str,
        message: str | None,
        num_objects: int,
        details: dict | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        sql = """INSERT INTO detection_runs (image_id, run_at, mode, status, message, num_objects, details)
                 VALUES (?,?,?,?,?,?,?)"""
        params = (image_id, utcnow(), mode, status, message, num_objects, json.dumps(details or {}))
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self.connect() as c:
                c.execute(sql, params)

    def list_detection_runs(self, image_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM detection_runs WHERE image_id = ? ORDER BY id DESC", (image_id,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["details"] = _loads(d["details"], {})
            out.append(d)
        return out

    def object_counts(self) -> dict[int, int]:
        """Active object count per image in one query (the catalog grid needs all of them)."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT image_id, COUNT(*) FROM objects WHERE is_deleted = 0 GROUP BY image_id").fetchall()
        return {int(r[0]): int(r[1]) for r in rows}

    def image_ids_with_label(self, text: str) -> set[int]:
        """Images that have an active object whose label contains ``text``."""
        pattern = "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT image_id FROM objects WHERE is_deleted = 0 AND label LIKE ? ESCAPE '\\'",
                (pattern,)).fetchall()
        return {int(r[0]) for r in rows}

    def composites_using(self, image_id: int) -> list[ImageRecord]:
        """Composite images whose provenance lists ``image_id`` as a source."""
        return [r for r in self.list_images()
                if r.is_composite and image_id in (r.source_asset_ids or {}).get("image_ids", [])]

    # ------------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            n_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            n_composites = conn.execute("SELECT COUNT(*) FROM images WHERE is_composite = 1").fetchone()[0]
            by_status = dict(conn.execute(
                "SELECT detection_status, COUNT(*) FROM images GROUP BY detection_status").fetchall())
            by_source = dict(conn.execute(
                "SELECT source, COUNT(*) FROM objects WHERE is_deleted = 0 GROUP BY source").fetchall())
            by_label = dict(conn.execute(
                """SELECT label, COUNT(*) FROM objects WHERE is_deleted = 0
                   GROUP BY label ORDER BY COUNT(*) DESC LIMIT 15""").fetchall())
            n_corrections = conn.execute(
                """SELECT COUNT(*) FROM objects
                   WHERE origin = 'manual' OR is_edited = 1 OR is_deleted = 1""").fetchone()[0]
            n_objects = conn.execute("SELECT COUNT(*) FROM objects WHERE is_deleted = 0").fetchone()[0]
        return {
            "images": n_images,
            "composites": n_composites,
            "objects": n_objects,
            "corrections": n_corrections,
            "by_status": by_status,
            "by_source": by_source,
            "by_label": by_label,
        }
