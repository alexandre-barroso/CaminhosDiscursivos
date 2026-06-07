#!/usr/bin/env python3
import argparse
import gzip
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

from preprocess import normalize_term


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_header(line):
    parts = line.strip().split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return None


def build_store(source_path, output_path, language, limit=None):
    output_path = Path(output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(tmp_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("CREATE TABLE vectors (word TEXT PRIMARY KEY, vector BLOB NOT NULL)")

    inserted = 0
    seen = 0
    dimension = None
    batch = []

    with open_text(source_path) as handle:
        first = handle.readline()
        header = parse_header(first)
        if header:
            _total, dimension = header
        else:
            handle.seek(0)
        for line in handle:
            seen += 1
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue
            raw_word = parts[0]
            word = normalize_term(raw_word)
            if not word:
                continue
            try:
                vector = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            if dimension is None:
                dimension = int(vector.shape[0])
            if int(vector.shape[0]) != dimension:
                continue
            batch.append((word, vector.tobytes()))
            if len(batch) >= 5000:
                conn.executemany("INSERT OR IGNORE INTO vectors(word, vector) VALUES (?, ?)", batch)
                inserted += conn.execute("SELECT changes()").fetchone()[0]
                batch.clear()
            if limit and seen >= limit:
                break
            if seen % 100000 == 0:
                print(f"{language}: scanned {seen:,} rows", file=sys.stderr, flush=True)

    if batch:
        before = conn.total_changes
        conn.executemany("INSERT OR IGNORE INTO vectors(word, vector) VALUES (?, ?)", batch)
        inserted += conn.total_changes - before

    count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    conn.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [
            ("language", language),
            ("source", os.path.basename(str(source_path))),
            ("dimension", str(dimension or 0)),
            ("count", str(count)),
            ("schema", "caminhos.vector_store.v1"),
        ],
    )
    conn.execute("CREATE INDEX IF NOT EXISTS vectors_word_idx ON vectors(word)")
    conn.commit()
    conn.close()
    tmp_path.replace(output_path)
    print(f"Built {output_path} with {count:,} vectors, dimension {dimension}.")


def main():
    parser = argparse.ArgumentParser(description="Build a normalized SQLite vector store from a fastText .vec or .vec.gz file.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", required=True, choices=["pt", "en", "es"])
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for smoke fixtures.")
    args = parser.parse_args()
    build_store(args.source, args.output, args.language, args.limit)


if __name__ == "__main__":
    main()
