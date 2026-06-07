#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

from preprocess import apply_vocab_cap, normalize_term, preprocess_text


SEED = 42
np.random.seed(SEED)

LANGUAGE_NAMES = {"pt": "Portuguese", "en": "English", "es": "Spanish"}
SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")

TEXT = {
    "en": {
        "language_names": {"pt": "Portuguese", "en": "English", "es": "Spanish"},
        "report_title": "Caminhos Discursivos - transient web report",
        "language": "Language",
        "corpora": "Corpora",
        "pairs": "Pairs",
        "retention": "Server-side retention: none. Uploaded corpora and intermediate files lived only in a temporary request workspace.",
        "preprocessing": "Preprocessing: lowercase, diacritics removed, numbers removed, symbols removed, language stopwords filtered, strict alphabetic tokens only.",
        "interpretive": "Interpretive note: geometric paths are starting points for discourse interpretation, not proof of causal or ideological relations.",
        "corpus": "Corpus",
        "clean_tokens": "Clean tokens",
        "vocabulary": "Vocabulary",
        "model": "Model",
        "truncated": "Note: corpus was truncated at the web-demo token cap.",
        "not_computed": "not computed",
        "pdf": "PDF",
        "png": "PNG",
        "figure_subtitle": "Corpus: {corpus} | Language: {language}",
        "zoom": "detail",
        "path_axis": "Path axis",
        "reason_one_word": "each term must reduce to exactly one alphabetic word after preprocessing",
        "reason_missing": "term not found in cleaned corpus: {items}",
        "reason_low_vocab": "corpus has too little vectorized vocabulary after preprocessing",
        "reason_no_vector": "term lacks a usable vector: {items}",
        "reason_no_path": "path could not be estimated",
    },
    "pt": {
        "language_names": {"pt": "português", "en": "inglês", "es": "espanhol"},
        "report_title": "Caminhos Discursivos - relatório web transitório",
        "language": "Língua",
        "corpora": "Corpora",
        "pairs": "Pares",
        "retention": "Retenção no servidor: nenhuma. Os corpora enviados e arquivos intermediários existiram apenas em uma pasta temporária da requisição.",
        "preprocessing": "Pré-processamento: minúsculas, diacríticos removidos, números removidos, símbolos removidos, stopwords filtradas por língua, apenas tokens alfabéticos estritos.",
        "interpretive": "Nota interpretativa: caminhos geométricos são pontos de partida para interpretação discursiva, não prova de relações causais ou ideológicas.",
        "corpus": "Corpus",
        "clean_tokens": "Tokens limpos",
        "vocabulary": "Vocabulário",
        "model": "Modelo",
        "truncated": "Nota: o corpus foi truncado no limite de tokens da demonstração web.",
        "not_computed": "não calculado",
        "pdf": "PDF",
        "png": "PNG",
        "figure_subtitle": "Corpus: {corpus} | Língua: {language}",
        "zoom": "detalhe",
        "path_axis": "Eixo do caminho",
        "reason_one_word": "cada termo deve virar exatamente uma palavra alfabética após o pré-processamento",
        "reason_missing": "termo não encontrado no corpus limpo: {items}",
        "reason_low_vocab": "o corpus tem vocabulário vetorizado insuficiente após o pré-processamento",
        "reason_no_vector": "termo sem vetor utilizável: {items}",
        "reason_no_path": "o caminho não pôde ser estimado",
    },
    "es": {
        "language_names": {"pt": "portugués", "en": "inglés", "es": "español"},
        "report_title": "Caminhos Discursivos - informe web transitorio",
        "language": "Lengua",
        "corpora": "Corpus",
        "pairs": "Pares",
        "retention": "Retención en el servidor: ninguna. Los corpus subidos y archivos intermedios existieron solo en una carpeta temporal de la solicitud.",
        "preprocessing": "Preprocesamiento: minúsculas, diacríticos eliminados, números eliminados, símbolos eliminados, stopwords filtradas por lengua, solo tokens alfabéticos estrictos.",
        "interpretive": "Nota interpretativa: los caminos geométricos son puntos de partida para la interpretación discursiva, no prueba de relaciones causales o ideológicas.",
        "corpus": "Corpus",
        "clean_tokens": "Tokens limpios",
        "vocabulary": "Vocabulario",
        "model": "Modelo",
        "truncated": "Nota: el corpus fue truncado en el límite de tokens de la demo web.",
        "not_computed": "no calculado",
        "pdf": "PDF",
        "png": "PNG",
        "figure_subtitle": "Corpus: {corpus} | Lengua: {language}",
        "zoom": "detalle",
        "path_axis": "Eje del camino",
        "reason_one_word": "cada término debe quedar como exactamente una palabra alfabética tras el preprocesamiento",
        "reason_missing": "término no encontrado en el corpus limpio: {items}",
        "reason_low_vocab": "el corpus tiene vocabulario vectorizado insuficiente tras el preprocesamiento",
        "reason_no_vector": "término sin vector utilizable: {items}",
        "reason_no_path": "no se pudo estimar el camino",
    },
}


def stable_hash(text):
    data = str(text).encode("utf-8")
    return int(hashlib.blake2b(data, digest_size=8).hexdigest(), 16) & 0xFFFFFFFF


def progress(stage, progress_value, detail=None):
    payload = {"stage": stage, "progress": int(progress_value)}
    if detail:
        payload["detail"] = detail
    print("PROGRESS\t" + json.dumps(payload, ensure_ascii=True), flush=True)


def safe_stem(value):
    stem = Path(value).stem
    stem = SAFE_NAME.sub("_", stem).strip("._-")
    return stem[:80] or "corpus"


def decode_text(path):
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


@dataclass
class VectorSpace:
    words: list
    vectors: np.ndarray
    key_to_index: dict

    @property
    def vector_size(self):
        return int(self.vectors.shape[1])

    def __contains__(self, word):
        return word in self.key_to_index

    def __getitem__(self, word):
        return self.vectors[self.key_to_index[word]]


class VectorStore:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Missing vector store: {self.path}")
        self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.dimension = int(self.metadata("dimension") or 0)
        if self.dimension <= 0:
            raise ValueError(f"Vector store has invalid dimension: {self.path}")

    def metadata(self, key):
        row = self.conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def get_many(self, words):
        unique = sorted(set(words))
        found = {}
        for start in range(0, len(unique), 700):
            chunk = unique[start : start + 700]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(f"SELECT word, vector FROM vectors WHERE word IN ({placeholders})", chunk).fetchall()
            for word, blob in rows:
                vector = np.frombuffer(blob, dtype=np.float32)
                if vector.shape[0] == self.dimension:
                    found[word] = vector.copy()
        return found


def build_corpus_vectors(sentences, vocab, pretrained, dimension):
    if not vocab:
        return None, "empty"

    enough_for_training = len(sentences) >= 3 and sum(len(sentence) for sentence in sentences) >= 20 and len(vocab) >= 5
    if enough_for_training:
        model = Word2Vec(
            sentences=sentences,
            vector_size=dimension,
            window=8,
            min_count=1,
            workers=1,
            sg=1,
            negative=8,
            seed=SEED,
            hashfxn=stable_hash,
            sorted_vocab=1,
            epochs=5,
        )
        for word, vector in pretrained.items():
            if word in model.wv.key_to_index:
                model.wv.vectors[model.wv.key_to_index[word]] = vector
        model.train(sentences, total_examples=len(sentences), epochs=3)
        words = list(model.wv.index_to_key)
        vectors = np.asarray(model.wv.vectors, dtype=np.float32)
        return VectorSpace(words=words, vectors=vectors, key_to_index={word: i for i, word in enumerate(words)}), "corpus_word2vec"

    words = sorted(word for word in vocab if word in pretrained)
    if len(words) < 3:
        return None, "too_little_vectorized_vocabulary"
    vectors = np.vstack([pretrained[word] for word in words]).astype(np.float32)
    return VectorSpace(words=words, vectors=vectors, key_to_index={word: i for i, word in enumerate(words)}), "pretrained_fallback"


@dataclass
class Geometry:
    space: VectorSpace
    mean: np.ndarray
    centered: np.ndarray
    components: np.ndarray
    projected: np.ndarray


def build_geometry(space):
    vectors = np.asarray(space.vectors, dtype=np.float32)
    if vectors.shape[0] < 3:
        return None
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    n_components = min(80, centered.shape[0], centered.shape[1])
    if n_components < 2:
        return None
    pca = PCA(n_components=n_components, svd_solver="full", random_state=SEED)
    projected = pca.fit_transform(centered)
    return Geometry(space=space, mean=mean, centered=centered, components=pca.components_, projected=projected)


def discursive_path(geometry, word1, word2, max_intermediates=14):
    space = geometry.space
    if word1 not in space or word2 not in space:
        return None
    idx1 = space.key_to_index[word1]
    idx2 = space.key_to_index[word2]
    n_components = geometry.projected.shape[1]
    dims = list(range(n_components, 1, -4))
    if 2 not in dims:
        dims.append(2)

    persistence = {}
    all_words = np.asarray(space.words)
    for dim in dims:
        all_r = geometry.projected[:, :dim]
        vec1_r = all_r[idx1]
        vec2_r = all_r[idx2]
        midpoint = (vec1_r + vec2_r) / 2
        dists = np.linalg.norm(all_r - midpoint, axis=1)
        candidate_count = min(80, len(dists))
        nearest = np.argpartition(dists, candidate_count - 1)[:candidate_count]
        nearest = sorted(nearest, key=lambda idx: (float(dists[idx]), str(all_words[idx])))
        path_vec = vec2_r - vec1_r
        path_len = np.linalg.norm(path_vec)
        if path_len < 1e-12:
            continue
        path_dir = path_vec / path_len
        for idx in nearest[:10]:
            word = str(all_words[idx])
            if idx in (idx1, idx2):
                continue
            vec = all_r[idx]
            proj_len = float(np.dot(vec - vec1_r, path_dir))
            if 0 < proj_len < path_len:
                persistence[word] = persistence.get(word, 0) + 1

    min_persistence = max(2, math.ceil(len(dims) * 0.28))
    robust = [word for word, count in persistence.items() if count >= min_persistence]
    positions = {}
    for word in robust:
        idx = space.key_to_index[word]
        totals = []
        for dim in dims:
            all_r = geometry.projected[:, :dim]
            v = all_r[idx]
            v1 = all_r[idx1]
            v2 = all_r[idx2]
            path_vec = v2 - v1
            path_len = np.linalg.norm(path_vec)
            if path_len < 1e-12:
                continue
            path_dir = path_vec / path_len
            totals.append(float(np.dot(v - v1, path_dir) / path_len))
        if totals:
            positions[word] = sum(totals) / len(totals)

    ordered = sorted(positions.items(), key=lambda item: (round(float(item[1]), 12), item[0]))
    intermediates = [word for word, _score in ordered[:max_intermediates]]
    return [word1] + intermediates + [word2]


def ui_text(ui_language):
    return TEXT.get(ui_language, TEXT["en"])


def finite_limits(points, pad_fraction=0.12):
    xmin = float(np.min(points[:, 0]))
    xmax = float(np.max(points[:, 0]))
    ymin = float(np.min(points[:, 1]))
    ymax = float(np.max(points[:, 1]))
    if abs(xmax - xmin) < 1e-9:
        xmin -= 0.5
        xmax += 0.5
    if abs(ymax - ymin) < 1e-9:
        ymin -= 0.5
        ymax += 0.5
    xpad = max((xmax - xmin) * pad_fraction, 0.06)
    ypad = max((ymax - ymin) * pad_fraction, 0.06)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def dense_zoom_box(points):
    if len(points) < 8:
        return None
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    n_core = max(5, min(len(points) - 1, int(math.ceil(len(points) * 0.58))))
    core = points[np.argsort(distances, kind="mergesort")[:n_core]]
    full = finite_limits(points, pad_fraction=0.04)
    xmin, xmax, ymin, ymax = finite_limits(core, pad_fraction=0.28)
    full_width = full[1] - full[0]
    full_height = full[3] - full[2]
    if (xmax - xmin) > full_width * 0.72 and (ymax - ymin) > full_height * 0.72:
        return None
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    max_width = full_width * 0.52
    max_height = full_height * 0.52
    if xmax - xmin > max_width:
        xmin = cx - max_width / 2
        xmax = cx + max_width / 2
    if ymax - ymin > max_height:
        ymin = cy - max_height / 2
        ymax = cy + max_height / 2
    return xmin, xmax, ymin, ymax


def choose_inset_bounds(points, xlim, ylim):
    candidates = [
        (0.57, 0.50, 0.39, 0.43),
        (0.06, 0.50, 0.39, 0.43),
        (0.57, 0.08, 0.39, 0.43),
        (0.06, 0.08, 0.39, 0.43),
    ]
    xmin, xmax = xlim
    ymin, ymax = ylim
    xr = xmax - xmin
    yr = ymax - ymin
    best = candidates[0]
    best_score = None
    for left, bottom, width, height in candidates:
        x0 = xmin + left * xr
        x1 = xmin + (left + width) * xr
        y0 = ymin + bottom * yr
        y1 = ymin + (bottom + height) * yr
        covered = (
            (points[:, 0] >= x0)
            & (points[:, 0] <= x1)
            & (points[:, 1] >= y0)
            & (points[:, 1] <= y1)
        )
        score = int(np.sum(covered)) - (0.2 if bottom > 0.45 else 0.0)
        if best_score is None or score < best_score:
            best_score = score
            best = (left, bottom, width, height)
    return best


def annotate_words(ax, coords, words, fontsize=8):
    x_span = max(float(np.ptp(coords[:, 0])), 1e-6)
    y_span = max(float(np.ptp(coords[:, 1])), 1e-6)
    for index, (word, point) in enumerate(zip(words, coords)):
        offset_y = 0.04 * y_span if index % 2 == 0 else -0.04 * y_span
        offset_x = 0.015 * x_span if index % 3 == 0 else 0
        ax.annotate(
            word,
            xy=(point[0], point[1]),
            xytext=(point[0] + offset_x, point[1] + offset_y),
            ha="center",
            va="center",
            fontsize=fontsize,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "0.82", "lw": 0.45, "alpha": 0.92},
        )


def draw_path(ax, coords, path_words, label_words=True, fontsize=8):
    ax.plot(coords[:, 0], coords[:, 1], color="0.18", linewidth=1.8, marker="o", markersize=4.8)
    for start, end in zip(coords[:-1], coords[1:]):
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={"arrowstyle": "-|>", "color": "0.22", "lw": 0.8, "shrinkA": 7, "shrinkB": 7},
        )
    if label_words:
        annotate_words(ax, coords, path_words, fontsize=fontsize)


def plot_path(space, path_words, title, subtitle, pdf_path, png_path, labels):
    vectors = np.vstack([space[word] for word in path_words]).astype(np.float32)
    if vectors.shape[0] == 2:
        coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        explained = None
    else:
        pca = PCA(n_components=2, svd_solver="full", random_state=SEED)
        coords = pca.fit_transform(vectors - np.mean(vectors, axis=0))
        explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    draw_path(ax, coords, path_words, label_words=True, fontsize=8)
    x0, x1, y0, y1 = finite_limits(coords, pad_fraction=0.18)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.text(0.125, 0.955, title, fontsize=11, ha="left", va="top")
    fig.text(0.125, 0.918, subtitle, fontsize=8, color="0.36", ha="left", va="top")
    if explained is not None:
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    else:
        ax.set_xlabel(labels["path_axis"])
        ax.set_ylabel(" ")
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    zoom_box = dense_zoom_box(coords)
    if zoom_box is not None:
        from matplotlib.patches import Rectangle

        xmin, xmax, ymin, ymax = zoom_box
        rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="0.44",
            linewidth=0.8,
            linestyle="--",
            alpha=0.72,
        )
        ax.add_patch(rect)
        inset = ax.inset_axes(choose_inset_bounds(coords, ax.get_xlim(), ax.get_ylim()))
        draw_path(inset, coords, path_words, label_words=True, fontsize=6.2)
        inset.set_xlim(xmin, xmax)
        inset.set_ylim(ymin, ymax)
        inset.set_title(labels["zoom"], fontsize=7, loc="left", pad=2, color="0.35")
        inset.grid(True, alpha=0.12, linewidth=0.5)
        inset.tick_params(labelsize=6, length=2)
        for spine in inset.spines.values():
            spine.set_edgecolor("0.62")
            spine.set_linewidth(0.6)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.13, top=0.80)
    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(output_dir, report):
    labels = ui_text(report.get("ui_language", "en"))
    localized_language = labels["language_names"].get(report["language"], report["language"])
    path = Path(output_dir) / "caminhos_report.txt"
    lines = []
    lines.append(labels["report_title"])
    lines.append("=" * 48)
    lines.append("")
    lines.append(f"{labels['language']}: {localized_language}")
    lines.append(f"{labels['corpora']}: {len(report['corpora'])}")
    lines.append(f"{labels['pairs']}: {len(report['pairs'])}")
    lines.append("")
    lines.append(labels["retention"])
    lines.append(labels["preprocessing"])
    lines.append(labels["interpretive"])
    lines.append("")
    for corpus in report["corpora"]:
        lines.append("-" * 48)
        lines.append(f"{labels['corpus']}: {corpus['name']}")
        lines.append(
            f"{labels['clean_tokens']}: {corpus['clean_tokens']} | "
            f"{labels['vocabulary']}: {corpus['vocab_size']} | "
            f"{labels['model']}: {corpus['model_mode']}"
        )
        if corpus.get("truncated"):
            lines.append(labels["truncated"])
        lines.append("")
        for result in corpus["results"]:
            pair_label = f"{result['left_original']} -> {result['right_original']}"
            if result["status"] == "ok":
                lines.append(f"{pair_label}: {' -> '.join(result['path'])}")
                lines.append(f"  {labels['pdf']}: {result['pdf']}")
                lines.append(f"  {labels['png']}: {result['png']}")
            else:
                lines.append(f"{pair_label}: {labels['not_computed']} ({result['reason']})")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def analyze(manifest, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    language = manifest["options"]["language"]
    ui_language = manifest["options"].get("uiLanguage", "en")
    labels = ui_text(ui_language)
    pairs = manifest["options"]["pairs"]
    limits = manifest["limits"]
    model_dir = Path(os.environ.get("CAMINHOS_MODEL_DIR") or manifest.get("modelDir") or "models")
    vector_store_path = model_dir / f"cc.{language}.300.sqlite"
    vector_store = VectorStore(vector_store_path)
    dimension = vector_store.dimension

    normalized_pairs = []
    for pair in pairs:
        left = normalize_term(pair["left"])
        right = normalize_term(pair["right"])
        normalized_pairs.append({
            "left_original": pair["left"],
            "right_original": pair["right"],
            "left": left,
            "right": right,
        })

    report = {
        "schema": "caminhos.web.report.v1",
        "language": language,
        "ui_language": ui_language,
        "pairs": normalized_pairs,
        "corpora": [],
    }

    total_corpora = len(manifest["input"]["corpora"])
    for corpus_index, corpus in enumerate(manifest["input"]["corpora"], start=1):
        progress("preprocess", 35 + (corpus_index - 1) * 45 / max(total_corpora, 1), corpus["name"])
        text = decode_text(corpus["path"])
        processed = preprocess_text(text, language, limits["maxTokensPerCorpus"])
        required_terms = [token for pair in normalized_pairs for token in (pair["left"], pair["right"]) if token]
        sentences, capped_vocab = apply_vocab_cap(
            processed["sentences"],
            processed["vocabulary"],
            limits["maxVocabPerCorpus"],
            required_terms,
        )
        vocab_words = set(capped_vocab.keys()) | set(required_terms)
        pretrained = vector_store.get_many(vocab_words)

        progress("train", 42 + (corpus_index - 1) * 45 / max(total_corpora, 1), corpus["name"])
        space, model_mode = build_corpus_vectors(sentences, capped_vocab, pretrained, dimension)
        geometry = build_geometry(space) if space is not None else None

        corpus_report = {
            "name": corpus["originalName"],
            "clean_tokens": int(sum(len(sentence) for sentence in sentences)),
            "raw_tokens": int(processed["raw_tokens"]),
            "vocab_size": int(len(capped_vocab)),
            "truncated": bool(processed["truncated"]),
            "model_mode": model_mode,
            "results": [],
        }

        progress("paths", 52 + (corpus_index - 1) * 45 / max(total_corpora, 1), corpus["name"])
        for pair_index, pair in enumerate(normalized_pairs, start=1):
            result = {
                "left_original": pair["left_original"],
                "right_original": pair["right_original"],
                "left": pair["left"],
                "right": pair["right"],
                "status": "error",
            }
            if not pair["left"] or not pair["right"]:
                result["reason"] = labels["reason_one_word"]
            elif pair["left"] not in capped_vocab or pair["right"] not in capped_vocab:
                missing = [token for token in (pair["left"], pair["right"]) if token not in capped_vocab]
                result["reason"] = labels["reason_missing"].format(items=", ".join(missing))
            elif geometry is None:
                result["reason"] = labels["reason_low_vocab"]
            elif pair["left"] not in geometry.space or pair["right"] not in geometry.space:
                missing = [token for token in (pair["left"], pair["right"]) if token not in geometry.space]
                result["reason"] = labels["reason_no_vector"].format(items=", ".join(missing))
            else:
                path_words = discursive_path(geometry, pair["left"], pair["right"])
                if not path_words:
                    result["reason"] = labels["reason_no_path"]
                else:
                    corpus_stem = safe_stem(corpus["originalName"])
                    pair_stem = f"{pair_index:02d}_{safe_stem(pair['left'])}_to_{safe_stem(pair['right'])}"
                    rel_pdf = f"figures/pdf/{corpus_index:02d}_{corpus_stem}/{pair_stem}.pdf"
                    rel_png = f"figures/png/{corpus_index:02d}_{corpus_stem}/{pair_stem}.png"
                    plot_path(
                        geometry.space,
                        path_words,
                        title=f"{pair['left']} -> {pair['right']}",
                        subtitle=labels["figure_subtitle"].format(
                            corpus=corpus["originalName"],
                            language=labels["language_names"].get(language, language),
                        ),
                        pdf_path=output_dir / rel_pdf,
                        png_path=output_dir / rel_png,
                        labels=labels,
                    )
                    result.update({"status": "ok", "path": path_words, "pdf": rel_pdf, "png": rel_png})
            corpus_report["results"].append(result)
        report["corpora"].append(corpus_report)

    (output_dir / "caminhos_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir, report)
    progress("zip", 88)


def main():
    parser = argparse.ArgumentParser(description="Run transient Caminhos Discursivos analysis.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    analyze(manifest, args.output)


if __name__ == "__main__":
    main()
