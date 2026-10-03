"""Local Qdrant document index. Only curated knowledge/*.md files are indexed."""
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import uuid

import httpx
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge"
INDEX_DIR = Path(os.getenv("WAFER_VECTOR_DIR", str(ROOT / ".wafer_vectors")))
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://127.0.0.1:11434/api/embed")
LOCK = threading.RLock()


def documents(directory=KNOWLEDGE_DIR):
    """Bound chunks, retain source line ranges, and reject symbolic links."""
    chunks = []
    for path in sorted(directory.glob("*.md")):
        if path.is_symlink() or path.stat().st_size > 100_000:
            raise ValueError("Knowledge files must be regular Markdown files under 100 KB.")
        lines = path.read_text(encoding="utf-8").splitlines()
        start, part, length = 1, [], 0
        for number, line in enumerate(lines, 1):
            if len(line) > 1800:
                raise ValueError(f"Split long source lines in {path.name} before indexing.")
            if part and length + len(line) > 1800:
                chunks.append({"source": path.name, "start_line": start,
                               "end_line": number - 1, "text": "\n".join(part)})
                start, part, length = number, [], 0
            part.append(line)
            length += len(line) + 1
        if part and any(part):
            chunks.append({"source": path.name, "start_line": start,
                           "end_line": len(lines), "text": "\n".join(part)})
    return chunks


def fingerprint(chunks):
    return hashlib.sha256(json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


async def embed(texts):
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(EMBED_URL, json={
                "model": EMBED_MODEL, "input": texts, "truncate": False,
                "options": {"num_gpu": int(os.getenv("OLLAMA_NUM_GPU", "0"))},
            })
            response.raise_for_status()
            vectors = response.json()["embeddings"]
        if len(vectors) != len(texts) or not vectors:
            raise ValueError("Wrong embedding count")
        size = len(vectors[0])
        if not size or any(len(v) != size or not all(math.isfinite(x) for x in v)
                           or not any(x != 0 for x in v) for v in vectors):
            raise ValueError("Invalid embedding values")
        return vectors
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, "Embedding unavailable. Check Ollama and OLLAMA_EMBED_MODEL.") from exc


def client_at(path):
    try:
        from qdrant_client import QdrantClient
        return QdrantClient(path=str(path))
    except ImportError as exc:
        raise HTTPException(503, "Install requirements-rag.in before using the knowledge index.") from exc
    except RuntimeError as exc:
        raise HTTPException(503, "Vector index is in use. Run one API worker and stop it before rebuilding.") from exc


def read_manifest():
    path = INDEX_DIR / "manifest.json"
    if not path.is_file():
        raise HTTPException(503, "Knowledge index missing. Run python -m wafer_llm_query.build_knowledge.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not {"collection", "embedding_model", "corpus_hash", "dimensions", "chunks"} <= manifest.keys():
            raise ValueError("Missing index metadata")
    except (ValueError, AttributeError) as exc:
        raise HTTPException(503, "Knowledge index metadata is invalid. Rebuild the index.") from exc
    if manifest["embedding_model"] != EMBED_MODEL or manifest["corpus_hash"] != fingerprint(documents()):
        raise HTTPException(503, "Knowledge index is stale or embedding model changed. Rebuild the index.")
    return manifest


def status():
    try:
        manifest = read_manifest()
        with LOCK:
            client = client_at(INDEX_DIR / "qdrant")
            try:
                if not client.collection_exists(manifest["collection"]):
                    raise HTTPException(503, "Vector collection missing. Rebuild the index.")
            finally:
                client.close()
        return {"ready": True, **manifest, "engine": "Qdrant local"}
    except HTTPException as exc:
        return {"ready": False, "detail": exc.detail, "engine": "Qdrant local"}


def save_index(chunks, vectors):
    from qdrant_client.models import Distance, PointStruct, VectorParams
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    collection = "knowledge_" + uuid.uuid4().hex
    with LOCK:
        client = client_at(INDEX_DIR / "qdrant")
        try:
            client.create_collection(collection, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
            client.upsert(collection, points=[PointStruct(id=i, vector=v, payload=c)
                                             for i, (c, v) in enumerate(zip(chunks, vectors))])
            manifest = {"collection": collection, "embedding_model": EMBED_MODEL,
                        "dimensions": len(vectors[0]), "chunks": len(chunks),
                        "corpus_hash": fingerprint(chunks)}
            temporary = INDEX_DIR / "manifest.tmp"
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(INDEX_DIR / "manifest.json")
        finally:
            client.close()
    return manifest


async def build():
    chunks = documents()
    if not chunks:
        raise ValueError("No knowledge documents found")
    vectors = await embed([c["text"] for c in chunks])
    return await asyncio.to_thread(save_index, chunks, vectors)


def search_vector(vector, manifest, top_k, min_score):
    if len(vector) != manifest["dimensions"]:
        raise HTTPException(503, "Embedding dimension changed. Rebuild the index.")
    with LOCK:
        client = client_at(INDEX_DIR / "qdrant")
        try:
            if not client.collection_exists(manifest["collection"]):
                raise HTTPException(503, "Vector collection missing. Rebuild the index.")
            points = client.query_points(manifest["collection"], query=vector,
                                         limit=top_k, score_threshold=min_score).points
            return [{"id": f"K{i}", "score": round(p.score, 6), **p.payload}
                    for i, p in enumerate(points, 1)]
        finally:
            client.close()


async def retrieve(question, top_k=3, min_score=0.4):
    manifest = await asyncio.to_thread(read_manifest)
    vector = (await embed([question]))[0]
    return await asyncio.to_thread(search_vector, vector, manifest, top_k, min_score)
