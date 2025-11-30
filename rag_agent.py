"""
Lightweight RAG agent for product datasheets (local upload, N-way comparison).

Features
- PDF/CSV/JSON ingestion with basic Markdown/table extraction
- Chunking with metadata for product/model/version
- OpenAI embeddings (`text-embedding-3-large`)
- Qdrant (local) as vector store with optional sparse/BM25
- Optional CPU rerank (bge cross-encoder)
- GPT-4o generation with source tagging

Notes
- Optional dependencies (Marker, camelot, tabula, OCR) are used if installed; otherwise
  the code falls back to pdfplumber/pymupdf extraction.
- Requires Qdrant running locally (default http://localhost:6333).
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from dotenv import load_dotenv

from dotenv import load_dotenv
load_dotenv(r"D:\python\.env")


# Optional heavy dependencies
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072  # text-embedding-3-large output size


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
    table_flag: bool = False


@dataclass
class RAGConfig:
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str = "./qdrant_local"  # fallback embedded storage if URL unreachable
    collection_prefix: str = "session_"
    top_k: int = 12
    rerank_top_n: int = 4
    table_boost: float = 1.2
    chunk_size: int = 450
    chunk_overlap: int = 50
    max_tokens_prompt: int = 2200
    use_reranker: bool = True
    sparse_enabled: bool = True


class RAGAgent:
    def __init__(self, openai_api_key: Optional[str] = None, config: Optional[RAGConfig] = None):
        self.cfg = config or RAGConfig()
        key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required.")
        self.openai = OpenAI(api_key=key)
        # Prefer HTTP endpoint; fallback to embedded (path) if unreachable
        try:
            self.qdrant = QdrantClient(url=self.cfg.qdrant_url, timeout=3)
            self.qdrant.get_collections()
        except Exception:
            self.qdrant = QdrantClient(path=self.cfg.qdrant_path)
        self.reranker = self._init_reranker() if self.cfg.use_reranker else None

    def _init_reranker(self):
        if CrossEncoder is None:
            return None
        try:
            return CrossEncoder("BAAI/bge-reranker-base")
        except Exception:
            return None

    # ---------------------------- Ingestion ----------------------------
    def ingest_files(
        self,
        files: List[Path],
        session_id: str,
        product_name: Optional[str] = None,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        region: Optional[str] = None,
        replace_existing: bool = False,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
    ) -> None:
        collection = self._session_collection(session_id)
        self._ensure_collection(collection)

        chunks: List[Chunk] = []
        for file_path in files:
            ext = file_path.suffix.lower()
            if replace_existing:
                self._delete_doc(collection, file_path.name)
            if ext != ".pdf":
                raise ValueError(f"Only PDF is supported. Unsupported file type: {file_path}")
            chunks.extend(
                self._process_pdf(
                    file_path,
                    session_id,
                    product_name,
                    model_id,
                    version,
                    region,
                    page_start=page_start,
                    page_end=page_end,
                )
            )

        if not chunks:
            return

        embeddings = self._embed_texts([c.text for c in chunks])
        points = []
        for chunk, emb in zip(chunks, embeddings):
            payload = {**chunk.metadata, "table_flag": chunk.table_flag, "markdown_blob": chunk.text}
            points.append(
                rest.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload=payload,
                )
            )
        self.qdrant.upsert(collection_name=collection, points=points)

    def _process_pdf(
        self,
        path: Path,
        session_id: str,
        product_name: Optional[str],
        model_id: Optional[str],
        version: Optional[str],
        region: Optional[str],
        page_start: Optional[int],
        page_end: Optional[int],
    ) -> List[Chunk]:
        text_blocks = self._pdf_text_blocks(path, page_start=page_start, page_end=page_end)
        table_blocks = self._pdf_tables(path, page_start=page_start, page_end=page_end)

        chunks: List[Chunk] = []
        base_meta = {
            "doc_id": path.name,
            "upload_session": session_id,
            "product_name": product_name,
            "model_id": model_id,
            "version": version,
            "region": region,
        }

        for page_idx, text in text_blocks:
            for chunk_text in chunk_texts(text, self.cfg.chunk_size, self.cfg.chunk_overlap):
                meta = {**base_meta, "page": page_idx + 1, "table_flag": False}
                chunks.append(Chunk(text=chunk_text, metadata=meta, table_flag=False))

        for tbl in table_blocks:
            md = tbl["markdown"]
            table_meta = {
                **base_meta,
                "page": tbl["page"],
                "section_path": tbl.get("section_path"),
                "table_flag": True,
                "col_types": tbl.get("col_types"),
                "units": tbl.get("units"),
            }
            chunks.append(Chunk(text=md, metadata=table_meta, table_flag=True))

        return chunks

    def _process_structured(
        self,
        path: Path,
        session_id: str,
        product_name: Optional[str],
        model_id: Optional[str],
        version: Optional[str],
        region: Optional[str],
    ) -> List[Chunk]:
        base_meta = {
            "doc_id": path.name,
            "upload_session": session_id,
            "product_name": product_name,
            "model_id": model_id,
            "version": version,
            "region": region,
            "table_flag": True,
        }
        if path.suffix.lower() == ".csv":
            if pd is None:
                raise RuntimeError("pandas is required to process CSV files.")
            df = pd.read_csv(path)
            md = df.to_markdown(index=False)
        else:
            data = json.loads(path.read_text())
            if pd is None:
                md = json.dumps(data, indent=2)
            else:
                md = pd.json_normalize(data).to_markdown(index=False)
        return [Chunk(text=md, metadata=base_meta, table_flag=True)]

    def _pdf_text_blocks(self, path: Path, page_start: Optional[int], page_end: Optional[int]) -> List[Tuple[int, str]]:
        blocks: List[Tuple[int, str]] = []
        if fitz is None:
            return blocks
        start_idx = (page_start - 1) if page_start else 0
        end_idx = (page_end - 1) if page_end else None
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                if i < start_idx:
                    continue
                if end_idx is not None and i > end_idx:
                    break
                text = page.get_text("text")
                if text.strip():
                    blocks.append((i, text))
        return blocks

    def _pdf_tables(self, path: Path, page_start: Optional[int], page_end: Optional[int]) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []
        if pdfplumber is None or pd is None:
            return tables
        start_idx = (page_start - 1) if page_start else 0
        end_idx = (page_end - 1) if page_end else None
        with pdfplumber.open(path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                if page_idx < start_idx:
                    continue
                if end_idx is not None and page_idx > end_idx:
                    break
                raw_tables = page.extract_tables()
                for tbl in raw_tables:
                    if not tbl:
                        continue
                    df = pd.DataFrame(tbl[1:], columns=tbl[0] if tbl[0] else None)
                    df = df.dropna(how="all", axis=1)
                    df = df.dropna(how="all", axis=0)
                    if len(df) == 0:
                        continue
                    markdown = df.to_markdown(index=False)
                    tables.append({"page": page_idx + 1, "markdown": markdown})
        return tables

    # ---------------------------- Query ----------------------------
    def query(
        self,
        question: str,
        session_id: str,
        product_names: Optional[List[str]] = None,
        model_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        collection = self._session_collection(session_id)
        top_k = top_k or self.cfg.top_k
        query_vector = self._embed_texts([question])[0]

        scored: List[Dict[str, Any]] = []
        qfilter = self._build_filter_qdrant(product_names, model_ids)
        search_params = rest.SearchParams(
            hnsw_ef=128,
            exact=False,
        )
        res = self.qdrant.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=qfilter,
            search_params=search_params,
        )
        for point in res:
            payload = point.payload or {}
            score = float(point.score or 0.0)
            if payload.get("table_flag"):
                score *= self.cfg.table_boost
            scored.append(
                {"text": payload.get("markdown_blob", payload.get("text", "")) or "", "payload": payload, "score": score}
            )

        reranked = self._rerank(question, scored)
        top_contexts = reranked[: self.cfg.rerank_top_n]
        prompt = build_prompt(question, top_contexts, self.cfg.max_tokens_prompt)
        answer = self._call_llm(prompt)

        return {"answer": answer, "contexts": top_contexts}

    def _build_filter_qdrant(self, product_names: Optional[List[str]], model_ids: Optional[List[str]]):
        must: List[rest.FieldCondition] = []
        if product_names:
            must.append(rest.FieldCondition(key="product_name", match=rest.MatchAny(any=product_names)))
        if model_ids:
            must.append(rest.FieldCondition(key="model_id", match=rest.MatchAny(any=model_ids)))
        return rest.Filter(must=must) if must else None

    def _rerank(self, question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.reranker:
            return sorted(docs, key=lambda x: x["score"], reverse=True)
        pairs = [[question, d["text"]] for d in docs]
        try:
            scores = self.reranker.predict(pairs)
            for doc, s in zip(docs, scores):
                doc["score"] = float(s)
        except Exception:
            pass
        return sorted(docs, key=lambda x: x["score"], reverse=True)

    # ---------------------------- Helpers ----------------------------
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = self.openai.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "제품/부품 데이터시트 근거로만 답변하세요. 출처를 태그로 표시하고, 모르면 모른다고 답하세요."},
            {"role": "user", "content": prompt},
        ]
        resp = self.openai.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.2)
        return resp.choices[0].message.content

    def _ensure_collection(self, name: str) -> None:
        existing = {c.name for c in self.qdrant.get_collections().collections}
        if name in existing:
            return
        vectors_config = rest.VectorParams(size=EMBED_DIM, distance=rest.Distance.COSINE)
        self.qdrant.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
        )

    def _session_collection(self, session_id: str) -> str:
        return f"{self.cfg.collection_prefix}{session_id}"

    def _delete_doc(self, collection: str, doc_id: str) -> None:
        cond = rest.FieldCondition(key="doc_id", match=rest.MatchValue(value=doc_id))
        self.qdrant.delete(collection_name=collection, points_selector=rest.Filter(must=[cond]))


def chunk_texts(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    words = text.split()
    step = max(1, chunk_size - overlap)
    idx = 0
    while idx < len(words):
        window = words[idx : idx + chunk_size]
        yield " ".join(window)
        idx += step


def build_prompt(question: str, contexts: List[Dict[str, Any]], max_tokens: int) -> str:
    lines = ["주어진 컨텍스트만 활용해 답변하세요. 출처를 [문서명, 페이지] 형식의 태그로 표시하세요.", "컨텍스트:"]
    for i, ctx in enumerate(contexts, start=1):
        payload = ctx["payload"]
        doc = payload.get("doc_id", "unknown")
        page = payload.get("page", "?")
        lines.append(f"[{i}] ({doc}, p{page}) {ctx['text']}")
    lines.append(f"\n질문: {question}")
    lines.append("답변:")
    prompt = "\n".join(lines)
    if len(prompt.split()) > max_tokens:
        prompt = " ".join(prompt.split()[:max_tokens])
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="Ingest documents into a session collection")
    ing.add_argument("--session", required=True, help="Session/collection id")
    ing.add_argument("--files", nargs="+", required=True, help="Paths to files (PDF only)")
    ing.add_argument("--product-name")
    ing.add_argument("--model-id")
    ing.add_argument("--version")
    ing.add_argument("--region")
    ing.add_argument("--replace-existing", action="store_true", help="Delete existing doc_id (filename) before ingest")
    ing.add_argument("--page-start", type=int, help="1-based start page to ingest (inclusive)")
    ing.add_argument("--page-end", type=int, help="1-based end page to ingest (inclusive)")
    ing.add_argument("--env-file", default=".env", help="Path to .env file containing OPENAI_API_KEY")

    qry = sub.add_parser("query", help="Query a session collection")
    qry.add_argument("--session", required=True, help="Session/collection id")
    qry.add_argument("--question", required=True)
    qry.add_argument("--product-name", action="append", help="Filter by product name (repeatable)")
    qry.add_argument("--model-id", action="append", help="Filter by model id (repeatable)")
    qry.add_argument("--top-k", type=int, default=12)
    qry.add_argument("--env-file", default=".env", help="Path to .env file containing OPENAI_API_KEY")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_path = Path(getattr(args, "env_file", ".env"))
    if env_path.exists():
        load_dotenv(env_path)
    agent = RAGAgent(config=RAGConfig())
    if args.command == "ingest":
        files = [Path(p) for p in args.files]
        agent.ingest_files(
            files=files,
            session_id=args.session,
            product_name=args.product_name,
            model_id=args.model_id,
            version=args.version,
            region=args.region,
            replace_existing=args.replace_existing,
            page_start=args.page_start,
            page_end=args.page_end,
        )
        print(f"Ingested {len(files)} file(s) into session '{args.session}'.")
    elif args.command == "query":
        res = agent.query(
            question=args.question,
            session_id=args.session,
            product_names=args.product_name,
            model_ids=args.model_id,
            top_k=args.top_k,
        )
        print("Answer:\n", res["answer"])
        print("\nContexts:")
        for ctx in res["contexts"]:
            print(f"- {ctx['payload'].get('doc_id')} p{ctx['payload'].get('page')}: score={ctx['score']:.3f}")
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
