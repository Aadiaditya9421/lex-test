"""
knowledge_base/ingest.py
─────────────────────────
Document loader, chunker, embedder and ChromaDB builder for LexAssist AI.

Usage (first time / when docs change):
    python -m knowledge_base.ingest

In graph/nodes.py:
    from knowledge_base.ingest import get_vectorstore
    vectorstore = get_vectorstore()
"""

import sys
import logging
import re
from pathlib import Path
from typing import Optional

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

_SEPARATORS = [
    "\n================================================================================",
    "\n--------------------------------------------------------------------------------",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]
_REQUIRED_META_KEYS = {"title", "category", "source", "year"}


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from document body."""
    text = text.strip()
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    try:
        meta = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()
    except yaml.YAMLError as e:
        log.warning(f"  YAML parse error: {e} — using empty metadata")
        meta = {}
        body = text

    clean_meta = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean_meta[k] = v
        else:
            clean_meta[k] = str(v)

    missing = _REQUIRED_META_KEYS - set(clean_meta.keys())
    if missing:
        log.warning(f"  Missing metadata keys: {missing}")

    return clean_meta, body


def load_documents(docs_dir: Optional[str] = None) -> list[Document]:
    """Load all .txt files from docs_dir, parse frontmatter, return LangChain Documents."""
    docs_path = Path(docs_dir or config.docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(
            f"docs_dir not found: {docs_path}\n"
            f"Ensure knowledge_base/docs/ exists and has .txt files."
        )

    txt_files = sorted(docs_path.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {docs_path}.")

    log.info(f"Loading {len(txt_files)} document(s) from {docs_path}")
    documents: list[Document] = []
    skipped = 0

    for path in txt_files:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            log.warning(f"  Cannot read {path.name}: {e} — skipping")
            skipped += 1
            continue

        if not raw.strip():
            log.warning(f"  {path.name} is empty — skipping")
            skipped += 1
            continue

        meta, body = parse_frontmatter(raw)
        meta["source_file"] = path.name
        meta.setdefault("title", path.stem.replace("_", " ").title())
        meta.setdefault("category", "general")

        word_count = len(body.split())
        documents.append(Document(page_content=body, metadata=meta))
        log.info(f"  Loaded  {path.name:<50} {word_count:>6,} words")

    log.info(f"\nLoad complete: {len(documents)} loaded, {skipped} skipped.")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for RAG retrieval."""
    if not documents:
        raise ValueError("No documents to chunk.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
    )

    all_chunks: list[Document] = []
    for doc_idx, doc in enumerate(documents):
        raw_chunks = splitter.split_documents([doc])
        for chunk_idx, chunk in enumerate(raw_chunks):
            chunk.metadata["chunk_index"] = chunk_idx
            chunk.metadata["total_chunks"] = len(raw_chunks)
            chunk.page_content = _clean_chunk(chunk.page_content)
        all_chunks.extend(raw_chunks)
        log.info(
            f"  Chunked [{doc_idx+1}/{len(documents)}] "
            f"{doc.metadata.get('source_file','?'):<50} → {len(raw_chunks):>3} chunks"
        )

    log.info(f"\nChunking complete: {len(all_chunks)} total chunks from {len(documents)} documents.")
    return all_chunks


def _clean_chunk(text: str) -> str:
    """Remove divider lines and collapse excess whitespace."""
    lines = text.split("\n")
    lines = [l for l in lines if not re.match(r"^[=\-]{10,}\s*$", l)]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model (cached after first call by HuggingFace)."""
    log.info(f"Loading embedding model: {config.embed_model}")
    return HuggingFaceEmbeddings(
        model_name=config.embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def build_vectorstore(
    chunks: Optional[list[Document]] = None,
    docs_dir: Optional[str] = None,
    force_rebuild: bool = False,
) -> Chroma:
    """Embed chunks and persist them to ChromaDB."""
    store_path = Path(config.chroma_persist_dir)

    if store_path.exists() and not force_rebuild:
        existing = get_vectorstore()
        count = existing._collection.count()
        if count > 0:
            log.info(f"ChromaDB already exists ({count} vectors). Use --rebuild to re-ingest.")
            return existing
        else:
            log.warning("ChromaDB exists but is empty — rebuilding.")

    if chunks is None:
        documents = load_documents(docs_dir)
        chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError("No chunks to embed.")

    if store_path.exists() and force_rebuild:
        import shutil
        log.info(f"Deleting existing ChromaDB at {store_path}...")
        shutil.rmtree(store_path)

    embeddings = _get_embeddings()
    log.info(f"Embedding {len(chunks)} chunks — this may take 1-3 minutes...")

    batch_size = 100
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        log.info(f"  Batch {batch_num}/{total_batches}  ({len(batch)} chunks)...")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(store_path),
                collection_name=config.collection_name,
            )
        else:
            vectorstore.add_documents(batch)

    final_count = vectorstore._collection.count()
    log.info(f"\nVectorStore built: {final_count:,} vectors at {store_path}")
    return vectorstore


def get_vectorstore() -> Chroma:
    """Load the persisted ChromaDB vectorstore for query-time retrieval."""
    store_path = Path(config.chroma_persist_dir)

    if not store_path.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {store_path}.\n"
            f"Run first: python -m knowledge_base.ingest"
        )

    embeddings = _get_embeddings()
    vectorstore = Chroma(
        persist_directory=str(store_path),
        embedding_function=embeddings,
        collection_name=config.collection_name,
    )
    count = vectorstore._collection.count()
    log.info(f"Loaded ChromaDB: {count:,} vectors from {store_path}")
    return vectorstore


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LexAssist AI — Knowledge base ingestion")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild ChromaDB")
    parser.add_argument("--docs-dir", type=str, default=None)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("LexAssist AI — Knowledge Base Ingestion")
    log.info("=" * 60)

    documents = load_documents(args.docs_dir)
    chunks = chunk_documents(documents)
    build_vectorstore(chunks=chunks, force_rebuild=args.rebuild)

    log.info("\nIngestion complete. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
