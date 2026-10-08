
from pathlib import Path
import chromadb

# Find the ResQNet root directory automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRONTEND_DIR = PROJECT_ROOT / "frontend"
BACKEND_DIR = PROJECT_ROOT / "backend"

DB_DIR = PROJECT_ROOT / "resqnet-ai" / "chroma_db"

# Use the same collection name as rag/query.py
db = chromadb.PersistentClient(path=str(DB_DIR))
collection = db.get_or_create_collection(name="resqnet_docs")

# Source code and documentation file types
EXTENSIONS = {
    ".java", ".jsx", ".js", ".tsx", ".ts",
    ".css", ".html", ".json", ".xml",
    ".properties", ".yml", ".yaml",
    ".sql", ".md", ".txt"
}

# Ignore generated or dependency folders
IGNORED_DIRS = {
    "node_modules", "target", "build", "dist",
    ".git", ".idea", ".vscode", "venv",
    "__pycache__", "chroma_db"
}

def split_text(text, size=1800, overlap=250):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks

def should_ignore(file):
    return any(part in IGNORED_DIRS for part in file.parts)

def ingest():
    indexed_files = 0
    indexed_chunks = 0
    errors = 0

    for source_dir in [FRONTEND_DIR, BACKEND_DIR]:
        if not source_dir.exists():
            print(f"Folder not found: {source_dir}")
            continue

        print(f"\nScanning: {source_dir}")

        for file in source_dir.rglob("*"):
            if not file.is_file():
                continue

            if file.suffix.lower() not in EXTENSIONS:
                continue

            if should_ignore(file):
                continue

            try:
                content = file.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

                if not content.strip():
                    continue

                chunks = split_text(content)
                relative_path = file.relative_to(PROJECT_ROOT).as_posix()

                for i, chunk in enumerate(chunks):
                    doc_id = f"{relative_path}::chunk_{i}"

                    # Chroma uses its default embedding function.
                    # query.py must use query_texts with this same collection.
                    collection.upsert(
                        ids=[doc_id],
                        documents=[chunk],
                        metadatas=[{
                            "source": relative_path,
                            "chunk": i
                        }]
                    )

                    indexed_chunks += 1

                indexed_files += 1
                print(f"Indexed: {relative_path} ({len(chunks)} chunks)")

            except Exception as error:
                errors += 1
                print(f"Error reading {file}: {error}")

    print("\n========== INDEXING SUMMARY ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Files indexed: {indexed_files}")
    print(f"Chunks stored: {indexed_chunks}")
    print(f"Errors: {errors}")
    print(f"Total documents in collection: {collection.count()}")
    print("======================================")

if __name__ == "__main__":
    ingest()
