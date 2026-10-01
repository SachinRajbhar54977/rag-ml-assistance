from rag_assistant.config import RAW_DATA_DIR
from rag_assistant.pipeline_registry import PDF_LOADERS, TEXT_SPLITTERS, CHUNKERS

# Load a real document using the registry, not a direct import
loader = PDF_LOADERS["pymupdf"]
docs = loader(RAW_DATA_DIR / "machine_learning.pdf")

# Pick a real page you already know well — page 104 (RuleFit definition)
doc_104 = next(d for d in docs if d.page == 104)

print(f"Testing on page 104, text length: {len(doc_104.text)} chars\n")

for splitter_name, splitter_fn in TEXT_SPLITTERS.items():
    units = splitter_fn(doc_104.text, chunk_size=500)
    print(f"=== splitter: {splitter_name!r} -> {len(units)} unit(s) ===")
    for i, u in enumerate(units[:3]):
        print(f"  unit {i}: {u[:80]!r}")
    print()

# Now pack the paragraph_sentence output using the one available chunker
units = TEXT_SPLITTERS["paragraph_sentence"](doc_104.text, chunk_size=500)
packer = CHUNKERS["fixed_size_char"]
chunks = packer(units, source=doc_104.source, page=doc_104.page, chunk_size=500, chunk_overlap=50)

print(f"=== chunker: 'fixed_size_char' packed {len(units)} units -> {len(chunks)} chunk(s) ===")
for c in chunks:
    print(f"  chunk_index={c.chunk_index}: {c.text[:80]!r}")