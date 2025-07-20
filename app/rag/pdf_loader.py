from langchain_core.tools import tool
from pathlib import Path
from typing import List
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK = 800
OVERLAP = 200

def extract_chunks(pdf: Path) -> List[str]:
    reader = PdfReader(str(pdf))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK,
        chunk_overlap=OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)
