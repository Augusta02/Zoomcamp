from tika import parser
from pathlib import Path

# Get the project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

pdf_file = DATA_FOLDER / "Alokam_Augusta_DA.pdf"

def extract_text_from_pdf(pdf_file: Path) -> str:
    parsed = parser.from_file(str(pdf_file))
    return parsed.get("content", "")