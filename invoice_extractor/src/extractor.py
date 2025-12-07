from tika import parser
from pathlib import Path

# Get the project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

pdf_file = DATA_FOLDER / "STATEMENT_FOR_Chinenye_Augusta_Alokam_821030812.pdf"

parsed = parser.from_file(str(pdf_file))
print(parsed["content"])