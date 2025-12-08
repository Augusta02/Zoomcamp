from tika import parser
from pathlib import Path
from pydantic_ai import Agent
from openai import OpenAI
import instructor
from dotenv import load_dotenv
import os
from schemas import CVExtraction, validate_cv_extraction
from pydantic import ValidationError
from ingestor import extract_text_from_pdf
load_dotenv()
PDF_FILE = Path(__file__).parent.parent / "data" / "Alokam_Augusta_DA.pdf"

def extract_cv_data(resume_text: str) -> CVExtraction:
    query= Agent(
        model="gpt-4o-mini",
        output_type=CVExtraction,
    )
    response = query.run_sync(resume_text)
    print("CVExtraction generated...")
    return response.output


# user input 
user_input = extract_text_from_pdf(PDF_FILE)
valid_data = validate_cv_extraction(user_input).model_dump_json()
print(valid_data)






# Call OpenAI with structured output using instructor

