from pathlib import Path
from pydantic_ai import Agent
from dotenv import load_dotenv
from schemas import CVExtraction
from ingestor import extract_text_from_pdf
import json

load_dotenv()

# Get the JSON schema from CVExtraction model
cv_schema = json.dumps(CVExtraction.model_json_schema(), indent=2)

SYSTEM_PROMPT = f"""You are a Senior Technical Recruiter with 15+ years of experience 
screening candidates for top tech companies like Google, Meta, and Amazon.

Here is the JSON schema for the CVExtraction model you must use as context 
for what information is expected:

{cv_schema}

REASONING STEP (Think before extracting):
1. First, scan the resume for GitHub or Portfolio links - these indicate HIGH POTENTIAL candidates
2. For each PAID job, calculate duration in years AND months separately
3. Identify all technical skills mentioned (languages, frameworks, tools, databases)
4. Identify soft skills (leadership, communication, problem-solving)
5. Separate PAID work from VOLUNTEERING/LEADERSHIP roles

Important Distinctions:
- work_experience: ONLY paid employment (jobs, internships, contracts)
  - Use 'years' for full years (e.g., 2020-2023 = 3 years, 0 months)
  - Use 'months' for partial years (e.g., Jan 2023 - June 2023 = 0 years, 6 months)
  
- volunteering: Unpaid roles including:
  - Volunteer positions
  - Leadership roles (student organizations, clubs)
  - Community service
  - Mentorship programs
  - Non-profit work (unless paid)

- If information is missing, use empty lists or "Not provided" """


def extract_cv_data(resume_text: str) -> CVExtraction:
    """Extract structured CV data using Senior Recruiter AI."""
    agent = Agent(
        model="gpt-4o-mini",
        output_type=CVExtraction,
        system_prompt=SYSTEM_PROMPT,
    )
    response = agent.run_sync(resume_text)
    print("CVExtraction generated...")
    return response.output


def process_cv(pdf_path: str | Path) -> CVExtraction:
    """Extract structured CV data from a PDF file.
        
    Returns:
        CVExtraction: Structured CV data with enhanced summary
    """
    # Extract raw text from PDF
    raw_text = extract_text_from_pdf(pdf_path)
    
    # Extract structured data using LLM
    cv_data = extract_cv_data(raw_text)
    
    return cv_data


def main():
    """Main entry point for CV extraction."""
    pdf_file = Path(__file__).parent.parent / "data" / "Alokam_Augusta_DA.pdf"

    cv_data = process_cv(pdf_file)
    
    print(cv_data.model_dump_json(indent=2))
    
    return cv_data


if __name__ == "__main__":
    main()
