from pathlib import Path
from pydantic_ai import Agent
from dotenv import load_dotenv
from schemas import CVExtraction
from ingestor import extract_text_from_pdf
import json
import os
import psycopg2
from psycopg2.extras import Json

load_dotenv()

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

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


def save_to_postgres(cv_data: CVExtraction) -> int:
    """Save CV extraction data to PostgreSQL.
    
    Returns:
        int: The ID of the inserted record
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        insert_query = """
            INSERT INTO cv_extractions (
                full_name, email, phone_number, location,
                technical_skills, soft_skills,
                work_experience, volunteering, links, summary
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id;
        """
        
        # Convert nested objects to JSON-serializable format
        work_exp_json = [job.model_dump() for job in cv_data.work_experience]
        volunteering_json = [vol.model_dump() for vol in cv_data.volunteering]
        links_json = [{"platform": link.platform, "url": str(link.url)} for link in cv_data.links]
        
        cursor.execute(insert_query, (
            cv_data.full_name,
            cv_data.email,
            cv_data.phone_number,
            cv_data.location,
            cv_data.technical_skills,          
            cv_data.soft_skills,               
            Json(work_exp_json),               
            Json(volunteering_json),           
            Json(links_json),                 
            cv_data.summary
        ))
        
        record_id = cursor.fetchone()[0]
        conn.commit()
        print(f"CV data saved to PostgreSQL with ID: {record_id}")
        return record_id
        
    except Exception as e:
        conn.rollback()
        print(f"Error saving to PostgreSQL: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def main():
    """Main entry point for CV extraction."""
    pdf_file = Path(__file__).parent.parent / "data" / "Alokam_Augusta_DA.pdf"
    output_file = Path(__file__).parent.parent / "output.json"

    cv_data = process_cv(pdf_file)
    
    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cv_data.model_dump_json(indent=2))
    print(f"Results saved to {output_file}")
    
    # Save to PostgreSQL
    save_to_postgres(cv_data)
    
    return cv_data


if __name__ == "__main__":
    main()
