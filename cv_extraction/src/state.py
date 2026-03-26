from typing import TypedDict, Optional, List
from pathlib import Path
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from extractor import process_cv 
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv

load_dotenv()

# Get Tavily API key from environment
tavily_api_key = os.getenv("TAVILY_API_KEY")
search = TavilySearch(max_results=5, api_key=tavily_api_key)
pdf_path = Path(__file__).parent.parent / "data" / "Alokam_Augusta_DA.pdf"


class AgentState(TypedDict):
    pdf_path: str
    candidate_data: dict
    github_verification: Optional[str]
    company_verification: Optional[List[str]]  # List since there can be multiple companies
    report: str


# ============ TOOLS ============

@tool
def extract_cv_tool(pdf_path: str) -> dict:
    """Extract structured CV data from a PDF file.
    
    Args:
        pdf_path: Path to the PDF resume file
        
    Returns:
        Dictionary containing extracted CV data with fields like
        full_name, email, technical_skills, work_experience, etc.
    """
    cv_data = process_cv(pdf_path)
    return cv_data.model_dump()


@tool
def verify_github(github_url: str) -> str:
    """Verify a GitHub profile by searching for it online.
    
    Args:
        github_url: URL of the GitHub profile
        
    Returns:
        String with verification results
    """
    results = search.invoke(f"GitHub profile {github_url}")
    if results:
        return f"GitHub verified: {github_url} - Found online presence"
    return f"GitHub not verified: {github_url}"


@tool  
def verify_company(company_name: str) -> str:
    """Verify if a company exists by searching for it online.
    
    Args:
        company_name: Name of the company
        
    Returns:
        String with verification results
    """
    results = search.invoke(f"{company_name} company")
    if results:
        return f"Company verified: {company_name}"
    return f"Company not verified: {company_name}"


# ============ NODES ============

def extract_node(state: AgentState) -> dict:
    """Extract the CV data from the PDF file."""
    print("Extracting CV data...")
    cv_data = extract_cv_tool.invoke({"pdf_path": state["pdf_path"]})
    return {"candidate_data": cv_data}


def github_node(state: AgentState) -> dict:
    """Verify the GitHub URL from extracted CV data."""
    print("Verifying GitHub profile...")
    
    # Get GitHub URL from extracted candidate_data
    links = state["candidate_data"].get("links", [])
    github_links = [link for link in links if link.get("platform") == "GitHub"]
    
    if github_links:
        github_url = github_links[0]["url"]
        result = verify_github.invoke({"github_url": github_url})
        return {"github_verification": result}
    
    return {"github_verification": "No GitHub profile found in CV"}


def company_node(state: AgentState) -> dict:
    """Verify all companies from work experience."""
    print("Verifying companies...")
    
    # Get companies from work_experience
    work_experience = state["candidate_data"].get("work_experience", [])
    
    verifications = []
    for job in work_experience:
        company_name = job.get("company", "")
        if company_name:
            result = verify_company.invoke({"company_name": company_name})
            verifications.append(result)
    
    return {"company_verification": verifications if verifications else ["No companies to verify"]}


def report_node(state: AgentState) -> dict:
    """Generate a final vetting report."""
    print("Generating report...")
    
    candidate = state["candidate_data"]
    
    report = f"""
=====================================
    CV VETTING REPORT
=====================================

Candidate: {candidate.get('full_name', 'N/A')}
Email: {candidate.get('email', 'N/A')}
Location: {candidate.get('location', 'N/A')}

Technical Skills: {', '.join(candidate.get('technical_skills', []))}

Work Experience: {len(candidate.get('work_experience', []))} positions

Verification Results:
------------------------
GitHub: {state.get('github_verification', 'Not checked')}

Companies:
{chr(10).join(state.get('company_verification', ['Not checked']))}

Summary: {candidate.get('summary', 'N/A')}
=====================================
"""
    print(report)
    return {"report": report}


# ============ BUILD GRAPH ============

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("extract", extract_node)
graph.add_node("github", github_node)
graph.add_node("company", company_node)
graph.add_node("report", report_node)

# Define the flow
graph.add_edge(START, "extract")
graph.add_edge("extract", "github")
graph.add_edge("github", "company")
graph.add_edge("company", "report")
graph.add_edge("report", END)

# Compile the graph
app = graph.compile()


# ============ RUN ============

if __name__ == "__main__":
    print("Starting CV Vetting Pipeline...\n")
    
    result = app.invoke({"pdf_path": str(pdf_path)})
    
    print("\nPipeline complete!")