# Import packages
from pydantic import BaseModel, Field, EmailStr, model_validator, HttpUrl
from typing import Literal, List, Optional
from dotenv import load_dotenv
load_dotenv()
import nest_asyncio
nest_asyncio.apply()


class OnlinePresence(BaseModel):
    platform: Literal['LinkedIn', 'GitHub', 'Portfolio', 'Twitter', 'Other']
    url: HttpUrl = Field(..., description="The full URL found in the CV")

class Job(BaseModel):
    company: str = Field(..., description="The company name")
    role: str = Field(..., description="The role/title of the job")
    years: Optional[int] = Field(0, description="Number of full years in this role")
    months: Optional[int] = Field(0, description="Additional months beyond full years (0-11)")


class Volunteer(BaseModel):
    organization: Optional[str] = Field(None, description="The organization name")
    role: Optional[str] = Field(None, description="The volunteer role or position")


class CVExtraction(BaseModel):
    full_name: str = Field(..., description="The candidate's full name")
    email: EmailStr = Field(..., description="Primary Email address")
    phone_number: str = Field(..., description="The phone number of the person")
    location: str = Field(..., description="The location of the person")
    technical_skills: List[str] = Field(..., description="The technical skills of the person")
    soft_skills: List[str] = Field(..., description="The soft skills of the person")
    work_experience: List[Job] = Field(..., description="List of paid jobs with company, role, and years - NOT volunteering")
    volunteering: List[Volunteer] = Field(default_factory=list, description="List of volunteer/leadership experiences with organization and role")
    links: List[OnlinePresence] = Field(..., description="The online presence of the person")
    summary: str = Field(..., description="A brief summary of the candidate's work experience")

    @model_validator(mode='after')
    def enhance_summary(self):
        """Add HIGH POTENTIAL tag and years/months of experience to summary."""
        # Check for GitHub or Portfolio links
        high_potential_platforms = {"GitHub", "Portfolio"}
        is_high_potential = any(
            link.platform in high_potential_platforms 
            for link in self.links
        )
        
        # Calculate total experience in months, then convert
        total_months = 0
        for job in self.work_experience:
            job_years = job.years or 0
            job_months = job.months or 0
            total_months += (job_years * 12) + job_months
        
        # Convert total months to years and round up
        # e.g., 2 years 5 months → 3 years, 6 months → 1 year
        total_years = total_months // 12
        remaining_months = total_months % 12
        
        # Round up: if any remaining months, add 1 year
        if remaining_months > 0:
            total_years += 1
        
        # Format experience string (always in years now)
        experience_str = f"{total_years} year" if total_years == 1 else f"{total_years} years"
        
        # Build prefix
        prefix_parts = []
        if is_high_potential:
            prefix_parts.append("[HIGH POTENTIAL]")
        prefix_parts.append(f"{experience_str} experience")
        
        # Prepend to summary if not already prefixed
        prefix = " ".join(prefix_parts)
        if not self.summary.startswith("["):
            self.summary = f"{prefix} {self.summary}"
        
        return self

    


# Define a function to statement extraction
def validate_cv_extraction(cv_extraction: str):
    """Validate user input from a JSON string and return a cv_extraction 
    instance if valid."""
    try:
        result = CVExtraction.model_validate_json(cv_extraction)
        print("cv_extraction validated...")
        return result
    except Exception as e:
        print(f" Unexpected error: {e}")
        return None
