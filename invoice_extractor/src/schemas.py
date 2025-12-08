# Import packages
from pydantic import BaseModel, Field, EmailStr, field_validator, model_validator, HttpUrl
from pydantic_ai import Agent
from typing import Literal, List, Optional
from datetime import datetime, date
import json
from openai import OpenAI
import instructor
from dotenv import load_dotenv
load_dotenv()
import nest_asyncio
nest_asyncio.apply()


class OnlinePresence(BaseModel):
    platform: Literal['LinkedIn', 'GitHub', 'Portfolio', 'Twitter', 'Other']
    url: HttpUrl = Field(..., description="The full URL found in the CV")

class Job(BaseModel):
    company: str 
    role: str = Field(..., description="The role of the job")
    year: Optional[int] = Field(None, description="Duration like '2020-2022' or '2 years'")

class CVExtraction(BaseModel):
    full_name: str = Field(..., description="The candidate's full name")
    email: EmailStr = Field(..., description="Primary Email address")
    phone_number: str = Field(..., description="The phone number of the person")
    location: str = Field(..., description="The location of the person")
    technical_skills: List[str] = Field(..., description="The technical skills of the person")
    soft_skills: List[str] = Field(..., description="The soft skills of the person")
    links: List[OnlinePresence] = Field(..., description="The online presence of the person")
    summary: str = Field(..., description="A brief summary of the candidate's work experience")

    


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
