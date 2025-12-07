# Import packages
from pydantic import BaseModel, Field, EmailStr, field_validator, model_validator
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


class Transactions(BaseModel):
    transaction_date: date = Field(..., alias="transactionDate", description="The date of the transaction")
    description: str = Field(..., alias="details", description="The description of the transaction")
    amount: float = Field(..., description="The amount of the transaction")
    transaction_type: Optional[Literal['debit', 'credit']] = Field(None, description="The type of the transaction")
    balance: float = Field(..., description="The balance of the account after the transaction")

class StatementExtraction(BaseModel):
    bank_name: str = Field(..., alias="bankName", description="The name of the bank")
    account_number: str = Field(..., alias="accountNumber", description="The account number")
    currency: str = Field(..., description="The currency of the account")
    start_date: Optional[date] = Field(None, alias='startDate', description="The start date of the statement")
    end_date: Optional[date] = Field(None, alias='endDate', description="The end date of the statement")
    opening_balance: Optional[float] = Field(None, alias='openingBalance', description="The opening balance of the account")
    closing_balance: Optional[float] = Field(None, alias='closingBalance', description="The closing balance of the account")
    total_debit: Optional[float] = Field(None, alias='totalDebit', description="The total debit of the account")
    total_credit: Optional[float] = Field(None, alias='totalCredit', description="The total credit of the account")
    transactions: List[Transactions] = Field(..., description="The transactions in the statement")

    @model_validator(mode='after')
    def calculate_transaction_types(self):
        txns = self.transactions
        for i, line in enumerate(txns):
            if i > 0:
                previous_balance = txns[i-1].balance
            elif self.opening_balance is not None:
                previous_balance = self.opening_balance
            else:
                # Skip first transaction if no opening balance
                continue
            current_balance = line.balance
            if current_balance >= previous_balance:
                line.transaction_type = 'credit'
            else:
                line.transaction_type = 'debit'
        return self


# Define a function to statement extraction
def validate_statement(statement: str):
    """Validate user input from a JSON string and return a StatementExtraction 
    instance if valid."""
    try:
        result = StatementExtraction.model_validate_json(statement)
        print("statement validated...")
        return result
    except Exception as e:
        print(f" Unexpected error: {e}")
        return None



valid_data = validate_statement(statement_extraction).model_dump_json()
print(valid_data)