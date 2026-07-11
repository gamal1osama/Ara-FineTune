"""
Pydantic schema for the translation task.
Extracted from notebook: Tasks > Translation section.
"""
from pydantic import BaseModel, Field


class Translation(BaseModel):
    translated_title: str = Field(
        ..., min_length=5, max_length=300,
        description="Suggested translated title of the news story.",
    )
    translated_content: str = Field(
        ..., min_length=5,
        description="The Full translated content of the news story.",
    )
