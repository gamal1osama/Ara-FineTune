"""
Pydantic schemas for the news details extraction task.
Extracted from notebook: Tasks > Details Extraction section.
"""
from typing import List, Literal

from pydantic import BaseModel, Field

StoryCategory = Literal[
    "politics", "sports", "art", "technology", "economy",
    "health", "entertainment", "science", "not_specified",
]

EntityType = Literal[
    "person-male", "person-female", "location", "organization",
    "event", "time", "quantity", "money", "product", "law",
    "disease", "artifact", "not_specified",
]


class Entity(BaseModel):
    entity_value: str = Field(
        ..., min_length=1, max_length=100,
        description="The actual name or value of the entity.",
    )
    entity_type: EntityType = Field(
        ...,
        description="The type or category of the recognized entity.",
    )


class NewsDetails(BaseModel):
    story_title: str = Field(
        ..., min_length=5, max_length=300,
        description="A fully informative and seo optimized title of the story.",
    )
    story_keywords: List[str] = Field(
        ..., min_length=1,
        description="Relevant Keywords associated with the story.",
    )
    story_summary: List[str] = Field(
        ..., min_length=1, max_length=5,
        description="Summarized keypoints about the story (1-5 points).",
    )
    story_category: StoryCategory = Field(
        ..., description="Category of the news story.",
    )
    story_entities: List[Entity] = Field(
        ..., min_length=1, max_length=10,
        description="List of mentioned entities in the story.",
    )
