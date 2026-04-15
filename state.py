from typing import Any

from pydantic import BaseModel, Field


class State(BaseModel):
    md: str = Field(default_factory=str)
    d: dict[str, Any] = Field(default_factory=dict)