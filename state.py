from typing import Any, Dict

from pydantic import BaseModel, Field


class State(BaseModel):
    md: str = Field(default_factory=str)
    d: Dict[str, Any] = Field(default_factory=Dict)