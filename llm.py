from pydantic import BaseModel

from context import Context


class LLM(BaseModel):
    
    def make_response(self, system_prompt: str, prompt: str) -> str:
        """
        会抛出异常
        """
        return NotImplemented