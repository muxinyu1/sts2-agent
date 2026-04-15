from abc import ABC

from pydantic import BaseModel


class Memory(ABC):
    def toMarkdown(self) -> str:
        return "" # TODO

class LongtermMemory(Memory, BaseModel):
    pass

class ShorttermMemory(Memory, BaseModel):
    pass