from pydantic import BaseModel

from state import State


class GameEnv(BaseModel):
    def post(self):
        pass
    
    def state(self) -> State:
        return NotImplemented
    
    def end_turn(self) -> State:
        return NotImplemented