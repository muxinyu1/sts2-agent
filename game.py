from typing import List

from pydantic import BaseModel



class Round(BaseModel):
    turn_index: int
    actions: List[str]

    def add_action(self, action: str):
        self.actions.append(action)

class Floor(BaseModel):
    turns: List[Round] | None
    summary: str

    def add_turn(self, turn: Round):
        if self.turns:
            self.turns.append(turn)

class Act(BaseModel):
    floors: List[Floor]

    def add_floor(self, floor: Floor):
        self.floors.append(floor)