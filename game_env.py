from typing import Any
import json

from pydantic import BaseModel

from state import State
from network import Response, proxy

class GameEnv(BaseModel):
    def post(self, tool_name: str, params: dict[str, Any]):
        final_params = params.copy()
        final_params["action"] = tool_name
        return proxy.post(final_params)
    
    def state(self) -> State:
        md = proxy.get({"format": "markdown"})
        jsn = proxy.get({"format": "json"})
        return State(md=md, d=json.loads(jsn))
    
    def end_turn(self) -> tuple[State, Response]:
        # TODO 结束回合，返回结束回合后的状态
        rsp = proxy.post({"action": "end_turn"})
        return self.state(), rsp
    
game_env_instance = GameEnv()