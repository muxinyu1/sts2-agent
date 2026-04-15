from typing import Any
import json

from pydantic import BaseModel

from state import State
from network import Proxy, Response

class GameEnv(BaseModel):

    proxy: Proxy | None

    def insert_proxy(self, proxy: Proxy):
        self.proxy = proxy

    def post(self, tool_name: str, params: dict[str, Any]):
        assert self.proxy
        final_params = params.copy()
        final_params["action"] = tool_name
        return self.proxy.post(final_params)
    
    def state(self) -> State:
        assert self.proxy
        md = self.proxy.get({"format": "markdown"})
        jsn = self.proxy.get({"format": "json"})
        return State(md=md, d=json.loads(jsn))
    
    def end_turn(self) -> tuple[State, Response]:
        # TODO 结束回合，返回结束回合后的状态
        assert self.proxy
        rsp = self.proxy.post({"action": "end_turn"})
        return self.state(), rsp
    
game_env_instance = GameEnv(proxy=None)