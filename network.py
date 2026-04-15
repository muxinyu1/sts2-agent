from typing import Any
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel


class Response(BaseModel):
    status: str
    error: str | None
    message: str | None

    def is_ok(self):
        return self.status == "ok"

class Proxy(BaseModel):

    base_url: str # format: http://xxxx
    port: int # xxxx

    def _singleplayer_url(self) -> str:
        return f"{self.base_url.rstrip('/') }:{self.port}/api/v1/singleplayer"

    def get(self, params: dict[str, Any]) -> str:
        # GET baseurl:port/api/v1/singleplayer?xxx=yyy&zzz=kkk, return response text
        query = urlencode(params or {}, doseq=True)
        url = self._singleplayer_url()
        if query:
            url = f"{url}?{query}"

        request = Request(url, method="GET")
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    
    def post(self, params: dict[str, Any]) -> Response:
        # POST baseurl:port/api/v1/singleplayer return response
        body = json.dumps(params or {}, ensure_ascii=False).encode("utf-8")
        request = Request(
            self._singleplayer_url(),
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=30) as response:
                raw_text = response.read().decode("utf-8")
        except HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            return Response(status="error", error=error_text or str(exc), message=None)
        except URLError as exc:
            return Response(status="error", error=str(exc.reason), message=None)

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return Response(
                status="error",
                error=f"Invalid JSON response: {raw_text}",
                message=None,
            )

        if isinstance(payload, dict):
            status = payload.get("status")
            if status in {"ok", "error"}:
                return Response(
                    status=status,
                    error=payload.get("error"),
                    message=payload.get("message"),
                )
            return Response(
                status="error",
                error=f"Unexpected response payload: {payload}",
                message=None,
            )

        return Response(
            status="error",
            error=f"Unexpected response type: {type(payload).__name__}",
            message=None,
        )

proxy = Proxy(base_url="TODO", port=1) # TODO