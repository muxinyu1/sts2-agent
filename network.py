from pydantic import BaseModel

class Response(BaseModel):
    pass

class SingleResponse(Response):
    status: str
    error: str | None
    message: str | None

    def is_ok(self):
        return self.status == "ok"

class MultipleResponse(Response):
    pass