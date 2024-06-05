from pydantic import BaseModel

class InputSchema(BaseModel):
    task: str
    objective: str
