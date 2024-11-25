from naptha_sdk.schemas import AgentConfig
from pydantic import BaseModel

class TaskExecutorPromptSchema(BaseModel):
    task: str
    objective: str

class InputSchema(BaseModel):
    tool_name: str
    tool_input_data: TaskExecutorPromptSchema

class TaskExecutorAgentConfig(AgentConfig):
    user_message_template: str
