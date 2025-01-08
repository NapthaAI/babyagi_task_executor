#!/usr/bin/env python
from dotenv import load_dotenv
from babyagi_task_executor.schemas import InputSchema, TaskExecutorAgentConfig
import json
from naptha_sdk.inference import InferenceClient
from naptha_sdk.schemas import AgentDeployment, AgentRunInput
from naptha_sdk.utils import get_logger
from naptha_sdk.user import sign_consumer_id
from typing import Dict
import asyncio

load_dotenv()

logger = get_logger(__name__)

class TaskExecutorAgent:
    def __init__(self, agent_deployment: AgentDeployment):
        self.agent_deployment = agent_deployment
        self.node = InferenceClient(self.agent_deployment.node)

    async def execute_task(self, inputs: InputSchema):
        if isinstance(self.agent_deployment.config, dict):
            self.agent_deployment.config = TaskExecutorAgentConfig(**self.agent_deployment.config)
        
        user_message_template = "You are given the following task: {{task}}. The task is to accomplish the following objective: {{objective}}."
        user_prompt = user_message_template.replace("{{task}}", inputs["tool_input_data"]["task"]).replace("{{objective}}", inputs["tool_input_data"]["objective"])

        messages = [
            {"role": "system", "content": json.dumps(self.agent_deployment.config.system_prompt)},
            {"role": "user", "content": user_prompt}
        ]

        llm_config = self.agent_deployment.config.llm_config

        input_ = {
            "messages": messages,
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens
        }

        response = await self.node.run_inference(
            input_
        )

        try:
            response_content = response.choices[0].message.content
            return response_content
        
        except Exception as e:
            logger.error(f"Failed to parse response: {response}. Error: {e}")
            return

async def run(module_run: Dict, *args, **kwargs):
    module_run = AgentRunInput(**module_run)
    logger.info(f"Running with inputs {module_run.inputs['tool_input_data']}")

    task_executor_agent = TaskExecutorAgent(module_run.deployment)
    method = getattr(task_executor_agent, module_run.inputs['tool_name'], None)

    return await method(module_run.inputs)


if __name__ == "__main__":
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    import os

    naptha = Naptha()

    # Configs
    deployment = asyncio.run(setup_module_deployment("agent", "babyagi_task_executor/configs/deployment.json", node_url = os.getenv("NODE_URL")))
    deployment = AgentDeployment(**deployment.model_dump())
    print("BabyAGI Task Executor Deployment:", deployment)

    input_params: Dict = {
        "tool_name": "execute_task",
        "tool_input_data": {
            "objective": "Write a blog post about the weather in London.",
            "task": "Weather pattern between year 1900 and 2000?"
        }
    }

    agent_run: Dict = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = asyncio.run(run(agent_run))
    logger.info(f"Final Response: {response}")
