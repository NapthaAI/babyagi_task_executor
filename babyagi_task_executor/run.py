from babyagi_task_executor.schemas import InputSchema
from babyagi_task_executor.utils import get_logger
from litellm import completion
import yaml


logger = get_logger(__name__)

def run(inputs: InputSchema, *args, **kwargs):
    logger.info(f"Running with inputs {inputs.objective}")
    logger.info(f"Running with inputs {inputs.task}")
    cfg = kwargs["cfg"]
    
    user_prompt = cfg["inputs"]["user_message_template"].replace("{{task}}", inputs.task).replace("{{objective}}", inputs.objective)

    messages = [
        {"role": "system", "content": cfg["inputs"]["system_message"]},
        {"role": "user", "content": user_prompt}
    ]

    result = completion(
        model=cfg["models"]["openai"]["model"],
        messages=messages,
        temperature=cfg["models"]["openai"]["temperature"],
        max_tokens=cfg["models"]["openai"]["max_tokens"],
    ).choices[0].message.content

    logger.info(f"Result: {result}")

    return result


if __name__ == "__main__":
    with open("babyagi_task_executor/component.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    inputs = InputSchema(
        task="Weather pattern between year 1900 and 2000?", 
        objective="Write a blog post about the weather in London."
    )

    run(inputs, cfg=cfg)


