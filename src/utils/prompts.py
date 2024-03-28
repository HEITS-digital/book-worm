from pydantic import BaseModel


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

class SummaryAnswerFormat(BaseModel):
    summary: str


def get_summarization_prompt(text_to_summarize: str):
    message = f'Summarize in maxmimum 3 sentences the following text: {text_to_summarize}. You MUST answer using the following json schema: '
    message_with_schema = f'{message}{SummaryAnswerFormat.schema_json()}'
    return {"prompt": get_prompt(message_with_schema), "output_schema": SummaryAnswerFormat.schema()}

def get_prompt(message: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{message} [/INST]'