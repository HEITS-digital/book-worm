from typing import Optional

from llama_cpp import LogitsProcessorList
from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor


def llamacpp_with_character_level_parser(prompt: str, model, token_enforcer,
                                         character_level_parser: Optional[CharacterLevelParser]) -> str:
    logits_processors: Optional[LogitsProcessorList] = None
    if character_level_parser:
        logits_processors = LogitsProcessorList(
            [build_llamacpp_logits_processor(token_enforcer, character_level_parser)])

    output = model(prompt, temperature=0.0, logits_processor=logits_processors, max_tokens=1024)
    text: str = output['choices'][0]['text']
    return text
