# %%
from langchain import hub

prompt = hub.pull("hwchase17/openai-functions-agent")
# %%
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much"}, {"output": "that's good"})
# %%
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict

formatted_history = messages_to_dict(memory.chat_memory.messages)

# %%
messages_from_dict(formatted_history)


# %%
def _format_history(messages):
    return [
        {
            "type": "human",
            "data": {
                "content": messages[0],
                "additional_kwargs": {},
                "response_metadata": {},
                "type": "human",
                "name": None,
                "id": None,
                "example": False,
            },
        },
        {
            "type": "ai",
            "data": {
                "content": messages[1],
                "additional_kwargs": {},
                "response_metadata": {},
                "type": "ai",
                "name": None,
                "id": None,
                "example": False,
                "tool_calls": [],
                "invalid_tool_calls": [],
                "usage_metadata": None,
            },
        },
    ]


# %%
history = [["test1", "test2"], ["test3", "test4"]]
formatted_history = [
    entry for messages in history for entry in _format_history(messages)
]
list_of_messages = messages_from_dict(formatted_history)
# %%
ConversationBufferMemory(memory_key="chat_history", chat_memory=list_of_messages)
