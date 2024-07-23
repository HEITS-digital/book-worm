from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages.utils import messages_from_dict

from scripts.book_utils import BookUtils
from scripts.bookworm_tools import (
    GetGenresTool,
    SearchAuthorTool,
    SearchGenreTool,
    SearchBookOnBookwormTool,
    SearchBookOnGoogleTool,
    GetDetailsFromBook,
)


class BookWorm:
    def __init__(self, history=list()):
        self.butils = BookUtils()

        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        self.tools = self._get_bookworm_tools()
        self.prompt = self._get_bookoworm_prompt()

        base_agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)

        if len(history) > 0:
            memory = self._get_memory_from_history(history)
        else:
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )

        self.agent = AgentExecutor(
            agent=base_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            memory=memory,
            max_iterations=100,
        )

    def _get_memory_from_history(self, history):
        formatted_history = [
            entry for messages in history for entry in self._format_history(messages)
        ]
        list_of_messages = messages_from_dict(formatted_history)
        retrieved_chat_history = ChatMessageHistory(messages=list_of_messages)
        return ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=retrieved_chat_history,
            return_messages=True,
        )

    def _format_history(self, messages):
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

    def _get_bookworm_tools(self):
        return [
            GetGenresTool(),
            SearchAuthorTool(metadata={"butils": self.butils}),
            SearchGenreTool(metadata={"butils": self.butils}),
            SearchBookOnBookwormTool(metadata={"butils": self.butils}),
            SearchBookOnGoogleTool(metadata={"butils": self.butils}),
            GetDetailsFromBook(metadata={"butils": self.butils}),
        ]

    def _get_bookoworm_prompt(self):
        prompt = hub.pull("hwchase17/openai-functions-agent")
        sys_message = SystemMessagePromptTemplate.from_template(
            "You are a librarian in the BookWorm library. If a book is not in the library you can only offer a short description."
        )
        prompt.messages[0] = sys_message
        return prompt

    def ask_bookworm(self, question):
        response = self.agent.invoke({"input": question})
        return response


def update_env_vars(env_file_path: str = None):
    import os
    from dotenv import dotenv_values

    if env_file_path:
        env_config = dotenv_values(env_file_path)
        os.environ = {**os.environ, **env_config}


if __name__ == "__main__":
    chat_history = []

    update_env_vars(".env")

    for query in [
        "Do you have the book called: The Wrecker?",
        "What did I just ask you?",
        "What is The Wrecker about?",
        "What is the name of the main character in The Wrecker?",
    ]:
        book_worm = BookWorm(chat_history)
        response = book_worm.ask_bookworm(query)
        print(response)
        print(response["output"])
        chat_history.append([response["input"], response["output"]])
