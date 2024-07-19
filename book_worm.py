from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_openai_tools_agent, AgentExecutor

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
    def __init__(self):
        self.butils = BookUtils()

        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        self.tools = self._get_bookworm_tools()
        self.prompt = self._get_bookoworm_prompt()

        base_agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)

        # memory = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        self.agent = AgentExecutor(
            agent=base_agent,
            tools=self.tools,
            # verbose=True,
            handle_parsing_errors=True,
            memory=memory,
            max_iterations=100,
        )

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

    def ask_bookworm(self, question, history=list()):
        response = self.agent.invoke({"input": question, "chat_history": history})
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

    book_worm = BookWorm()
    for query in [
        "Do you have the book called: The Wrecker?",
        "What did I just ask you?",
        "What is The Wrecker about?",
        "What is the name of the main character in The Wrecker?",
    ]:
        response = book_worm.ask_bookworm(query, chat_history)
        print(response)
        print(response["output"])
        chat_history.append([response["input"], response["output"]])
