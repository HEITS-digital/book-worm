from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
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

        memory = ChatMessageHistory()

        agent_executor = AgentExecutor(agent=base_agent, tools=self.tools, verbose=True)

        self.agent = RunnableWithMessageHistory(
            agent_executor,
            # This is needed because in most real world scenarios, a session id is needed
            # It isn't really used here because we are using a simple in memory ChatMessageHistory
            lambda session_id: memory,
            input_messages_key="input",
            history_messages_key="chat_history",
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

    def ask_bookworm(self, question, session_id):
        response = self.agent.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        return response


def update_env_vars(env_file_path: str = None):
    import os
    from dotenv import dotenv_values

    if env_file_path:
        env_config = dotenv_values(env_file_path)
        os.environ = {**os.environ, **env_config}


if __name__ == "__main__":
    update_env_vars(".env")
    book_worm = BookWorm()

    book_worm.ask_bookworm(
        "I am looking for a book from Robert Louis Stevenson. Do you know any?"
    )
