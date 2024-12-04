from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages.utils import messages_from_dict

from .tools import (
    GetGenresTool,
    SearchAuthorTool,
    SearchGenreTool,
    SearchBookOnBookwormTool,
    GetDetailsFromBook,
)
from .vectordb import VectorDB


class Agent:
    def __init__(self):
        self.gutenberg_api = "http://127.0.0.1:8000/api/gutenberg"
        self.llm = ChatOpenAI(model="gpt-4o")
        self.vector_db = VectorDB(gutenberg_api=self.gutenberg_api)
        self.tools = self._get_bookworm_tools()
        self.prompt = self._get_bookoworm_prompt()
        self.base_agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)

    def ask_bookworm(self, question, history=list()):
        if len(history) > 0:
            memory = self._get_memory_from_history(history)
        else:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.agent = AgentExecutor(
            agent=self.base_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            memory=memory,
            max_iterations=100,
        )

        response = self.agent.invoke({"input": question})
        return response

    def _get_bookworm_tools(self):
        return [
            GetGenresTool(),  # TODO: this uses a static file. Need to find a better way
            SearchAuthorTool(metadata={"api_url": f"{self.gutenberg_api}/get-author-by-query/"}),
            SearchGenreTool(metadata={"api_url": f"{self.gutenberg_api}/get-genre-by-query/"}),
            SearchBookOnBookwormTool(metadata={"api_url": f"{self.gutenberg_api}/get-book-by-query/"}),
            GetDetailsFromBook(metadata={"vector_db": self.vector_db}),
        ]

    def _get_bookoworm_prompt(self):
        prompt = hub.pull("hwchase17/openai-functions-agent")
        sys_message = SystemMessagePromptTemplate.from_template(
            "You are a librarian in the BookWorm library. If a book is not in the library you can only offer a short description."
        )
        prompt.messages[0] = sys_message
        return prompt

    def _get_memory_from_history(self, history):
        list_of_messages = messages_from_dict(history)
        retrieved_chat_history = ChatMessageHistory(messages=list_of_messages)
        return ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=retrieved_chat_history,
            return_messages=True,
        )
