from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages.utils import messages_from_dict

from .tools import (
    GetBookDetailsByID,
)
from .vectordb import VectorDB


class Agent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.vector_db = VectorDB()
        self.tools = self._get_bookworm_tools()
        self.prompt = self._get_bookoworm_prompt()
        self.base_agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)

    def ask_bookworm(self, question, history=list()):
        if len(history) > 0:
            memory = self._get_memory_from_history(history)
        else:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        agent = AgentExecutor(
            agent=self.base_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            memory=memory,
            max_iterations=100,
        )

        response = agent.invoke({"input": question})
        return response

    def _get_bookworm_tools(self):
        return [
            GetBookDetailsByID(metadata={"vector_db": self.vector_db}),
        ]

    def _get_bookoworm_prompt(self):
        prompt = hub.pull("hwchase17/openai-functions-agent")
        sys_message = SystemMessagePromptTemplate.from_template(
            "You help people learn AI. You NEVER answer questions without providing a source from a article or book."
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
