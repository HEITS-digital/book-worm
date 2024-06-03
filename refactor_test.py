# %%
import os
from dotenv import dotenv_values

from langchain import hub
from langchain.prompts import SystemMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import tool

from scripts.book_utils import BookUtils

from typing import List
# %%
def update_env_vars(env_file_path: str=None):
    if env_file_path:
        env_config = dotenv_values(env_file_path)
        os.environ = {**os.environ, **env_config}
# %%
encoder = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
# %%
butils = BookUtils()
result = butils.search_book_on_bookworm("meditations aurelius")
# %%
class SearchAuthorInput(BaseModel):
    author_name: str = Field(description="should be the name of a person")

class SearchBookInput(BaseModel):
    query: str = Field(description="should be a search query")

class SearchGenreInput(BaseModel):
    book_genre: str = Field(description="should be a book genre to search for")

class GetDetailsInput(BaseModel):
    book_key: str = Field(description="should be the identification key for a book")
    query: str = Field(description="should be a search query for information inside the book")

@tool("get-genres-tool", return_direct=False)
def get_genres() -> List[str]:
    """Returns the list of genres for books available in the BookWorm library."""
    return open("supported_genres.txt", 'r').readlines()

@tool("search-author-tool", args_schema=SearchAuthorInput, return_direct=False)
def search_author(author_name: str) -> List[dict]:
    """Look up books by author in the BookWorm library."""
    return butils.search_author(author_name)

@tool("search-genre-tool", args_schema=SearchGenreInput, return_direct=False)
def search_genre(book_genre: str) -> List[dict]:
    """Look up books from a specific genre in the BookWorm library."""
    return butils.search_genre(book_genre)

@tool("search-book-on-bookworm-tool", args_schema=SearchBookInput, return_direct=False)
def search_book_on_bookworm(query: str) -> List[dict]:
    """Look up books in the BookWorm library. This is preferred when searching using query."""
    return butils.search_book_on_bookworm(query)

@tool("search-book-on-google-tool", args_schema=SearchBookInput, return_direct=False)
def search_book_on_google(query: str) -> List[dict]:
    """Used for looking up books that are not in the BookWorm library. No additional information can be provided about these books."""
    return butils.search_book_on_google(query)

@tool("get-book-details-tool", args_schema=GetDetailsInput, return_direct=False)
def get_book_details(book_key: str, query: str) -> str:
    """Look up for information inside a book from the BookWorm library."""
    return butils.load_book_as_documents(query)
# %%
tools = [get_genres, search_book_on_bookworm, search_book_on_google, search_author, search_genre]

prompt = hub.pull("hwchase17/openai-functions-agent")
sys_message = SystemMessagePromptTemplate.from_template("You are a librarian in the BookWorm library. You are not able to give more information about a book outside the library, except the description.")
prompt.messages[0] = sys_message

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# %%
agent_executor.invoke({"input": "I'm thinking of a book about some speaking animals that deals with political motives. Do you know it?"})
# agent_executor.invoke({"input": "I like science-fiction. Can you recommend any good books?"})
 # %%
agent_executor.invoke({"input": "Do have any book on the count of monte somethin something?"})
# %%
"""TODO
1. can ask questions about the contents of a book
2. has memory of past conversations
"""