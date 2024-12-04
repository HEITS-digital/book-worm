import os
import requests

from pydantic import BaseModel, Field
from langchain_core.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from typing import List, Optional, Type


class SearchAuthorInput(BaseModel):
    author_name: str = Field(description="should be the name of a person")


class SearchGenreInput(BaseModel):
    book_genre: str = Field(description="should be a book genre to search for")


class SearchBookInput(BaseModel):
    query: str = Field(description="should be a search query")


class GetDetailsInput(BaseModel):
    book_name: str = Field(description="the name of the book to get information from")
    query: str = Field(description="what to search for inside the book")


class GetGenresTool(BaseTool):
    name: str = "get-genres-tool"
    description: str = "Returns the list of genres for books available in the BookWorm library."
    return_direct: bool = False

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[str]:
        # External variable return when tool runs
        current_path = os.path.dirname(__file__)
        supported_genres_path = os.path.join(current_path, "static/supported_genres.txt")
        return open(supported_genres_path, "r").readlines()

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[str]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")


class SearchAuthorTool(BaseTool):
    name: str = "search-author-tool"
    description: str = "Look up books by author in the BookWorm library"
    return_direct: bool = False
    args_schema: Type[BaseModel] = SearchAuthorInput

    def _run(self, author_name: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[dict]:
        # External variable return when tool runs
        response = requests.get(self.metadata["api_url"], {"query": author_name})
        return response.text

    async def _arun(
        self,
        author_name: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[dict]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")


class SearchGenreTool(BaseTool):
    name: str = "search-genre-tool"
    description: str = "Look up books from a specific genre in the BookWorm library."
    return_direct: bool = False
    args_schema: Type[BaseModel] = SearchGenreInput

    def _run(self, book_genre: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[dict]:
        # External variable return when tool runs
        response = requests.get(self.metadata["api_url"], {"query": book_genre})
        return response.text

    async def _arun(
        self,
        book_genre: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[dict]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")


class SearchBookOnBookwormTool(BaseTool):
    name: str = "search-book-on-bookworm-tool"
    description: str = "Look up books in the BookWorm library. This is preferred when searching using query."
    return_direct: bool = False
    args_schema: Type[BaseModel] = SearchBookInput

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[dict]:
        # External variable return when tool runs
        response = requests.get(self.metadata["api_url"], {"query": query})
        return response.text

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[dict]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")


class GetDetailsFromBook(BaseTool):
    name: str = "get-details-from-book-tool"
    description: str = "Look up for information inside a book from the BookWorm library."
    return_direct: bool = False
    args_schema: Type[BaseModel] = GetDetailsInput

    def _run(
        self,
        book_name: str,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[dict]:
        # External variable return when tool runs
        return self.metadata["vector_db"].get_relevant_text(book_name, query)

    async def _arun(
        self,
        book_name: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[dict]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")
