from pydantic import BaseModel, Field
from langchain_core.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from typing import List, Optional, Type

from ai_library.services import get_articles


class SearchAuthorInput(BaseModel):
    author_name: str = Field(description="should be the name of a person")


class SearchGenreInput(BaseModel):
    book_genre: str = Field(description="should be a book genre to search for")


class SearchBookTitle(BaseModel):
    book_title: str = Field(description="should be a book title to search for")


class GetDetailsInput(BaseModel):
    book_id: str = Field(description="the id of the book to get information for")
    query: str = Field(description="what to search for inside the book")


class SearchAuthorTool(BaseTool):
    name: str = "search-author-tool"
    description: str = "Look up books or articles by author in the BookWorm library"
    return_direct: bool = False
    args_schema: Type[BaseModel] = SearchAuthorInput

    def _run(self, author_name: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[dict]:
        # External variable return when tool runs
        response = get_articles({"author": author_name})
        return response

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
        response = get_articles({"source_type": book_genre})
        return response

    async def _arun(
        self,
        book_genre: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[dict]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")


class SearchBookTitleTool(BaseTool):
    name: str = "search-book-title-tool"
    description: str = "Look up books by title in the BookWorm library."
    return_direct: bool = False
    args_schema: Type[BaseModel] = SearchBookTitle

    def _run(self, book_title: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[dict]:
        # External variable return when tool runs
        response = get_articles({"title": book_title})
        return response

    async def _arun(
        self,
        book_title: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[dict]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")


class GetDetailsFromBook(BaseTool):
    name: str = "get-details-from-book-tool"
    description: str = "Look up for information inside a book from the BookWorm library based on book_id."
    return_direct: bool = False
    args_schema: Type[BaseModel] = GetDetailsInput

    def _run(
        self,
        book_id: str,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[dict]:
        # External variable return when tool runs
        return self.metadata["vector_db"].get_relevant_text(book_id, query)

    async def _arun(
        self,
        book_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[dict]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")
