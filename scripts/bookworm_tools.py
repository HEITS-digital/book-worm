from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.base import RunnableSerializable
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
    name = "get-genres-tool"
    description = (
        "Returns the list of genres for books available in the BookWorm library."
    )
    return_direct = False

    def _run(
        self, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[str]:
        # External variable return when tool runs
        return open("supported_genres.txt", "r").readlines()

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[str]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")


class SearchAuthorTool(BaseTool):
    name = "search-author-tool"
    description = "Look up books by author in the BookWorm library"
    return_direct = False
    args_schema: Type[BaseModel] = SearchAuthorInput

    def _run(
        self, author_name: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[dict]:
        # External variable return when tool runs
        return self.metadata["butils"].search_author(author_name)

    async def _arun(
        self,
        author_name: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[dict]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")


class SearchGenreTool(BaseTool):
    name = "search-genre-tool"
    description = "Look up books from a specific genre in the BookWorm library."
    return_direct = False
    args_schema: Type[BaseModel] = SearchGenreInput

    def _run(
        self, book_genre: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[dict]:
        # External variable return when tool runs
        return self.metadata["butils"].search_genre(book_genre)

    async def _arun(
        self,
        book_genre: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[dict]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")


class SearchBookOnBookwormTool(BaseTool):
    name = "search-book-on-bookworm-tool"
    description = "Look up books in the BookWorm library. This is preferred when searching using query."
    return_direct = False
    args_schema: Type[BaseModel] = SearchBookInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[dict]:
        # External variable return when tool runs
        return self.metadata["butils"].search_book_on_bookworm(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[dict]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")


class SearchBookOnGoogleTool(BaseTool):
    name = "search-book-on-google-tool"
    description = "Used for looking up books that are not in the BookWorm library. No additional information can be provided about these books."
    return_direct = False
    args_schema: Type[BaseModel] = SearchBookInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[dict]:
        # External variable return when tool runs
        return self.metadata["butils"].search_book_on_google(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[dict]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")


class GetDetailsFromBook(BaseTool):
    name = "get-details-from-book-tool"
    description = "Look up for information inside a book from the BookWorm library."
    return_direct = False
    args_schema: Type[BaseModel] = GetDetailsInput

    def _run(
        self,
        book_name: str,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[dict]:
        # External variable return when tool runs
        return self.metadata["butils"].get_relevant_text(book_name, query)

    async def _arun(
        self,
        book_name: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> List[dict]:
        """Use the tool asynchronously."""
        raise NotImplementedError("Not implemented")
