import json
import requests

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.redis import Redis
from langchain_community.vectorstores import Chroma

from scripts.gutenberg import Gutenberg


class BookUtils:
    def __init__(self):
        self.encoder = OpenAIEmbeddings()
        self.guten = Gutenberg()

        self.user_id = "my-user"  # replace this with a UUID
        self.last_book = None
        self.last_message = None
        self.last_user_response = dict()
        self.redis_url = "redis://localhost:6379"
        self.redis_schema = "redis_schema.yaml"

    def get_relevant_text(self, book_name, query):
        bookworm_key = self.get_book_id_by_name(book_name)

        try:
            vector_db = Redis.from_existing_index(
                self.encoder,
                index_name=bookworm_key,
                redis_url=self.redis_url,
                schema=self.redis_schema,
            )
        except ValueError:
            book_text = self.get_book_contents_by_id(bookworm_key)
            documents = self.split_documents(book_text, 2048)
            vector_db = Redis.from_texts(
                [document.page_content for document in documents],
                self.encoder,
                redis_url=self.redis_url,
                index_name=bookworm_key,
            )
            vector_db.write_schema(self.redis_schema)

        search_result = vector_db.similarity_search(query)

        documents = self.split_documents(search_result, 256)
        small_chunks_vector_db = Chroma.from_documents(documents, self.encoder)
        search_result = small_chunks_vector_db.similarity_search(query)

        page_contents = [doc.page_content for doc in search_result]
        return page_contents

    @staticmethod
    def split_documents(book_text, chunk_size):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        if type(book_text) == str:
            documents = text_splitter.create_documents([book_text])
        else:
            documents = text_splitter.split_documents(book_text)

        return documents

    def get_book_contents_by_id(self, bookworm_key):
        self.guten.download(bookworm_key)
        book_text = next(self.guten.text(bookworm_key))
        return book_text

    def get_book_id_by_name(self, book_name):
        for book in self.last_user_response.get("on_bookworm", []):
            if book["title"] == book_name:
                return book["bookworm_key"]

        self.search_book_on_bookworm(book_name)

        for book in self.last_user_response.get("on_bookworm", []):
            if book["title"] == book_name:
                return book["bookworm_key"]

    def search_author(self, message, top_k=3):
        relevant_books = {"on_bookworm": [], "not_on_bookworm": []}

        search_result = self.guten.search(f"language:en AND author: {message}")

        found_books = 0
        for book_meta in search_result:
            if "Index" in book_meta["title"][0]:
                continue

            # search for books (need descriptions) using google API
            data = self._query_google_books(book_meta["title"])

            description = ""
            if len(data) > 0:
                description = data[0].get("description")

            # this info will be used to answer questions about the book
            book_info = {
                "is_available": True,
                "title": book_meta["title"],
                "authors": message,
                "description": description,
                "genre": book_meta.get("subject", "UNKNOWN"),
                "bookworm_key": book_meta.get("key", None),
            }

            relevant_books["on_bookworm"].append(book_info)
            found_books += 1

            if found_books == top_k:
                break

        self.last_message = message
        self.last_user_response = relevant_books
        return relevant_books

    def search_genre(self, message, top_k=3):
        relevant_books = {"on_bookworm": [], "not_on_bookworm": []}

        generator = self.guten.search(f"language:en AND subject: {message}")
        books_of_genre = [next(generator) for _ in range(top_k)]

        for book_meta in books_of_genre:
            if "Index" in book_meta["title"][0]:
                continue

            # search for books (need descriptions) using google API
            data = self._query_google_books(book_meta["title"])

            description = ""
            if len(data) > 0:
                description = data[0].get("description")

            # this info will be used to answer questions about the book
            book_info = {
                "is_available": True,
                "title": book_meta["title"],
                "authors": book_meta["author"],
                "description": description,
                "genre": book_meta.get("subject", "UNKNOWN"),
                "bookworm_key": book_meta.get("key", None),
            }

            relevant_books["on_bookworm"].append(book_info)

        self.last_message = message
        self.last_user_response = relevant_books
        return relevant_books

    def search_book_on_bookworm(self, message):
        relevant_books = {"on_bookworm": [], "not_on_bookworm": []}

        # search for books (need descriptions) using google API
        data = self._query_google_books(message)

        for item in data:
            title = item["title"]
            main_author = item.get("authors", [""])[0]

            # check if we have that book in our database
            book_meta = list(
                self.guten.search(
                    f"language:en AND title:{title} AND author: {main_author}"
                )
            )

            is_in_library = False
            genre = item.get("categories", "UNKNOWN")
            bookworm_key = None
            # if we have no information about the book in the DB we skip it
            if len(book_meta) > 0:
                book_meta = book_meta[0]
                if "Index" not in book_meta["title"][0]:
                    is_in_library = True
                    genre = book_meta.get("subject", genre)
                    bookworm_key = book_meta.get("key", bookworm_key)

            # this info will be used to answer questions about the book
            book_info = {
                "title": title,
                "authors": main_author,
                "description": item.get("description", ""),
                "genre": genre,
                "bookworm_key": bookworm_key,
            }

            if is_in_library:
                relevant_books["on_bookworm"].append(book_info)
            else:
                relevant_books["not_on_bookworm"].append(book_info)

        self.last_message = message
        self.last_user_response = relevant_books
        return relevant_books["on_bookworm"]

    def search_book_on_google(self, message):
        if self.last_message == message:
            return self.last_user_response

        self.search_book_on_bookworm(message)
        return self.last_user_response

    def _query_google_books(self, message):
        url = "https://www.googleapis.com/books/v1/volumes"
        params = {"q": message}
        resp = requests.get(url, params=params)
        raw_json = json.loads(resp.content)
        return self._filter_google_books(raw_json)

    @staticmethod
    def _filter_google_books(data):
        filtered_data = []
        for item in data.get("items", [{}]):
            volume_info = item.get("volumeInfo", {})

            # if there is no such book in english, we skip it
            if volume_info.get("language", None) != "en":
                continue

            # if the book has no authors, we skip it
            authors = volume_info.get("authors", None)
            if authors is None:
                continue

            filtered_data.append(volume_info)
        return filtered_data
