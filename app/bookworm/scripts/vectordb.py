import os

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.redis import Redis
from langchain_community.vectorstores import Chroma

from ai_library.services import get_content_by_article_ids


class VectorDB:
    def __init__(self):
        self.encoder = OpenAIEmbeddings()
        self.redis_url = "redis://localhost:6380"

        current_path = os.path.dirname(__file__)
        self.redis_schema = os.path.join(current_path, "static/redis_schema.yaml")

    # TODO: replace book_name with book_id
    def get_relevant_text(self, book_id, query):
        try:
            vector_db = Redis.from_existing_index(
                self.encoder,
                index_name=book_id,
                redis_url=self.redis_url,
                schema=self.redis_schema,
            )
        except ValueError:
            book_text = self.get_book_contents_by_id(book_id)
            documents = self.split_documents(book_text, 2048)
            vector_db = Redis.from_texts(
                [document.page_content for document in documents],
                self.encoder,
                redis_url=self.redis_url,
                index_name=book_id,
            )
            vector_db.write_schema(self.redis_schema)

        search_result = vector_db.similarity_search(query)

        documents = self.split_documents(search_result, 256)
        small_chunks_vector_db = Chroma.from_documents(documents, self.encoder)
        search_result = small_chunks_vector_db.similarity_search(query)

        page_contents = [doc.page_content for doc in search_result]
        return page_contents

    def get_book_contents_by_id(self, book_id):
        response = get_content_by_article_ids(id=[book_id])
        return response[0].text

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
