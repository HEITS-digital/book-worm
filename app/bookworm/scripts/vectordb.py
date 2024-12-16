import os

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.redis import Redis
from langchain_community.vectorstores import Chroma

from ai_library.services import get_all_chapter_names, get_content_by_article_ids


class VectorDB:
    def __init__(self):
        self.encoder = OpenAIEmbeddings()
        self.redis_url = "redis://localhost:6380"

        current_path = os.path.dirname(__file__)
        self.redis_schema = os.path.join(current_path, "static/redis_schema.yaml")

        self.chapter_names = []
        self.chapters_names_chroma = None
        self.init_titles_chorma()

    # TODO: should do this once every X days and share it between instances
    def init_titles_chorma(self):
        result = get_all_chapter_names()
        self.chapter_names = [r["chapter_name"] for r in result]

        chunk_size = max([len(x) for x in self.chapter_names])
        documents = self.create_documents_from_str_list(self.chapter_names, chunk_size)

        self.chapters_names_chroma = Chroma.from_documents(documents, self.encoder)

    def get_most_similar_chapter_id(self, book_title):
        closest_chapter = self.chapters_names_chroma.similarity_search(book_title, k=1)[0].page_content
        return {"book_id": self.chapter_names.index(closest_chapter) + 1}

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
            documents = self.split_documents(book_text, 2048 * 2)
            vector_db = Redis.from_texts(
                [document.page_content for document in documents],
                self.encoder,
                redis_url=self.redis_url,
                index_name=book_id,
            )
            vector_db.write_schema(self.redis_schema)

        search_result = vector_db.similarity_search(query)

        documents = self.split_documents(search_result, 1024)
        small_chunks_vector_db = Chroma.from_documents(documents, self.encoder)
        search_result = small_chunks_vector_db.similarity_search(query)

        page_contents = [doc.page_content for doc in search_result]
        return page_contents

    def get_book_contents_by_id(self, book_id):
        response = get_content_by_article_ids(article_ids=[book_id])
        return response[0]["text"]

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

    @staticmethod
    def create_documents_from_str_list(str_list, chunk_size):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.create_documents(str_list)
