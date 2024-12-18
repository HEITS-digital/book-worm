from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_library.services import get_all_chapter_names, get_content_by_article_ids, get_articles
from bookworm.scripts.utils import get_retriever, update_vectorstore, update_vectorstore_with_chapters


class VectorDB:
    def __init__(self):
        self.chapter_names = []
        self.init_titles()

    # TODO: should do this once every X days and share it between instances
    def init_titles(self):
        result = get_all_chapter_names()
        self.chapter_names = [r["chapter_name"] for r in result]

        # retriever = get_retriever("chapters")
        # update_vectorstore_with_chapters(retriever, self.chapter_names)

    def get_most_similar_chapter_id(self, book_title):
        retriever = get_retriever("chapters")
        search_result = retriever.vectorstore.similarity_search(book_title)
        closest_chapter = search_result[0].page_content

        return {"book_id": self.chapter_names.index(closest_chapter) + 1}

    def get_relevant_text(self, book_id, query):
        retriever = get_retriever("chunks")
        # book_text = self.get_book_contents_by_id(book_id)
        # chapter_name = self.get_book_chapter_by_id(book_id)
        # update_vectorstore(retriever, book_text, chapter_name)

        search_result = retriever.vectorstore.similarity_search(query)

        return search_result

    @staticmethod
    def get_book_contents_by_id(book_id):
        response = get_content_by_article_ids(article_ids=[book_id])
        return response[0]["text"]

    @staticmethod
    def get_book_chapter_by_id(book_id):
        response = get_articles({"id": [book_id]})
        print(f"chapter_name: {response[0]}")
        return response[0]["chapter_name"]

    @staticmethod
    def create_documents_from_str_list(str_list, chunk_size):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.create_documents(str_list)
