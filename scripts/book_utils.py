import json
import requests

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from scripts.gutenberg import Gutenberg


class BookUtils:
    def __init__(self):
        self.encoder = OpenAIEmbeddings()
        self.guten = Gutenberg()
        self.vector_db = None

        self.user_id = "my-user" # replace this with a UUID
        self.last_book = None
        self.last_msessage = None
        self.last_user_response = dict()


    def get_relevant_text(self, book_name, query):
        if book_name != self.last_book:
            bookworm_key = self.get_book_id_by_name(book_name)
            book_text = self.get_book_contents_by_id(bookworm_key)
            self.build_db_from_book_text(book_text)
            self.last_book = bookworm_key

        docs = self.vector_db.similarity_search(query)
        page_contents = [doc.page_content for doc in docs]
        return page_contents


    def build_db_from_book_text(self, book_text):
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=512,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.create_documents([book_text])
        self.vector_db = Chroma.from_documents(documents, self.encoder)


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


    def search_author(self, message):
        relevant_books = {
            "on_bookworm": [],
            "not_on_bookworm": []
        }

        for book_meta in self.guten.search(f"language:en AND author: {message}"):
            if "Index" in book_meta['title'][0]:
                continue

            # search for books (need descriptions) using google API
            data = self._query_google_books(book_meta['title'])

            description = ""
            if len(data) > 0:
                description = data[0].get("description")
            
            # this info will be used to answer questions about the book
            book_info = {
                "is_available": True,
                "title": book_meta['title'],
                "authors": message,
                "description": description,
                "genre": book_meta.get("subject", "UNKNOWN"),
                "bookworm_key": book_meta.get("key", None)
            }        

            relevant_books["on_bookworm"].append(book_info)
        
        self.last_message = message
        self.last_user_response = relevant_books
        return relevant_books    
        

    def search_genre(self, message, top_k=3):
        relevant_books = {
            "on_bookworm": [],
            "not_on_bookworm": []
        }

        generator = self.guten.search(f"language:en AND subject: {message}")
        books_of_genre = [next(generator) for _ in range(top_k)]
        
        for book_meta in books_of_genre:
            if "Index" in book_meta['title'][0]:
                continue

            # search for books (need descriptions) using google API
            data = self._query_google_books(book_meta['title'])

            description = ""
            if len(data) > 0:
                description = data[0].get("description")
            
            # this info will be used to answer questions about the book
            book_info = {
                "is_available": True,
                "title": book_meta['title'],
                "authors": book_meta['author'],
                "description": description,
                "genre": book_meta.get("subject", "UNKNOWN"),
                "bookworm_key": book_meta.get("key", None)
            }        

            relevant_books["on_bookworm"].append(book_info)

        self.last_message = message
        self.last_user_response = relevant_books
        return relevant_books    


    def search_book_on_bookworm(self, message):
        relevant_books = {
            "on_bookworm": [],
            "not_on_bookworm": []
        }

        # search for books (need descriptions) using google API
        data = self._query_google_books(message)

        for item in data:
            title = item['title']
            main_author = item.get('authors', [""])[0]

            # check if we have that book in our database
            book_meta = list(self.guten.search(f"language:en AND title:{title} AND author: {main_author}"))
            
            is_in_library = False
            genre = item.get("categories", "UNKNOWN")
            bookworm_key = None
            # if we have no information about the book in the DB we skip it
            if len(book_meta) > 0:
                book_meta = book_meta[0]
                if "Index" not in book_meta['title'][0]:
                    is_in_library = True
                    genre = book_meta.get("subject", genre)
                    bookworm_key = book_meta.get("key", bookworm_key)
                
            # this info will be used to answer questions about the book
            book_info = {
                "title": title,
                "authors": main_author,
                "description": item.get("description", ""),
                "genre": genre,
                "bookworm_key": bookworm_key
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
        params = {'q': message}
        resp = requests.get(url, params=params)
        raw_json = json.loads(resp.content)
        return self._filter_google_books(raw_json)
    

    def _filter_google_books(self, data):
        filtered_data = []
        for item in data.get('items', [{}]):
            volume_info = item.get("volumeInfo", {})

            # if there is no such book in english, we skip it
            if volume_info.get('language', None) != 'en':
                continue
            
            # if the book has no authors, we skip it
            authors = volume_info.get('authors', None)
            if authors is None:
                continue

            filtered_data.append(volume_info)
        return filtered_data
    
