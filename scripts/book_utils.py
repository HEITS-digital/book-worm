import json
import requests

from scripts.gutenberg import Gutenberg


class BookUtils:
    def __init__(self):
        self.guten = Gutenberg()
        self.user_id = "my-user" # replace this with a UUID
        self.last_msessage = None
        self.last_user_response = None

    def search_author(self, message):
        relevant_books = []

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

            relevant_books.append(book_info)
        
        self.last_message = message
        self.last_user_response = relevant_books
        return relevant_books    
        

    def search_genre(self, message, top_k=3):
        relevant_books = []

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

            relevant_books.append(book_info)

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
    
