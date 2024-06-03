import json
import requests

from scripts.gutenberg import Gutenberg


class BookUtils:
    def __init__(self):
        self.guten = Gutenberg()
    

    def search_author(self, author):
        relevant_books = []

        for book_meta in self.guten.search(f"language:en AND author: {author}"):
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
                "more_details": True,
                "title": book_meta['title'],
                "authors": author,
                "description": description,
                "genre": book_meta.get("subject", "UNKNOWN")
            }        

            relevant_books.append(book_info)
        return relevant_books    
        

    def search_genre(self, genre, top_k=3):
        relevant_books = []

        generator = self.guten.search(f"language:en AND subject: {genre}")
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
                "more_details": True,
                "title": book_meta['title'],
                "authors": book_meta['author'],
                "description": description,
                "genre": book_meta.get("subject", "UNKNOWN")
            }        

            relevant_books.append(book_info)
        return relevant_books    


    def search_book(self, message):
        relevant_books = []

        # search for books (need descriptions) using google API
        data = self._query_google_books(message)

        for item in data:
            title = item['title']
            main_author = item.get('authors', [""])[0]

            # check if we have that book in our database
            book_meta = list(self.guten.search(f"language:en AND title:{title} AND author: {main_author}"))
            
            is_in_library = False
            genre = item.get("categories", "UNKNOWN")
            # if we have no information about the book in the DB we skip it
            if len(book_meta) > 0:
                book_meta = book_meta[0]
                if "Index" not in book_meta['title'][0]:
                    is_in_library = True
                    book_meta.get("subject", genre)
                
            # this info will be used to answer questions about the book
            book_info = {
                "is_available": is_in_library,
                "more_details": is_in_library,
                "title": title,
                "authors": main_author,
                "description": item.get("description", ""),
                "genre": genre
            }

            relevant_books.append(book_info)

        return relevant_books


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
    
