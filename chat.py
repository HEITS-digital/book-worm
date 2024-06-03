import json
import requests

import numpy as np

from gutenberg import Gutenberg
from book_worm import BookWorm


class Chat:
    def __init__(self):
        self.handles_dict = [
            {"name": "search for a book, play or poem", "handler": self.search_book},
            {"name": "search for an author, writer or person", "handler": self.search_book},
            {"name": "search for a genre", "handler": self.search_genre},
            {"name": "more information", "handler": self.more_information},
        ]

        self.bookworm = BookWorm()
        self.guten = Gutenberg()
        handle_names = [item["name"] for item in self.handles_dict]
        self.genres_embeddings = np.array(self.bookworm.embedding_model.encode(handle_names))


    def parse_message(self, message):
        queries_embeddings = np.array(self.bookworm.embedding_model.encode(message))
        similarity = np.dot(queries_embeddings, self.genres_embeddings.T)
        index = np.argmax(similarity)
        return self.handles_dict[index]["handler"](message)


    def search_book(self, message):
        book_info = self._look_for_book_in_library(message)
        return self._format_book_search_results_for_user(book_info)


    def search_genre(self, message):
        # TODO: this should be drawn from our txt file
        return 'search_genre ' + message


    def more_information(self, message):
        return 'more_information ' + message


    def _query_google_books(self, message):
        url = "https://www.googleapis.com/books/v1/volumes"
        params = {'q': message}
        resp = requests.get(url, params=params)
        return json.loads(resp.content)
    

    def _look_for_book_in_library(self, message):
        data = self._query_google_books(message)
        book_info = {}

        for item in data['items']:
            # if there is no such book in english, we skip it
            if item['volumeInfo']['language'] != 'en':
                continue
            # if the book has no authors, we skip it
            authors = item['volumeInfo'].get('authors', None)
            if authors is None:
                continue
            
            title = item['volumeInfo']['title']
            author = authors[0]
            # this info will be used to extract the summary
            book_info[title] = {"google": item['volumeInfo'], "gutenberg": None}
            
            book_meta = list(self.guten.search(f"language:en AND title:{title} AND author: {author}"))
            # if we have no information about the book in the DB we skip it
            if len(book_meta) == 0:
                continue
            book_meta = book_meta[0]
            if "Index" in book_meta['title'][0]:
                continue
            # this info will be used to answer questions about the book
            book_info[title]["gutenberg"] = book_meta

        return book_info

    def _format_book_search_results_for_user(self, book_info):
        if book_info == {}:
            return "I've searched all over the internet and even asked a friend, but I wan't able to find any book on that topic."
        
        first_result = list(book_info.values())[0]
        for book in book_info.values():
            if book["gutenberg"] is not None:
                first_result = book
                break

        google_result = first_result["google"]
        title = google_result['title']
        subtitle = google_result.get('subtitle', None)
        if subtitle is not None:
            title = "-".join([title, subtitle])
        author = google_result["authors"][0]
        summary = google_result["description"]

        if first_result["gutenberg"] == None:
            return f"So ... I found a listing called: `{title}` by {author}. \nUnfortunately, I don't have it in my library. \nHowever ... I can give you a summary of the book if you would like."
        return f"Great Success! I found a listing called: `{title}` by {author}. We also have it in the libray if you would like to know more about it."

    def extract_genres(text):
        genres = []
        supported_genres = [x.strip().lower() for x in open("supported_genres.txt", "r").readlines()]
        for genre in supported_genres:
            if genre in text.lower():
                genres.append(genre)
        return genres