import numpy as np
from book_worm import BookWorm


def search_book(message):
    return 'search_book ' + message


def search_author(message):
    return 'search_author ' + message


def search_genre(message):
    return 'search_genre ' + message


def suggest_genre(message):
    return 'suggest_genre ' + message


def more_information(message):
    return 'more_information ' + message


handles_dict = [
    {"name": "search for a book, play or poem", "handler": search_book},
    {"name": "search for an author, writer or person", "handler": search_author},
    {"name": "search for a genre", "handler": search_genre},
    {"name": "suggest genre", "handler": suggest_genre},
    {"name": "more information", "handler": more_information},
]


class Chat:
    def __init__(self):
        self.bookworm = BookWorm()
        handle_names = [item["name"] for item in handles_dict]
        self.genres_embeddings = np.array(self.bookworm.embedding_model.encode(handle_names))

    def parse_message(self, message):
        queries_embeddings = np.array(self.bookworm.embedding_model.encode(message))
        similarity = np.dot(queries_embeddings, self.genres_embeddings.T)
        index = np.argsort(-similarity)[0]

        return handles_dict[index]["handler"](message)


