import json
import requests
from .gutenberg_pool import GutenbergPool


gutenberg_pool = GutenbergPool(max_size=10)


def gutenberg_connection(func):
    """Decorator that reports the execution time."""

    def wrap(*args, **kwargs):
        gutten = gutenberg_pool.acquire()
        result = func(gutten, *args, **kwargs)
        gutenberg_pool.release(gutten)
        return result

    return wrap


@gutenberg_connection
def get_book_meta_by_title_author(guten, title, author, top_k=1):
    result = guten.search(f"language:en AND title:{title} AND author: {author}")
    return list(result)[:top_k]


@gutenberg_connection
def get_book_contents_by_id(gutten, bookworm_key):
    gutten.download(bookworm_key)
    book_text = next(gutten.text(bookworm_key))
    return book_text


@gutenberg_connection
def search_author(guten, message, top_k=3):
    relevant_books = {"on_bookworm": [], "not_on_bookworm": []}

    search_result = guten.search(f"language:en AND author: {message}")

    found_books = 0
    for book_meta in search_result:
        if "Index" in book_meta["title"][0]:
            continue

        # search for books (need descriptions) using google API
        data = query_google_books(book_meta["title"])

        description = ""
        if len(data) > 0:
            description = data[0].get("description")
        bookworm_key = book_meta.get("key", None)
        online_book_url = get_gutenberg_url(bookworm_key) if bookworm_key else ""

        # this info will be used to answer questions about the book
        book_info = {
            "is_available": True,
            "title": book_meta["title"],
            "authors": message,
            "description": description,
            "genre": book_meta.get("subject", "UNKNOWN"),
            "bookworm_key": bookworm_key,
            "online_book_url": online_book_url,
        }

        relevant_books["on_bookworm"].append(book_info)
        found_books += 1

        if found_books == top_k:
            break

    return relevant_books


def search_book(query):
    relevant_books = {"on_bookworm": [], "not_on_bookworm": []}

    # search for books (need descriptions) using google API
    data = query_google_books(query)

    for item in data:
        title = item["title"]
        main_author = item.get("authors", [""])[0]

        # check if we have that book in our database
        book_meta = get_book_meta_by_title_author(title=title, author=main_author)

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
        online_book_url = get_gutenberg_url(bookworm_key) if bookworm_key else ""

        # this info will be used to answer questions about the book
        book_info = {
            "title": title,
            "authors": main_author,
            "description": item.get("description", ""),
            "genre": genre,
            "bookworm_key": bookworm_key,
            "online_book_url": online_book_url,
        }

        if is_in_library:
            relevant_books["on_bookworm"].append(book_info)
        else:
            relevant_books["not_on_bookworm"].append(book_info)

    return relevant_books


def get_book_id_by_title(title, last_user_response=dict()):
    for book in last_user_response.get("on_bookworm", []):
        if book["title"] == title:
            return {"id": book["bookworm_key"]}

    response = search_book(title)

    for book in response.get("on_bookworm", []):
        if book.get("title", "") == title:
            return {"id": book["bookworm_key"]}

    return {"id": None}


def get_gutenberg_url(bookworm_key):
    return f"https://www.gutenberg.org/cache/epub/{bookworm_key}/pg{bookworm_key}-images.html"


def filter_google_books(data):
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


def query_google_books(message):
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": message}
    resp = requests.get(url, params=params)
    raw_json = json.loads(resp.content)
    return filter_google_books(raw_json)
