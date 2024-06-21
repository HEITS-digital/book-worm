# %%
import json
import requests

from scripts.gutenberg import Gutenberg


DB_PATH = "~/.gutenberg"

# Default number of worker processes for parallel downloads.
DOWNLOAD_POOL_SIZE = 4

# Where to find the Gutenberg catalog. Must be the address of the bz2 file, not
# the zip file.
CATALOG_URL = "http://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2"

# Where to find the list of Gutenberg mirrors.
MIRRORS_URL = "https://www.gutenberg.org/MIRRORS.ALL"

guten = Gutenberg()
# %%

query = "Do you know any books written by Aurelius"

url = "https://www.googleapis.com/books/v1/volumes"
params = {'q': query}

resp = requests.get(url, params=params)

data = json.loads(resp.content)
# %%
google_books = {}
guten_books = {}

for item in data['items']:
    if item['volumeInfo']['language'] != 'en':
        continue

    authors = item['volumeInfo'].get('authors', None)
    if authors is None:
        continue

    title = item['volumeInfo']['title']
    author = authors[0]

    google_books[title] = item['volumeInfo']

    book_meta = list(guten.search(f"language:en AND title:{title} AND author: {author}"))
    
    if len(book_meta) == 0:
        continue

    book_meta = book_meta[0]

    if "Index" in book_meta['title'][0]:
        continue

    guten_books[book_meta['title'][0]] = book_meta

guten_books
# %%



book_meta
# %%
