# %%
from book_worm import BookWorm

bookworm = BookWorm()
# %%
user_queries = [
    "I want to read a book about Nostradamus and how he invented the light",
    "Do you know the author Elena Ferrante?",
    "Tell me about 'Romeo and Juliet' by Shakespeare",
    "Do you know any books written by Joules Verne",
    "I want to read about a thriller",
    "The Catcher in the Rye was a wonderful book. Could you suggest a similar one?",
    "Can you go into more details?",
    "What genres do you have?",
]
handles = [
    "search for a book, play or poem",
    "search for an author, writer or person",
    "search for a genre",
    "suggest genre",
    "more information",
]


# %%
import numpy as np

queries_embeddings = np.array(bookworm.embedding_model.embed_documents(user_queries))
genres_embeddings = np.array(bookworm.embedding_model.embed_documents(handles))
# %%


similarity = np.dot(queries_embeddings, genres_embeddings.T)
print(similarity)
# %%
from scripts.gutenberg import Gutenberg

# %%
# Default database path.
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
subjects = {}
# "author", "title", "language"
for rez in guten.search("language:en"):
    for subj in rez["subject"]:
        if subj not in subjects:
            subjects[subj] = 0
        subjects[subj] += 1

subjects = {
    k: v for k, v in sorted(subjects.items(), key=lambda item: item[1], reverse=True)
}

top_100_subjects = []
for key in subjects:
    if key != key.upper():
        top_100_subjects.append(key)
    if len(top_100_subjects) == 100:
        break

open("supported_genres.txt", "w").write("\n".join(top_100_subjects))
# %%
