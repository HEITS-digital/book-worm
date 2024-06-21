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

subjects = {k: v for k, v in sorted(subjects.items(), key=lambda item: item[1], reverse=True)}

top_100_subjects = []
for key in subjects:
    if key != key.upper() and len(key.split(" ")) == 1:
        top_100_subjects.append(key)
    if len(top_100_subjects) == 100:
        break

open("supported_genres.txt", 'w').write("\n".join(top_100_subjects))
