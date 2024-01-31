# %%
import nltk
import numpy as np
from sentence_transformers import util
from langchain_community.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from book_worm_chains import fetch_summarizing_chain
from utils import degree_centrality_scores

MODEL_PATH = "models/firefly-llama2-7b-chat.Q5_K_M.gguf"
EMBEDDING_MODEL_PATH = "all-mpnet-base-v2"
SENTENCE_TRANSFORMERS_HOME="/Users/gianistatie/Documents/ecree/models/embeddings"

# %%
nltk.download('punkt')

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_PATH, 
    cache_folder=SENTENCE_TRANSFORMERS_HOME
)

model = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.75,
    max_tokens=1000,
    n_ctx=1024,
    top_p=1,
    verbose=True,  # Verbose is required to pass to the callback manager
)

# %%
# data = open("books/MarcusAurelius.txt", "r").read()
data = open("books/Frankenstein-Mary Wollstonecraft.txt", "r").read()

data = [line.replace("\n", " ") for line in data.split("\n\n")]
data = [line for line in data if line != line.upper()]
data = "\n".join(data)

# %%
k = 5

# Split the document into sentences
sentences = nltk.sent_tokenize(data)

# Compute the sentence embeddings
embeddings = embedding_model.embed_documents(sentences)

# Compute the pair-wise cosine similarities
cos_scores = util.cos_sim(embeddings, embeddings).numpy()

# Compute the centrality for each sentence
centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

#%%

# We argsort so that the first element is the sentence with the highest score
most_central_sentence_indices = np.argsort(-centrality_scores)

central_central_sentence_indices = most_central_sentence_indices[len(sentences)//2:]

# Get sentence that have the highest central index
top_sentences = [sentences[idx].strip() for idx in most_central_sentence_indices[:k]] + \
    [sentences[idx].strip() for idx in central_central_sentence_indices[:k]]

# %%
documents = [
    Document(page_content=text, metadata={"source": "local"})
    for text in top_sentences
]

summarizing_chain = fetch_summarizing_chain(model)
output = summarizing_chain.run(documents)
# %%
