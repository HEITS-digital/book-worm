from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

SENTENCE_TRANSFORMERS_HOME="./models/embeddings"
EMBEDDING_MODEL_PATH = "all-mpnet-base-v2"
embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            cache_folder=SENTENCE_TRANSFORMERS_HOME
        )

raw_documents = TextLoader('books/MarcusAurelius.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = Chroma.from_documents(documents, embedding_model)


query = "When was Marcus Aurelius born?"
docs = db.similarity_search(query)
print(docs[0].page_content)
