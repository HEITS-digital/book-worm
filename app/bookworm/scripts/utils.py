import uuid

from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bookworm.scripts.byte_store import PostgresByteStore
from bookworm.scripts.database import CONNECTION_STRING


def get_retriever(collection_name):
    embeddings = OpenAIEmbeddings()
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    store = PostgresByteStore(CONNECTION_STRING, collection_name)
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    return retriever


def update_vectorstore(retriever, book_text, chapter_name):
    id_key = "doc_id"

    documents = split_documents(book_text, 2048 * 2)

    doc_ids = [str(uuid.uuid4()) for _ in documents]
    chapter_names = [chapter_name for _ in documents]

    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    all_sub_docs = []
    for i, doc in enumerate(documents):
        doc_id = doc_ids[i]
        sub_docs = child_text_splitter.split_documents([doc])
        for sub_doc in sub_docs:
            sub_doc.metadata[id_key] = doc_id
        all_sub_docs.extend(sub_docs)

    retriever.vectorstore.add_documents(all_sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, documents, chapter_names)))


def split_documents(book_text, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    if type(book_text) == str:
        documents = text_splitter.create_documents([book_text])
    else:
        documents = text_splitter.split_documents(book_text)

    return documents
