import json

from langchain_community.document_loaders.text import TextLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from lmformatenforcer import JsonSchemaParser
from transformers import pipeline, AutoTokenizer

from src.utils.prompts import get_question_prompt, SummaryAnswerFormat
from src.utils.utils import llamacpp_with_character_level_parser
from optimum.onnxruntime import ORTModelForQuestionAnswering

SENTENCE_TRANSFORMERS_HOME = "./models/embeddings"
EMBEDDING_MODEL_PATH = "all-mpnet-base-v2"


class Query:
    def __init__(self, model, token_enforcer):
        self.model = model
        self.token_enforcer = token_enforcer
        self.first_split_contents = []
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            cache_folder=SENTENCE_TRANSFORMERS_HOME
        )

    def query_book(self, text_location, question):
        raw_documents = TextLoader(text_location).load()

        if not self.first_split_contents:
            self.first_split_contents = self.get_page_contents(question, 1000, raw_documents)

        parsed_page_contents = self.get_page_contents(question, 256)

        merged_text = ' '.join(parsed_page_contents)
        prompt = get_question_prompt(question, merged_text.replace('\n', '').replace('  ', ''))

        model = ORTModelForQuestionAnswering.from_pretrained(model_id, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        optimum_qa = pipeline("question-answering", model=self.model, tokenizer=self.token_enforcer, handle_impossible_answer=False)
        prediction = optimum_qa(question=question, context=merged_text.replace('\n', '').replace('  ', ''))
        print(prediction)

        result = llamacpp_with_character_level_parser(prompt, self.model, self.token_enforcer,
                                                      JsonSchemaParser(SummaryAnswerFormat.schema()))
        return json.loads(result)

    def get_page_contents(self, question, chunk_size, raw_documents=None):
        text_splitter = CharacterTextSplitter(separator=".", chunk_size=chunk_size, chunk_overlap=0)
        if raw_documents:
            documents = text_splitter.split_documents(raw_documents)
        else:
            documents = text_splitter.create_documents(self.first_split_contents)
        db = Chroma.from_documents(documents, self.embedding_model)
        docs = db.similarity_search(question)
        return [doc.page_content for doc in docs]
