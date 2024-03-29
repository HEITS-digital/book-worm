import re
import json
from typing import Optional

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama, LogitsProcessorList
from lmformatenforcer import CharacterLevelParser, JsonSchemaParser
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor, build_token_enforcer_tokenizer_data

from src.utils.lexrank import degree_centrality_scores
from src.utils.prompts import get_summarization_prompt, get_question_prompt

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma


SENTENCE_TRANSFORMERS_HOME="./models/embeddings"
MODEL_PATH = "models/llama-2-13b-chat.Q2_K.gguf"
EMBEDDING_MODEL_PATH = "all-mpnet-base-v2"


class BookWorm:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        self.embedding_model = SentenceTransformer(
            model_name_or_path=EMBEDDING_MODEL_PATH,
            cache_folder=SENTENCE_TRANSFORMERS_HOME
        )

        self.model = Llama(
            model_path=MODEL_PATH,
            temperature=0.75,
            max_tokens=1000,
            n_ctx=1024,
            top_p=1,
            verbose=True,  # Verbose is required to pass to the callback manager
        )

        self.token_enforcer = build_token_enforcer_tokenizer_data(self.model)


    def get_central_sentences(self, text, k=10):
        # Split the document into sentences
        sentences = [x.text for x in self.nlp(text).sents]

        # Compute the sentence embeddings
        embeddings = self.embedding_model.encode(sentences)

        # Compute the pair-wise cosine similarities
        cos_scores = util.cos_sim(embeddings, embeddings).numpy()

        # Compute the centrality for each sentence
        centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

        # We argsort so that the first element is the sentence with the highest score
        most_central_sentence_indices = np.argsort(-centrality_scores)

        # Get sentence that have the highest central index
        sentences = [sentences[idx].strip() for idx in most_central_sentence_indices[:k]]
        return sentences


    def summarize_sentences(self, sentences):
        text_to_summarize = " ".join(sentences[:4]) # TODO: find a better way to manage max_token limitation
        summarization_prompt = get_summarization_prompt(text_to_summarize)

        output = self._llamacpp_with_character_level_parser(
            summarization_prompt["prompt"],
            JsonSchemaParser(summarization_prompt["output_schema"])
        )

        return json.loads(output)


    def extract_authors(self, text):
        authors = []
        doc = self.nlp(text)
        for chunk in doc.noun_chunks:
            contains_entity = any([x.ent_type_ != "" for x in chunk])
            if contains_entity:
                authors.append(chunk.text)
        return authors


    def extract_book_titles(text):
        titles = []
        text = "Tell me about 'Romeo and Juliet' by Shakespeare"
        regex_str = r"([\"'])(?:(?=(\\?))\2.)*?\1"
        for text_match in re.finditer(regex_str, text):
            titles.append(text_match.group())
        return titles


    def extract_genres(text):
        genres = []
        supported_genres = [x.strip().lower() for x in open("supported_genres.txt", "r").readlines()]
        for genre in supported_genres:
            if genre in text.lower():
                genres.append(genre)
        return genres


    def _llamacpp_with_character_level_parser(self, prompt: str, character_level_parser: Optional[CharacterLevelParser]) -> str:
        logits_processors: Optional[LogitsProcessorList] = None
        if character_level_parser:
            logits_processors = LogitsProcessorList([build_llamacpp_logits_processor(self.token_enforcer, character_level_parser)])

        output = self.model(prompt, temperature=0.0, logits_processor=logits_processors, max_tokens=1024)
        text: str = output['choices'][0]['text']
        return text

    def query_book(self, question):
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            cache_folder=SENTENCE_TRANSFORMERS_HOME
        )
        raw_documents = TextLoader('books/MarcusAurelius.txt').load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        db = Chroma.from_documents(documents, embedding_model)

        docs = db.similarity_search(question)
        page_contents = [doc.page_content for doc in docs]
        prompt = get_question_prompt(question, page_contents[0])
        result = self.model(prompt, temperature=0.0, max_tokens=1024)
        print(result)


if __name__ == "__main__":
    from src.utils.chapterize import parse_document_in_chapters

    book_worm = BookWorm()

    book_worm.query_book("When was Marcus Aurelius born?")
    # data = open("data/book1-txt.txt", "r").read()
    # chapters = parse_document_in_chapters(data)

    # print(f"NB OF CHAPTERS: {len(chapters)}")
    # print("#"*20)
    #
    # chapter_text = " ".join(chapters[20])
    # top_sentences = book_worm.get_central_sentences(chapter_text)
    # print("~~~TOP SENTENCES~~~")
    # print("\n".join(top_sentences))
    # print("#"*20)
    #
    # output = book_worm.summarize_sentences(top_sentences)
    # print("~~~OUTPUT~~~")
    # print(output)
