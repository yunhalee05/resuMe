from typing import (
    Dict,
    Optional,
)
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

from resume.db.resume_reader import ResumeReader

class VectorStore:
    def __init__(self, gcs_bucket: str, gcs_projects_path: str, gcs_qna_path: str, gcs_introduce_path: str, use_gcs = True, cache_file: str = "answer_cache.json", name: str = "Yoonha Lee"):
        resume_reader = ResumeReader(gcs_bucket, gcs_projects_path, gcs_qna_path, gcs_introduce_path, use_gcs, cache_file, name)
        embeddings = OpenAIEmbeddings()
        persist_dir = "db/chroma"
        if os.path.exists(persist_dir):
            self.vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        else:
            self.vectordb = Chroma.from_texts(resume_reader.docs, embeddings, metadatas=resume_reader.meta, persist_directory=persist_dir)
            self.vectordb.persist()

        # all_data = self.vectordb.get()
        # for i, (doc, meta) in enumerate(zip(all_data["documents"], all_data["metadatas"])):
        #     print(f"[{i}] {doc}")
        #     print(f"META: {meta}")

        # self.cache_file = cache_file
        # if os.path.exists(cache_file):
        #     with open(cache_file, "r", encoding="utf-8") as f:
        #         self.answer_cache = json.load(f)
        # else:
        #     self.answer_cache = {}
    
    def get_similar_data(self, question: str, k: int, filters: Optional[Dict[str, str]] = None):
        results = self.vectordb.similarity_search(
                question,
                k=k,
                filter=filters if filters else None
            )
        
