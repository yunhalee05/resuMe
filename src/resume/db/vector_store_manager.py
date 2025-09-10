from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import json, os
from datetime import datetime
from google.cloud import storage
import tempfile
from dateutil import parser

class VectorStoreManager:
    def __init__(self, gcs_bucket: str, gcs_projects_path: str, gcs_qna_path: str, gcs_introduce_path: str, use_gcs = True, cache_file: str = "answer_cache.json", name: str = "Yoonha Lee"):
        load_dotenv(override=True)
        self.name = name 
        self.gcs_bucket = gcs_bucket

        if use_gcs:
            try:
                self.storage_client = storage.Client()
                print("GCS 클라이언트 초기화 성공")
            except Exception as e:
                print(f"GCS 클라이언트 초기화 실패: {e}")

        self.docs = []
        self.meta = []
        
        reader_func = self._read_from_gcs if use_gcs else self._read_from_local
        projects = reader_func(gcs_projects_path, is_json=True)
        qna = reader_func(gcs_qna_path, is_json=True)
        summary = reader_func(gcs_introduce_path, is_json=False)
        
        self._project_json_to_docs(projects)
        self._qna_json_to_docs(qna)
        self._text_to_docs(summary)

        embeddings = OpenAIEmbeddings()
        persist_dir = "db/chroma"
        if os.path.exists(persist_dir):
            self.vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        else:
            self.vectordb = Chroma.from_texts(self.docs, embeddings, metadatas=self.meta, persist_directory=persist_dir)
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
        
        
    def _read_from_gcs(self, file_path: str, is_pdf: bool = False, is_json: bool = False) -> str:
        """GCS에서 파일을 읽어오는 메서드"""
        if not hasattr(self, 'storage_client') or not self.storage_client:
            print("GCS 클라이언트가 초기화되지 않았습니다.")
            return ""

        try:
            bucket = self.storage_client.bucket(self.gcs_bucket)
            blob = bucket.blob(file_path)
            
            if is_pdf:
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    reader = PdfReader(temp_file.name)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                    os.unlink(temp_file.name) 
                    return text
            elif is_json:
                text = blob.download_as_text(encoding='utf-8')
                return json.loads(text)
            else:
                return blob.download_as_text(encoding='utf-8')
                
        except Exception as e:
            print(f"GCS에서 파일 읽기 실패 ({file_path}): {e}")
            return ""

    def _read_from_local(self, file_path: str, is_pdf: bool = False, is_json: bool = False) -> str:
        """로컬 파일에서 읽어오는 메서드"""
        try:
            if is_pdf:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text
            elif is_json:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            print(f"로컬 파일 읽기 실패 ({file_path}): {e}")
            return ""

    def _project_json_to_docs(self, data: dict): 
        doc_type = "projects"
        if "projects" in data:
            for p in data["projects"]:
                content = json.dumps(p, ensure_ascii=False, indent=2)
                period_from = self._parse_date(p.get("period", {}).get("from"))
                period_to = self._parse_date(p.get("period", {}).get("to"))

                tmp = {
                    "doc_type": doc_type,
                    "company": p.get("company"),
                    "role": ", ".join(p.get("role", [])) if isinstance(p.get("role"), list) else p.get("role"),
                    "period": f"{p['period'].get('from', '')}~{p['period'].get('to', '')}" if isinstance(p.get("period"), dict) else p.get("period"),
                    "period_from": period_from.isoformat() if period_from else None,
                    "period_to": period_to.isoformat() if period_to else None,
                    "tech_stack": ", ".join([t["name"] for t in p.get("tech_stack", [])])
                }
                self.docs.append(content)
                self.meta.append(tmp)

    def _parse_date(self, val):
        if not val:
            return None
        if val.upper() in ["ING", "CURRENT", "PRESENT"]:
            return datetime.now()
        try:
            return datetime.strptime(val, "%Y.%m")
        except ValueError:
            try:
                return parser.parse(val)
            except Exception:
                return None

    def _qna_json_to_docs(self, data: dict): 
        doc_type = "qna"
        for q in data:
            content = json.dumps(q, ensure_ascii=False, indent=2)
            tmp = {
                "doc_type": doc_type, 
                "topic_tags": ", ".join(q.get("topic_tags", [])) if isinstance(q.get("topic_tags"), list) else q.get("topic_tags"),
            }
            self.docs.append(content)
            self.meta.append(tmp)

    def _text_to_docs(self, data: str):
        doc_type = "summary"
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(data) 
        for c in chunks:
            self.docs.append(c)
            self.meta.append({
                "doc_type": doc_type
            })  
