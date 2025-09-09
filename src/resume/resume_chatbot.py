from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import json, os
from datetime import datetime
from openai import AsyncOpenAI
from google.cloud import storage
import tempfile
from dateutil import parser
from history_store import HistoryStore
from resume import history_store



class ResumeChatbot:
    def __init__(self, gcs_bucket: str, gcs_projects_path: str, gcs_qna_path: str, gcs_introduce_path: str, use_gcs = True, cache_file: str = "answer_cache.json"):
        load_dotenv(override=True)
        self.client = AsyncOpenAI()
        self.name = "Yoonha Lee"
        self.gcs_bucket = gcs_bucket
        self.history_store = HistoryStore()

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

        self.cache_file = cache_file
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                self.answer_cache = json.load(f)
        else:
            self.answer_cache = {}
        
        self.conversation_history = []
        
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
                period_from = self.parse_date(p.get("period", {}).get("from"))
                period_to = self.parse_date(p.get("period", {}).get("to"))

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

    def parse_date(self, val):
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

    def save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.answer_cache, f, ensure_ascii=False, indent=2)

    def get_cached_answer(self, question: str):
        """질문이 캐시에 있으면 답변 반환"""
        if question in self.answer_cache:
            return self.answer_cache[question]["answer"]
        return None

    def add_to_cache(self, question: str, answer: str, category: str):
        """캐시에 새 답변 추가"""
        entry = {
            "answer": answer,
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
        self.answer_cache[question] = entry
        self.save_cache()


    # Agent 1: 질문 분류기
    async def classify_question(self, question: str) -> str:
        """질문을 카테고리 + 검색 메타데이터 필드로 분류"""
        categories = ["프로젝트 경험", "기술스택", "협업", "자기소개", "학습 경험"]
        prompt = f"""
        분류할 질문: "{question}"
        이력서 데이터 메타데이터 필드:
        - company
        - role
        - period, period_from, period_to
        - tech_stack
        - topic_tags (고정값: {{"문제 해결", "프로젝트 경험", "학습 경험", "기술 스택", "지원 동기", "협업"}})
        
        카테고리 → 메타데이터 매핑:
        - 프로젝트 경험 → [company, role, period, period_from, period_to, tech_stack]
        - 기술 스택 → [tech_stack]
        - 학습 경험 → [topic_tags]
        - 협업 → [topic_tags]
        - 지원 동기 → [topic_tags]
        - 자기소개 → [summary]

        카테고리 → doc_type 매핑:
        - 프로젝트 경험, 기술 스택 → "projects"
        - 학습 경험, 협업, 지원 동기 → "qna"
        - 자기소개 → "summary"

        추가 규칙:
        - 항상 filter에 doc_type 포함 (예: {{"doc_type": "projects"}})
        - 질문에 특정 값이 있으면 해당 메타데이터 필드에 regex 조건 추가 가능
        - 프로젝트 경험 질문에 "최근/마지막" → time_condition="recent"
        - 프로젝트 경험 질문에 "처음/첫번째" → time_condition="first"
        - 그 외 → time_condition="none"
        - summary 카테고리의 경우 filter에는 doc_type만 포함
        
        출력 형식(JSON):
        {{
            "category": "<카테고리>",
            "time_condition": "<recent|first|none>",
            "filters": {{
                "doc_type": "<projects|qna|summary>",
                "<필드명>": {{"$regex": ".*<값>.*"}}  
            }}
        }}
        """

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"} 
        )

        return json.loads(response.choices[0].message.content.strip())
       

   
    # Agent 2: 지식 검색기 (RAG Agent)
    async def retrieve_context(self, question: str, category_info: dict) -> str:
        """질문과 가장 유사한 카테고리 정보에 맞는 메타데이터 기반 Resume/summary 부분을 검색"""
        time_condition = category_info.get("time_condition", "none")
        filters = category_info.get("filters", {})  
        category = category_info.get("category")
        k = 5

        try:
            if(time_condition != "none" and category == "프로젝트 경험"):
                results = self.vectordb.similarity_search(
                    question,
                    k=20,
                    filter=filters if filters else None
                )
                def parse_date_safe(val):
                    try:
                        return datetime.fromisoformat(val)
                    except Exception:
                        return datetime.min
                if time_condition == "recent":
                    results = sorted(results, key=lambda r: parse_date_safe(r.metadata.get("period_from")), reverse=True)[:k]
                elif time_condition == "first":
                    results = sorted(results, key=lambda r: parse_date_safe(r.metadata.get("period_from")), reverse=False)[:k]
            else :
                results = self.vectordb.similarity_search(
                    question,
                    k=k,
                    filter=filters if filters else None
                )
        except Exception:
            results = self.vectordb.similarity_search(question, k=3)

        # for r in results:
        #     print(r.page_content)
        #     print("META:", r.metadata)
        #     print("-" * 50)

        if not results:
            return ""

        return "\n".join([r.page_content for r in results])

    async def is_context_valid(self, question: str, threshold: float = 0.2) -> bool:
        results = self.vectordb.similarity_search_with_score(question, k=1)
        if not results:
            return False
        _, score = results[0]
        return score >= threshold

    # Agent 3: 답변 생성기 (Persona Agent)
    async def persona_answer(self, question: str, category: str, context: str) -> str:
        """이력서 주인공(Yoonha Lee)의 톤으로 답변 생성"""
        prompt = f"""
        당신은 {self.name}으로서 행동하고 있습니다. 
        당신은 {self.name}의 웹사이트에서 질문에 답변하고 있으며, 
        특히 {self.name}의 경력, 배경, 기술 및 경험과 관련된 질문에 응답하고 있습니다. 
        당신의 책임은 {self.name}을 웹사이트 상에서 가능한 한 충실하게 대표하는 것입니다. 
        당신은 질문에 답하기 위해 {self.name}의 자기소개 요약과 경력기술서를 제공받았습니다. 
        잠재적인 고객이나 미래의 고용주가 웹사이트에 방문했을 때 대화하는 것처럼, 
        전문적이고 매력적인 태도로 답변해야 합니다. 
        프로젝트 관련 질문을 한다면 STAR 구조(Situation, Task, Action, Result)로 답변을 정리하고,
        각 요소는 1문장씩, 총 4문장 이내로 구성합니다. 
        resume/summary에 없는 질문에는 절대 새로운 사실을 만들어내지 않습니다.
        만약 resume/summary에 해당 정보가 전혀 없다면, 
        "제 이력서에는 해당 정보가 없습니다."라고만 대답하세요.
        추측하거나 새로운 사실을 만들어내지 마세요.

        질문: {question}
        분류: {category}
        이력서 및 요약에서 가져온 컨텍스트: {context}

        답변 지침:
        - 반드시 한국어로 대답한다.
        - 1인칭 시점("저는 ...")으로 말한다.
        - 답변은 실제 면접 대화처럼 자연스럽게, 문장 끝을 다양하게 사용한다. (예: ~했습니다 / ~한 경험이 있습니다 / ~한 것이 기억에 남습니다)
        - '감사합니다' 같은 형식적인 마무리 문구는 사용하지 않는다.
        - 불필요하게 장황하지 않고, 핵심만 담아 3~5문장 정도로 답한다.
        - 글을 읽는 듯한 딱딱한 어투가 아니라, 편안하지만 전문적인 면접 톤으로 한다.
        - 위와 같은 문맥과 함께, {self.name}으로서 사용자에게 응답함을 명심한다.
        - 전문 지식을 가진 면접 응시자로 대답한다.
        - 개인 적인 경험과 성과, 배운점을 강조한다. 
        - 인터뷰 응답자 형식의 대화 형식을 유지한다. 
        """
        messages = [{"role": "system", "content": prompt}]

        if len(self.conversation_history) > 3: 
            summary_text = await self.summarize_history(self.conversation_history[-3:]) 
            messages.append({"role": "system", "content": f"이전 대화 요약: {summary_text}"})
            recent_history = self.conversation_history[-3:]
        else:
            recent_history = self.conversation_history

        for turn in recent_history:
            messages.append({"role": "user", "content": turn["q"]})
            messages.append({"role": "assistant", "content": turn["a"]})
        messages.append({"role": "user", "content": question})

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        return response.choices[0].message.content

    # Agent 4 스타일 보정 Agent
    async def refine_answer(self, answer: str) -> str:
        """답변을 면접 톤으로 최종 다듬기 (길면 줄이고, 핵심 강조)"""
        prompt = f"""
        아래는 면접 답변 초안입니다:
        {answer}

        이 답변을 다음 기준으로 다듬어주세요:
        - 반드시 한국어로 대답한다.
        - 실제 면접 대화처럼 자연스럽고 자신감 있는 어투로 바꾼다.
        - 답변이 너무 길면 핵심만 담아 5문장 이내로 줄인다.
        - 성과와 핵심 경험을 명확히 강조한다.
        - 글을 읽는 듯한 어투 대신, 구어체 면접 답변처럼 자연스럽게 표현한다.
        """
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content

    # Agent 5 대화 요약기 Agent 
    async def summarize_history(self, history):
        text = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history])
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"아래 대화를 5문장 이내로 요약해줘:\n{text}"}]
        )
        return response.choices[0].message.content.strip()



    async def chat(self, message: str, history: list, session_id: str):
        print(session_id)
        print(history)

        # 캐시에 있다면 답변 
        cached = self.get_cached_answer(message)
        if cached:
            return cached
        
        if not await self.is_context_valid(message):
            return "제 이력서나 요약에는 해당 정보가 포함되어 있지 않아서 답변드리기 어려워요."

        # 1) 질문 분류
        category = await self.classify_question(message)

        # 2) 관련 컨텍스트 검색
        context = await self.retrieve_context(message, category)
        if not context.strip():
            final_answer = "제 이력서나 요약에는 해당 정보가 포함되어 있지 않아서 답변드리기 어려워요."
            return final_answer

        # 3) Persona 답변 생성
        # recent_history = self.history_store.get_summary(session_id, self.summarize_history)
        draft_answer = await self.persona_answer(message, category, context)
        if "제 이력서에는 해당 정보가 없습니다." in draft_answer:
            return "제 이력서나 요약에는 해당 정보가 포함되어 있지 않아서 답변드리기 어려워요."

        # 4) 스타일 보정
        final_answer = await self.refine_answer(draft_answer)

        # 5) 대화 기록 저장
        self.conversation_history.append({"q": message, "a": final_answer})
        self.add_to_cache(message, final_answer, category)

        print(self.conversation_history)

        return final_answer


