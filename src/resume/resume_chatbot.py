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


class ResumeChatbot:
    def __init__(self, gcs_bucket: str, gcs_resume_path: str, gcs_summary_path: str, cache_file: str = "answer_cache.json"):
        load_dotenv(override=True)
        self.client = AsyncOpenAI()
        self.name = "Yoonha Lee"
        self.gcs_bucket = gcs_bucket

        try:
            self.storage_client = storage.Client()
            print("GCS 클라이언트 초기화 성공")
        except Exception as e:
            print(f"GCS 클라이언트 초기화 실패: {e}")
        
        resume = self._read_from_gcs(gcs_resume_path, is_pdf=True)
        summary = self._read_from_gcs(gcs_summary_path, is_pdf=False)
    
        full_text = summary + "\n\n" + resume
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_text(full_text)

        embeddings = OpenAIEmbeddings()
        persist_dir = "db/chroma"
        if os.path.exists(persist_dir):
            self.vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        else:
            self.vectordb = Chroma.from_texts(docs, embeddings, persist_directory=persist_dir)
            self.vectordb.persist()

        self.cache_file = cache_file
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                self.answer_cache = json.load(f)
        else:
            self.answer_cache = {}
        
        self.conversation_history = []
        
    def _read_from_gcs(self, file_path: str, is_pdf: bool = False) -> str:
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
            else:
                # 텍스트 파일인 경우 직접 읽기
                return blob.download_as_text(encoding='utf-8')
                
        except Exception as e:
            print(f"GCS에서 파일 읽기 실패 ({file_path}): {e}")
            return ""




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
        """질문을 카테고리로 분류"""
        categories = ["프로젝트 경험", "기술스택", "협업", "자기소개", "학습 경험"]
        prompt = f"""
        분류할 질문: "{question}"
        아래 카테고리 중 가장 알맞은 하나로 분류해줘:
        {categories}
        반드시 위 카테고리 중 하나만 출력해.
        """
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

   
    # Agent 2: 지식 검색기 (RAG Agent)
    async def retrieve_context(self, question: str) -> str:
        """질문과 가장 유사한 Resume/summary 부분을 검색"""
        results = self.vectordb.similarity_search(question, k=3)
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
            summary_text = await self.summarize_history(self.conversation_history[:-3]) 
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



    async def chat(self, message: str, history: list):
        # 캐시에 있다면 답변 
        cached = self.get_cached_answer(message)
        if cached:
            return cached

        if not await self.is_context_valid(message):
            return "제 이력서나 요약에는 해당 정보가 포함되어 있지 않아서 답변드리기 어려워요."

        # 1) 질문 분류
        category = await self.classify_question(message)

        # 2) 관련 컨텍스트 검색
        context = await self.retrieve_context(message)
        if not context.strip():
            final_answer = "제 이력서나 요약에는 해당 정보가 포함되어 있지 않아서 답변드리기 어려워요."
            return final_answer

        # 3) Persona 답변 생성
        draft_answer = await self.persona_answer(message, category, context)
        if "제 이력서에는 해당 정보가 없습니다." in draft_answer:
            return "제 이력서나 요약에는 해당 정보가 포함되어 있지 않아서 답변드리기 어려워요."

        # 4) 스타일 보정
        final_answer = await self.refine_answer(draft_answer)

        # 5) 대화 기록 저장
        self.conversation_history.append({"q": message, "a": final_answer})
        self.add_to_cache(message, final_answer, category)

        return final_answer


