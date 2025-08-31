from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import json, os
from datetime import datetime


def main() -> int:
    load_dotenv(override=True)
    openai = OpenAI()
    reader = PdfReader("src/resume/me/resume.pdf")
    resume = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            resume += text

    with open("src/resume/me/summary.txt", "r", encoding="utf-8") as f:
        summary = f.read()
        
    full_text = summary + "\n\n" + resume
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_text(full_text)

    embeddings = OpenAIEmbeddings()
    persist_dir = "db/chroma"
    if os.path.exists(persist_dir):
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        vectordb = Chroma.from_texts(docs, embeddings, persist_directory=persist_dir)
        vectordb.persist()

    CACHE_FILE = "answer_cache.json"

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            answer_cache = json.load(f)
    else:
        answer_cache = {}

    def save_cache():
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(answer_cache, f, ensure_ascii=False, indent=2)

    def get_cached_answer(question: str):
        """질문이 캐시에 있으면 답변 반환"""
        if question in answer_cache:
            return answer_cache[question]["answer"]
        return None

    def add_to_cache(question: str, answer: str, category: str):
        """캐시에 새 답변 추가"""
        entry = {
            "answer": answer,
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
        answer_cache[question] = entry
        save_cache()

    name = "Yoonha Lee"

    # Agent 1: 질문 분류기
    def classify_question(question: str) -> str:
        """질문을 카테고리로 분류"""
        categories = ["프로젝트 경험", "기술스택", "협업", "자기소개", "학습 경험"]
        prompt = f"""
        분류할 질문: "{question}"
        아래 카테고리 중 가장 알맞은 하나로 분류해줘:
        {categories}
        반드시 위 카테고리 중 하나만 출력해.
        """
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content.strip()


    # Agent 2: 지식 검색기 (RAG Agent)
    def retrieve_context(question: str) -> str:
        """질문과 가장 유사한 Resume/summary 부분을 검색"""
        results = vectordb.similarity_search(question, k=3)  
        return "\n".join([r.page_content for r in results])

    def is_context_valid(question: str, threshold: float = 0.6) -> bool:
        results = vectordb.similarity_search_with_score(question, k=1)
        if not results:
            return False
        _, score = results[0]
        return score >= threshold

    # Agent 3: 답변 생성기 (Persona Agent)
    def persona_answer(question: str, category: str, context: str, history: list) -> str:
        """이력서 주인공(Yoonha Lee)의 톤으로 답변 생성"""
        prompt = f"""
        당신은 {name}으로서 행동하고 있습니다. 
        당신은 {name}의 웹사이트에서 질문에 답변하고 있으며, 
        특히 {name}의 경력, 배경, 기술 및 경험과 관련된 질문에 응답하고 있습니다. 
        당신의 책임은 {name}을 웹사이트 상에서 가능한 한 충실하게 대표하는 것입니다. 
        당신은 질문에 답하기 위해 {name}의 자기소개 요약과 경력기술서를 제공받았습니다. 
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
        - 위와 같은 문맥과 함께, {name}으로서 사용자에게 응답함을 명심한다.
        - 전문 지식을 가진 면접 응시자로 대답한다.
        - 개인 적인 경험과 성과, 배운점을 강조한다. 
        - 인터뷰 응답자 형식의 대화 형식을 유지한다. 
        """
        messages = [{"role": "system", "content": prompt}]

        if len(history) > 3: 
            summary_text = summarize_history(history[:-3]) 
            messages.append({"role": "system", "content": f"이전 대화 요약: {summary_text}"})
            recent_history = history[-3:]
        else:
            recent_history = history

        for turn in recent_history:
            messages.append({"role": "user", "content": turn["q"]})
            messages.append({"role": "assistant", "content": turn["a"]})
        messages.append({"role": "user", "content": question})

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content

    # Agent 4 스타일 보정 Agent
    def refine_answer(answer: str) -> str:
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
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content

    # Agent 5 대화 요약기 Agent 
    def summarize_history(history):
        text = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history])
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"아래 대화를 5문장 이내로 요약해줘:\n{text}"}]
        )
        return response.choices[0].message.content.strip()


    # -----------------------------------------------------
    # 📌 Step 3: Conversational Memory (대화 맥락 유지)
    # -----------------------------------------------------
    conversation_history = []

    def chat(message, history):
        # 캐시에 있다면 답변 
        cached = get_cached_answer(message)
        if cached:
            return cached

        if not is_context_valid(message):
            return "제 이력서나 요약에는 해당 정보가 포함되어 있지 않아서 답변드리기 어려워요."

        # 1) 질문 분류
        category = classify_question(message)

        # 2) 관련 컨텍스트 검색
        context = retrieve_context(message)
        if not context.strip():
            final_answer = "제 이력서나 요약에는 해당 정보가 포함되어 있지 않아서 답변드리기 어려워요."
            conversation_history.append({"q": message, "a": final_answer})
            return final_answer

        # 3) Persona 답변 생성
        draft_answer = persona_answer(message, category, context, conversation_history)
        if "제 이력서에는 해당 정보가 없습니다." in draft_answer:
            return "제 이력서나 요약에는 해당 정보가 포함되어 있지 않아서 답변드리기 어려워요."

        # 4) 스타일 보정
        final_answer = refine_answer(draft_answer)

        # 5) 대화 기록 저장
        conversation_history.append({"q": message, "a": final_answer})
        add_to_cache(message, final_answer, category)

        return final_answer

    gr.ChatInterface(chat).launch()

    return 0



if __name__ == "__main__":
    main()