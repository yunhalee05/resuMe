# Agent 5 대화 요약기 Agent 
from openai import AsyncOpenAI


class Summarizer:  
    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client

    async def summarize_history(self, history: list[dict[str, str]]) -> str:
        text = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history])
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"아래 대화를 5문장 이내로 요약해줘:\n{text}"}]
        )
        return response.choices[0].message.content.strip()
