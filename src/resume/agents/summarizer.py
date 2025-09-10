from resume.history_store import HistoryStore

# Agent 5 대화 요약기 Agent 
class Summarizer:  
    def __init__(self, client) -> None:
        self.client = client
        self.history_store = HistoryStore()

    async def summarize_history(self, history):
        text = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history])
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"아래 대화를 5문장 이내로 요약해줘:\n{text}"}]
        )
        return response.choices[0].message.content.strip()
