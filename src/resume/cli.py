import argparse
import sys
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="resume", description="Resume CLI")
    subparsers = parser.add_subparsers(dest="command")

    greet = subparsers.add_parser("greet", help="Print a greeting")
    greet.add_argument("name", nargs="?", default="World", help="Name to greet")

    parser.add_argument("-v", "--version", action="version", version="0.1.0")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv(override=True)
    openai = OpenAI()
    reader = PdfReader("me/resume.pdf")
    resume = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            resume += text
    print(text)

    with open("me/summary.txt", "r", encoding="utf-8") as f:
        summary = f.read()

    name = "Yoonha Lee"

    system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
    particularly questions related to {name}'s career, background, skills and experience. \
    Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
    You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. \
    Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
    If you don't know the answer, say so."

    system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
    system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

    system_prompt


    def chat(message, history):
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
        response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content

    gr.ChatInterface(chat, type="messages").launch()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


