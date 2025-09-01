import os
import gradio as gr
from resume_chatbot import ResumeChatbot

def main():
    bot = ResumeChatbot(
        gcs_bucket=os.getenv('GCS_BUCKET'),
        gcs_resume_path=os.getenv('GCS_RESUME_PATH', 'resume.pdf'),
        gcs_summary_path=os.getenv('GCS_SUMMARY_PATH', 'summary.txt')
    )
    # # 로컬 파일 사용 (기존 방식)
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # bot = ResumeChatbot(
    #     use_gcs=False,
    #     resume_path=os.path.join(BASE_DIR, "me", "resume.pdf"),
    #     summary_path=os.path.join(BASE_DIR, "me", "summary.txt")
    # )
    gr.ChatInterface(bot.chat).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()