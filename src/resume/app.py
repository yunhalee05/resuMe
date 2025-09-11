import os
import gradio as gr
import uuid

from resume.resume_chatbot import ResumeChatbot


def init_session():
    return str(uuid.uuid4())

def main():
    # bot = ResumeChatbot(
    #     gcs_bucket=os.getenv('GCS_BUCKET'),
    #     gcs_projects_path=os.getenv('GCS_PROJECTS_PATH', 'projects.json'),
    #     gcs_qna_path=os.getenv('GCS_QNA_PATH', 'qna.json'),
    #     gcs_introduce_path=os.getenv('GCS_INTRODUCE_PATH', 'introduce.txt')
    # )
    # 로컬 파일 사용 (기존 방식)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bot = ResumeChatbot(
        gcs_bucket=os.getenv('GCS_BUCKET'),
        gcs_projects_path=os.path.join(BASE_DIR, "me", "projects.json"),
        gcs_qna_path=os.path.join(BASE_DIR, "me", "qna.json"),
        gcs_introduce_path=os.path.join(BASE_DIR, "me", "introduce.txt"),
        use_gcs=False,
    )

    with gr.Blocks(css="""
        @import url('https://fonts.googleapis.com/css?family=Black+Han+Sans:400')
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background: #ffffff;
            color: #2f3437;
        }
        .notion-header {
            padding: 30px 20px;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 20px;
        }
        .notion-title {
            font-size: 2.2em;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .notion-subtitle {
            font-size: 1.2em;
            color: #6b7280;
            margin-top: 6px;
            font-weight: 600;
            font-family: 'Black Han Sans', sans-serif;  
        }
        .notion-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .notion-button {
            background: #111827 !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
        }
        .notion-textbox input {
            border-radius: 8px !important;
            border: 1px solid #d1d5db !important;
        }
    """) as demo:
        # 상단 헤더
        gr.HTML(
            """
            <div class="notion-header">
                <div class="notion-title">resu<span style="color:#fac53e;">Me</span></div>
                <div class="notion-subtitle">Yoonha Lee 의 커리어 아바타와 대화로 경험해보세요! </div>
            </div>
            """
        )

        # 메인 카드
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    height=500,
                    bubble_full_width=False,
                    show_copy_button=True,
                    label="Chat"
                )
                msg = gr.Textbox(
                    placeholder="Ask something...",
                    label="Message",
                    elem_classes="notion-textbox"
                )
                clear = gr.Button("Clear Chat", elem_classes="notion-button")

        session_id = gr.State(init_session)
        # 응답 함수
        async def respond(message, history, session_id):
            response = await bot.chat(message, history, session_id)   
            history.append((message, response))
            return "", history, session_id

        msg.submit(respond, [msg, chatbot, session_id], [msg, chatbot, session_id])
        clear.click(lambda: None, None, chatbot, queue=False)


    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)



if __name__ == "__main__":
    main()