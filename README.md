<div align="center">
    <br/>
    <h1><strong><em>💼 resuMe</em></strong></h1>
    <img width="800" alt="resuMe Demo Screenshot" src="docs/demo.png">
    <br/>
    <br/>
    <p>
      <strong><em>resuMe</em></strong>는 <strong>AI 기반 커리어 아바타</strong> 프로젝트입니다.<br/> 
      사용자의 이력서와 요약 텍스트를 기반으로 RAG(검색 증강 생성) 챗봇을 구축하여,<br/>
      방문자가 마치 본인과 대화하듯 경력과 경험을 탐색할 수 있습니다.<br/>
      OpenAI API, LangChain, Gradio, Google Cloud Storage를 활용하여 인터랙티브한 포트폴리오 경험을 제공합니다.<br/>
    </p>
</div>
    <br/>
    <br/>

<div>
    <h2>🚀 Shortcut</h2>
<div> 

- [__AI Agent Architecture__](#agent-architecture)
- [__Workflow__](#workflow)
- [__Implementation__](#implementation)
- [__Infrastructure__](#infrastructure)
- [__Tech Stack Used__](#tech)
- [__Features__](#feature)

</div>
    <br/>
    <br/>
<div>
    <h2 id="agent-architecture">🧩 AI Agent Architecture</h2>
    <p>
    resuMe는 <strong>Multi-Agent System</strong>으로 Prompt Chaining 워크플로우 방식으로  설계되었습니다.  
    각 에이전트는 독립적인 역할을 수행하며, 파이프라인 방식으로 연결되어 질문에 대해 최적화된 답변을 생성합니다.
    </p>
    <ul>
        <li>
            <h3> 분류기(Question Classifier Agent)</h3> 
            <p>
                질문을 프로젝트 경험, 기술 스택, 협업 등 카테고리로 분류
            </p>
        </li>
        <li>
            <h3> 검색기(RAG Retriever Agent)</h3> 
            <p>
                Chroma 벡터DB에서 이력서/요약 기반 문맥 검색
            </p>
        </li>
        <li>
            <h3> Persona Agent</h3> 
            <p>
                1인칭 면접 톤으로 답변 생성, STAR 구조 응답 (Situation, Task, Action, Result)
            </p>
        </li>
        <li>
            <h3> 스타일 보정 Agent</h3> 
            <p>
                응답을 간결하고 자신감 있는 면접 어투로 최종 다듬기
            </p>
        </li>
        <li>
            <h3> 대화 요약기 Agent</h3> 
            <p>긴 대화를 5문장 이내로 요약, 프롬프트 최적화
            </p>
        </li>
    </ul>
</div>

<div>
    <h2 id="workflow">⚡️ Workflow</h2>

1. **Input Handling**
   - Gradio UI에서 메시지를 입력받고, 비동기 Task로 전달
2. **Classification**
   - 분류기 Agent가 질문의 카테고리를 결정
3. **Context Retrieval**
   - RAG 검색기로 유사도가 가장 높은 resume/summary 부분 검색
4. **Persona Answering**
   - OpenAI API를 통해 이력서 주인공(본인) 톤으로 답변 생성
5. **Answer Refinement**
   - 스타일 보정기로 답변을 5문장 이내 면접 톤으로 최적화
6. **History Management**
   - 최근 3턴 유지 + 이전 대화 요약 저장
7. **Caching**
   - 동일 질문 재사용 시 JSON 캐시에서 즉시 응답
</div>

<div>
    <h2 id="implementation">🍑 Implementation</h2>
    <ul>
      <li>
        <h3>✔️ Multi-Agent Pipeline</h3> 
          <p>단일 LLM 호출 대신, 에이전트를 분리하여 유지보수성과 확장성을 확보했습니다.
            질문 분류기 → 지식 검색기 → Persona 답변 생성기 → 스타일 보정기 → 대화 요약기의 파이프라인 구조를 설계했습니다.
          </p>
      </li>
        <li>
        <h3>✔️ RAG (검색 증강 생성)</h3> 
        <p>
          이력서 및 요약을 Chroma 벡터DB에 저장하고, 의미 기반 검색을 통해 답변 컨텍스트를 제공합니다.
        </p>
        </li>
        <li>
        <h3>✔️ Context Validation</h3> 
          <p>유사도 임계값 검증을 통해, 이력서와 무관한 질문에는 "답변 불가"를 반환하도록 가이드라인을 통해 안정성을 강화했습니다.</p>
      </li>
      <li>
        <h3>✔️ GCS 연동</h3> 
          <p>
            Google Cloud Storage(GCS)에서 <code>resume.pdf</code>와 <code>summary.txt</code>를 불러와 초기 이력서를 반영할 수 있도록 구성하였습니다.
          </p>
      </li>
      <li>
        <h3>✔️ Docker & Github workflow를 통한 CI/CD 자동화</h3> 
          <p>
            Github workflow를 통해서 도커 이미지를 생성하고 이를 활용하여 Google Cloud Artifact Registry, Cloud Run을 활용한 배포를 자동화할 수 있도록 구성하였습니다. 
          </p>
      </li>
      <li>
        <h3>✔️ 대화 이력 요약</h3> 
          <p>
            긴 대화 맥락을 요약하여 프롬프트 크기를 최적화하고, 최근 대화에 집중할 수 있도록 설계하였습니다.
          </p>
      </li>
      <li>
        <h3>✔️ UI 커스터마이징</h3> 
          <p>
            Gradio 활용하여 채팅 UI를 구현하였습니다.
          </p>
      </li>
      <li>
        <h3>✔️ Answer Style Control</h3> 
          <p>초기에는 답변이 형식적이었으나, "스타일 보정 Agent"를 추가하여 인터뷰 톤을 유지하도록 개선했습니다.</p>
      </li>
    </ul>
</div>

<br/>
<br/>

<div>
    <h2 id="infrastructure">🔦 Product Infrastructure</h2>
    <img src="docs/infrastructure.png" width="90%">
</div>

<br/>
<br/>

<div>
    <h2 id="tech">🛠 Tech Stack Used</h2>
    <ul>
      <li>
        <h4>Frontend (UI)</h4> 
        <p>Gradio, Custom CSS</p>
      </li>
      <li>
        <h4>Backend</h4> 
        <p>Python, AsyncOpenAI, LangChain, ChromaDB</p>
      </li>
      <li>
        <h4>Data</h4> 
        <p>Google Cloud Storage (Resume, Summary)</p>
      </li>
      <li>
        <h4>Deploy</h4> 
        <p>Docker, Google Cloud Run</p>
      </li>
    </ul>
</div>

<br/>
<br/>

<div id="feature">
</div>
    
🪵 Features
--
<h4> 🥕 이력서/요약 기반 질문 응답 (RAG) </h4>
<h4> 🥕 질문 분류 및 카테고리화 (프로젝트 경험, 기술 스택 등) </h4>
<h4> 🥕 캐시 시스템을 통한 응답 속도 최적화 </h4>
<h4> 🥕 Gradio 기반 인터랙티브 채팅 UI </h4>
<h4> 🥕 구어체 면접 톤 응답 (스타일 보정) </h4>
<h4> 🥕 대화 이력 요약 및 컨텍스트 관리 </h4>
<h4> 🥕 Google Cloud Storage 연동 (PDF/텍스트 형식의 초기 데이터 관리) </h4>

<br/>
<br/>
<br/>


