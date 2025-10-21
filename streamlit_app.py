# app_streamlit.py
import os, re, json, datetime, requests
import streamlit as st
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import initialize_agent, AgentType, load_tools
from langchain_classic.tools import Tool
from langchain_community.tools import ShellTool

shell_tool = ShellTool()
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.retrievers import WikipediaRetriever
from langchain_classic.schema import HumanMessage, AIMessage, SystemMessage

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4.1",
    openai_api_key="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",
    openai_api_base="https://models.github.ai/inference",  # 👈 이게 base_url 역할
)
# ──────────────────────────────
# ✅ Tools, Memory, and Agent Setup
# ──────────────────────────────
tools = load_tools(["ddg-search", "requests_all", "llm-math"], llm=llm, allow_dangerous_tools=True)
tools.append(Tool(name="python_repl", func=PythonREPLTool().run, description="Python 코드 실행 도구"))
tools.append(shell_tool)
retriever = WikipediaRetriever(lang="ko")
tools.append(Tool(name="wiki", func=retriever.get_relevant_documents, description="위키백과 검색"))
tools.append(Tool(name="time_now", func=lambda _: f"현재 시각: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Asia/Seoul)", description="현재 시간을 반환합니다."))

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# ──────────────────────────────
# ✅ Streamlit UI
# ──────────────────────────────
st.set_page_config(page_title="PIXAL Assistant", page_icon="🤖", layout="wide")

st.markdown("""
<h2 style="text-align:center;">🤖 PIXAL Assistant</h2>
<p style="text-align:center;color:gray;">LangChain + GitHub Models Integration</p>
<hr>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input field
if prompt := st.chat_input("메시지를 입력하세요..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent
    with st.chat_message("assistant"):
        with st.spinner("PIXAL이 생각 중입니다..."):
            try:
                response = agent.run(prompt)
                text = str(response)
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    try:
                        obj = json.loads(match.group(0))
                        text = obj.get("action_input") or obj.get("Final Answer") or obj.get("content") or text
                    except Exception:
                        pass
            except Exception as e:
                text = f"⚠️ 오류 발생: {e}"

            st.markdown(text)
            st.session_state["messages"].append({"role": "assistant", "content": text})

st.sidebar.markdown("### ⚙️ 옵션")
if st.sidebar.button("🧹 대화 초기화"):
    st.session_state["messages"] = []
    memory.clear()
    st.rerun()

st.sidebar.markdown("Made with ❤️ by PIXAL")
