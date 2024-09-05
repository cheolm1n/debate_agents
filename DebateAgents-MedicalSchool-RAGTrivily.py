import streamlit as st
from typing import Callable, List
from agents import DialogueAgent, DialogueSimulator, DialogueAgentWithTools
from langchain.tools.retriever import create_retriever_tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

### Tool 생성 --------------------------------------------------------------------------------------------------------------------
### RAG
# Retriever를 생성
vector1 = FAISS.load_local('db/vector1', OpenAIEmbeddings(), allow_dangerous_deserialization=True)
vector2 = FAISS.load_local('db/vector2', OpenAIEmbeddings(), allow_dangerous_deserialization=True)

doctor_retriever = vector1.as_retriever(search_kwargs={"k": 5})
gov_retriever = vector2.as_retriever(search_kwargs={"k": 5})

doctor_retriever_tool = create_retriever_tool(
    doctor_retriever,
    name="document_search", # 밑에처럼 설명을 넣어주는게 좋다고 함. 
    description="This is a document about the Korean Medical Association's opposition to the expansion of university medical schools. "
    "Refer to this document when you want to present a rebuttal to the proponents of medical school expansion.",
)

gov_retriever_tool = create_retriever_tool(
    gov_retriever,
    name="document_search",
    description="This is a document about the Korean government's support for the expansion of university medical schools. "
    "Refer to this document when you want to provide a rebuttal to the opposition to medical school expansion.",
)

### 인터넷 검색 도구 
# Tavily 라는 검색 도구 사용 
search = TavilySearchResults(k=5) # 검색결과 5개 가져오도록.

# 토론 주제 선정
topic = "2024 현재, 대한민국 대학교 의대 정원 확대 충원은 필요한가?"

# RAG로 할 때
names = {
    "Doctor Union(의사협회)": [doctor_retriever_tool],  # 의사협회 에이전트 도구 목록
    "Government(대한민국 정부)": [gov_retriever_tool],  # 정부 에이전트 도구 목록
}

# 검색 기반 도구로 할때 
names_search = {
    "Doctor Union(의사 협회)": [search],  # 의사협회 에이전트 도구 목록
    "Government(대한민국 정부)": [search],  # 정부 에이전트 도구 목록
}


## 토론 Agent의 시스템 메시지 생성 --------------------------------------------------------------------------------------------------------------------
conversation_description = f"""Here is the topic of conversation: {topic}
The participants are: {', '.join(names.keys())}"""

agent_descriptions = {
    "Doctor Union(의사협회)": "의사협회는 의료계의 권익을 보호하고 의사들의 이해관계를 대변하는 기관입니다. 의사들의 업무 환경과 안전을 중시하며, 환자 안전과 질 높은 의료 서비스를 제공하기 위해 노력합니다. "
    "지금도 의사의 수는 충분하다는 입장이며, 의대 증원은 필수 의료나 지방 의료 활성화에 대한 실효성이 떨어집니다. 의대 증원을 감행할 경우, 의료 교육 현장의 인프라가 갑작스러운 증원을 감당하지 못할 것이란 우려를 표합니다.",
    "Government(대한민국 정부)": "대한민국 정부는 국가의 행정을 책임지는 주체로서, 국민의 복지와 발전을 책임져야 합니다. "
    "우리나라는 의사수가 절대 부족한 상황이며, 노인인구가 늘어나면서 의료 수요가 급증하고 있습니다. OECD 국가들도 최근 의사수를 늘렸습니다. 또한, 증원된 의사 인력이 필수의료와 지역 의료로 갈 수있도록 튼튼한 의료사고 안정망 구축 및 보상 체계의 공정성을 높이고자 합니다.",
}

# - 에이전트의 이름과 설명을 알립니다.
# - 에이전트는 도구를 사용하여 정보를 찾고 대화 상대방의 주장을 반박해야 합니다.
# - 에이전트는 출처를 인용해야 하며, 가짜 인용을 하거나 찾아보지 않은 출처를 인용해서는 안 됩니다.
# - 에이전트는 자신의 관점에서 말을 마치는 즉시 대화를 중단해야 합니다.

def generate_system_message(name, description, tools):
    return f"""{conversation_description}
    
Your name is {name}.

Your description is as follows: {description}

Your goal is to persuade your conversation partner of your point of view.

DO look up information with your tool to refute your partner's claims. 
DO cite your sources.

DO NOT fabricate fake citations.
DO NOT cite any source that you did not look up.

DO NOT restate something that has already been said in the past.
DO NOT add anything else.

DO NOT speak from the perspective of other participants.

Stop speaking the moment you finish speaking from your perspective.

Answer in KOREAN.
"""


agent_system_messages = {
    name: generate_system_message(name, description, tools)
    for (name, tools), description in zip(names.items(), agent_descriptions.values())
}

# 세부 주제 설정
specified_topic = "정부는 2025년 입시부터 의대 입학정원을 2000명 늘린다고 발표했습니다. 이에 의사단체는 전국에서 규탄집회를 열어 반발하고 있습니다. 의대 정원 확대를 둘러싼 논란 쟁점을 짚어보고, 필수 의료와 지역 의료 해법에 대해서 토론해주세요."

# AGENT 생성 --------------------------------------------------------------------------------------------------------------------

agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.2),
        tools=tools,
    )
    for (name, tools), system_message in zip(
        names.items(), agent_system_messages.values()
    )
]

agents_with_search = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.2),
        tools=tools,
    )
    for (name, tools), system_message in zip(
        names_search.items(), agent_system_messages.values()
    )
]

agents.extend(agents_with_search)

# 다음 발언자를 선택하도록 하는 함수
def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx

# 토론 시작 --------------------------------------------------------------------------------------------------------------------
max_iters = 3  # 최대 반복 횟수를 6으로 설정
n = 0  # 반복 횟수를 추적하는 변수를 0으로 초기화

# DialogueSimulator 객체를 생성하고, agents와 select_next_speaker 함수를 전달
simulator = DialogueSimulator(
    agents=agents_with_search, selection_function=select_next_speaker)

# 시뮬레이터를 초기 상태로 리셋
simulator.reset()

# Moderator가 지정된 주제를 제시
simulator.inject("Moderator", specified_topic)



# streamlit --------------------------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="이기는 편 우리 편",
    layout="wide",
)

st.markdown("# AI Vs AI 🥊")

st.markdown("AI에게 페르소나를 부여하여 Agent로 RAG, 인터넷 검색 두 가지 Tool을 주어 토론하게 하였습니다.")

st.markdown("토론 주제는 의대 입학 정원 확대로 **의사 협회측 AI**와 **정부측 AI**가 토론을 진행하도록 하겠습니다.")


# 화자 정의 
speakers = {
    "Doctor Union(의사 협회)": "🧑‍⚕️",
    "Government(대한민국 정부)": "👨‍⚖️",
    "사회자": "🤖"
}

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사용자 입력
user_input = st.chat_input("메시지를 입력하세요.")

if user_input:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "🧑"})
    with st.chat_message("user", avatar="🧑"):
        st.write(user_input)

    # 사회자 메시지 추가
    st.session_state.messages.append({"role": "assistant", "content": specified_topic, "avatar": "🤖"})
    with st.chat_message("assistant", avatar="🤖"):
        st.write(specified_topic)
    
    while n < max_iters:  # 최대 반복 횟수까지 반복합니다.
        name, message = (
            simulator.step()
        )  # 시뮬레이터의 다음 단계를 실행하고 발언자와 메시지를 받아옵니다.
        
        st.session_state.messages.append({"role": "assistant", "content": message, "avatar": speakers[name]})

        # 대화 기록 표시
        with st.chat_message("assistant", avatar=speakers[name]):
            st.write(message)
            
        n += 1
            
        
        