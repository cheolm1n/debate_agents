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
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="이기는 편 우리 편",
    layout="wide",
)


### Tool 생성 --------------------------------------------------------------------------------------------------------------------
### 인터넷 검색 도구 
# Tavily 라는 검색 도구 사용 
search = TavilySearchResults(k=5) # 검색결과 5개 가져오도록.

# 토론 주제 선정
topic = st.chat_input("토론 주제를 입력해주세요.")
word_limit = 50

# 검색 기반 도구로 할때 
names_search = {
    "Pro(찬성)": [search],  # 의사협회 에이전트 도구 목록
    "Con(반대)": [search],  # 정부 에이전트 도구 목록
}


## 토론 Agent의 시스템 메시지 생성 --------------------------------------------------------------------------------------------------------------------

# 화자 설명 생성 
conversation_description = f"""Here is the topic of conversation: {topic}
Identify the groups that represent support and opposition on a topic and describe their characteristics."""

agent_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of the conversation participant."
)

def generate_agent_description(name):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(
            content=f"""{conversation_description}
            Please reply with a description of {name}, in {word_limit} words or less in expert tone. 
            Speak directly to {name}.
            Give them a point of view.
            Do not add anything else. Answer in KOREAN."""
        ),
    ]
    agent_description = ChatOpenAI(temperature=0)(agent_specifier_prompt).content
    return agent_description


# 각 참가자의 이름에 대한 에이전트 설명을 생성
agent_descriptions = {name: generate_agent_description(name) for name in names_search}


## 세부 주제 설정
# model = ChatOpenAI(temperature=1.0)#(topic_specifier_prompt).content
# parser = JsonOutputParser()
# prompt = PromptTemplate(
#     template="""
#         You are the moderator. 
#         Please break down the topic '{topic}' into specific subtopics for discussion.
#         Please reply with the specified quest in 100 words or less.
#         Consider the participants: {agent_descriptions}.  
#         Do not add anything else.
#         Answer in Korean.
        
#         topic : 2024 현재, 한일 관계 개선을 위해 강제징용 배상 문제를 조속히 해결해야 하는가?
#         answer :
#         '1. 강제징용 배상 문제 해결의 중요성에 대한 의견은?\n2. 강제징용 문제 해결이 한일 관계의 안정과 협력에 미치는 영향은?\n3. 강제징용 문제 해결로 상호 신뢰와 협력을 증진시킬 수 있는 방안은 무엇인가요?\n4. 강제징용 문제 해결이 지역 안보와 경제 발전에 미치는 긍정적인 영향은 무엇인가요?\n5. 강제징용 문제 해결을 통해 역사적 상처를 치유하며 미래를 위한 건설적인 관계 구축에 어떻게 기여할 수 있을까요?'
#         topic : {topic}
#         answer :
#         '""",
#     input_variables=["topic", "agent_descriptions"],
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )

# chain = prompt | model #| parser
# response = chain.invoke({"topic": topic, "agent_descriptions": agent_descriptions}).content
# subtopics = response.split('\n')

# 주제 소개
topic_specifier_prompt = [
    # 주제를 더 구체적으로 만들 수 있습니다.
    SystemMessage(content="You can make a topic more specific."),
    HumanMessage(
        content=f"""{topic}
        
        You are the moderator. 
        Please make the topic more specific.
        Please reply with the specified quest in 100 words or less.
        Consider the participants: {agent_descriptions}.  
        Introduce the topic and start the debate.
        Do not add anything else.
        Answer in Korean.""" 
    ),
]

# 구체화된 주제를 생성합니다.
specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content


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
    for (name, tools), description in zip(names_search.items(), agent_descriptions.values())
}


# AGENT 생성 --------------------------------------------------------------------------------------------------------------------
agents = [
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


# 다음 발언자를 선택하도록 하는 함수
def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx


# 토론 시작 --------------------------------------------------------------------------------------------------------------------
max_iters = 6  
n = 0  # 반복 횟수 초기화

# DialogueSimulator 객체를 생성하고, agents와 select_next_speaker 함수를 전달
simulator = DialogueSimulator(
    agents=agents, selection_function=select_next_speaker)

# 시뮬레이터 초기화
simulator.reset()

# Moderator가 주제를 제시
simulator.inject("Moderator", specified_topic)



# streamlit --------------------------------------------------------------------------------------------------------------------

st.markdown("# AI Vs AI 🥊")

st.markdown("AI에게 페르소나를 부여하여 Agent로 인터넷 검색 Tool을 주어 토론하게 하였습니다.")

st.markdown("토론 주제는 찬성과 반대로 나뉘어 토론할 수 있는 주제로 입력해주세요!")
st.markdown("#### 💡 GPT가 추천해준 대한민국 최근 이슈들 중 찬성 반대로 토론할만한 주제")
st.markdown("* 2024년 현재, 대한민국 대학교 의대 정원 확대 충원은 필요한가?")
st.markdown("* 2024 현재, 한일 관계 개선을 위해 강제징용 배상 문제를 조속히 해결해야 하는가?")
st.markdown("* 2024 현재, 대한민국의 주택 시장 안정을 위해 현행 부동산 규제 정책이 필요한가?")
st.markdown("* 2024 현재, 청년 실업 문제 해결을 위해 정부의 적극적인 일자리 개입이 필수적인가?")
st.markdown("* 2024 현재, 디지털 전환 시대에 개인정보 보호 강화를 위해 규제를 더욱 강화해야 하는가?")
st.markdown("* 2024 현재, 대한민국의 지속 가능한 성장을 위해 탄소중립 정책을 강력히 추진해야 하는가?")


# 화자 정의 
speakers = {
    "Pro(찬성)": "🙆‍♂️",
    "Con(반대)": "🙅‍♂️",
    "사회자": "🤖"
}

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []


if topic:
    # 사용자 메시지 추가
    # st.session_state.messages.append({"role": "user", "content": topic, "avatar": "🧑"})
    with st.chat_message("user", avatar="🧑"):
        st.write(topic)

    # 사회자 메시지 추가
    # st.session_state.messages.append({"role": "assistant", "content": specified_topic, "avatar": "🤖"})
    with st.chat_message("assistant", avatar="🤖"):
        st.write(specified_topic)
    
    while n < max_iters:  # 최대 반복 횟수까지 반복합니다.
        name, message = (
            simulator.step()
        )  # 시뮬레이터의 다음 단계를 실행하고 발언자와 메시지를 받아옵니다.
        
        # st.session_state.messages.append({"role": "assistant", "content": message, "avatar": speakers[name]})

        # 대화 기록 표시
        with st.chat_message("assistant", avatar=speakers[name]):
            st.write(message)
            
        n += 1
            
        
        