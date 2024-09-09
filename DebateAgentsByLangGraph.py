import streamlit as st
from typing import List
from agents import DialogueAgentWithTools, DialogueSimulator
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv

load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(page_title="이기는 편 우리 편", layout="wide")

# 검색 도구 초기화
search_tool = TavilySearchResults(k=5)
names_search = {"Pro(찬성)": [search_tool], "Con(반대)": [search_tool]}

# 토론 주제 입력
topic = st.chat_input("토론 주제를 입력해주세요.")
word_limit = 50


# 시스템 메시지와 에이전트 설명 생성
def generate_agent_description(name, topic):
    agent_specifier_prompt = [
        SystemMessage(content="You can add detail to the description of the conversation participant."),
        HumanMessage(
            content=f"Here is the topic of conversation: {topic}. Please reply with a description of {name}, in {word_limit} words or less in expert tone. Speak directly to {name}. Answer in KOREAN.")
    ]
    return ChatOpenAI(temperature=0)(agent_specifier_prompt).content


# 각 참가자에 대한 설명 생성
agent_descriptions = {name: generate_agent_description(name, topic) for name in names_search}


# 토론 주제를 구체화
def specify_topic(topic, agent_descriptions):
    prompt = [
        SystemMessage(content="You can make a topic more specific."),
        HumanMessage(content=f"{topic}\nPlease make the topic more specific. Consider the participants: {agent_descriptions}. Answer in Korean.")
    ]
    return ChatOpenAI(temperature=1.0)(prompt).content


specified_topic = specify_topic(topic, agent_descriptions)


# 에이전트의 시스템 메시지 생성
def generate_system_message(name, description, tools):
    return f"""Here is the topic of conversation: {topic}
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
Answer in KOREAN."""


agent_system_messages = {
    name: generate_system_message(name, description, tools)
    for (name, tools), description in zip(names_search.items(), agent_descriptions.values())
}

# 에이전트 생성
agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.2),
        tools=tools
    )
    for (name, tools), system_message in zip(names_search.items(), agent_system_messages.values())
]


# 다음 발언자를 선택하는 함수
def select_next_speaker(step: int, agents: List[DialogueAgentWithTools]) -> int:
    return step % len(agents)


# 시뮬레이터 생성 및 토론 시작
simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
simulator.reset()
simulator.inject("Moderator", specified_topic)

# Streamlit 인터페이스
st.markdown("# AI Vs AI 🥊")
st.markdown("AI에게 페르소나를 부여하여 Agent로 인터넷 검색 Tool을 주어 토론하게 하였습니다.")
st.markdown("토론 주제는 찬성과 반대로 나뉘어 토론할 수 있는 주제로 입력해주세요!")

# 사용자에게 주제 출력
if topic:
    with st.chat_message("user", avatar="🧑"):
        st.write(topic)

    with st.chat_message("assistant", avatar="🤖"):
        st.write(specified_topic)

    # 최대 반복 횟수만큼 토론을 진행
    for n in range(6):
        name, message = simulator.step()
        with st.chat_message("assistant", avatar={"Pro(찬성)": "🙆‍♂️", "Con(반대)": "🙅‍♂️"}[name]):
            st.write(message)
