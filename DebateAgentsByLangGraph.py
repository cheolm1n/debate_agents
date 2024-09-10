import streamlit as st
from dotenv import load_dotenv
from langchain.schema import SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage

load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(page_title="이기는 편 우리 편", layout="wide")

# 검색 도구 초기화
search_tool = TavilySearchResults(k=5)
names_search = {"Pro(찬성)": [search_tool], "Con(반대)": [search_tool]}

# 토론 주제 입력
topic = st.chat_input("토론 주제를 입력해주세요.")
word_limit = 50

agent_descriptions = {}
pros_agent = None
cons_agent = None


# 토론 참가자에 대한 설명 생성
def generate_agent_description(name, topic):
    agent_specifier_prompt = [
        SystemMessage(content="You can add detail to the description of the conversation participant."),
        HumanMessage(
            content=f"Here is the topic of conversation: {topic}. Please reply with a description of {name}, in {word_limit} words or less in expert tone. Speak directly to {name}. Answer in KOREAN.")
    ]
    return ChatOpenAI(temperature=0)(agent_specifier_prompt).content


# 토론 참가자 정보를 통해 주제를 더 구체화
def specify_topic(topic, agent_descriptions):
    prompt = [
        SystemMessage(content="You can make a topic more specific."),
        HumanMessage(content=f"{topic}\nPlease make the topic more specific. Consider the participants: {agent_descriptions}. Answer in Korean.")
    ]
    return ChatOpenAI(temperature=1.0)(prompt).content


# 각 토론자 에이전트의 시스템 메시지 생성
# def generate_system_message(name, description, tools):
#     return f"""Here is the topic of conversation: {topic}
# Your name is {name}.
# Your description is as follows: {description}
# Your goal is to persuade your conversation partner of your point of view.
# DO look up information with your tool to refute your partner's claims.
# DO cite your sources.
# DO NOT fabricate fake citations.
# DO NOT cite any source that you did not look up.
# DO NOT restate something that has already been said in the past.
# DO NOT add anything else.
# DO NOT speak from the perspective of other participants.
# Stop speaking the moment you finish speaking from your perspective.
# Answer in KOREAN."""


# 다음 발언자를 선택하는 함수
# def select_next_speaker(step: int, agents: List[DialogueAgentWithTools]) -> int:
#     return step % len(agents)


# 시뮬레이터 생성 및 토론 시작
# simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
# simulator.reset()
# simulator.inject("Moderator", specified_topic)

# Streamlit 인터페이스
st.markdown("# AI Vs AI 🥊")
st.markdown("AI에게 페르소나를 부여하여 Agent로 인터넷 검색 Tool을 주어 토론하게 하였습니다.")
st.markdown("토론 주제는 찬성과 반대로 나뉘어 토론할 수 있는 주제로 입력해주세요!")

# 사용자에게 주제 출력
# if topic:
#     with st.chat_message("user", avatar="🧑"):
#         st.write(topic)
#
#     with st.chat_message("assistant", avatar="🤖"):
#         st.write(specified_topic)
#
#     # 최대 반복 횟수만큼 토론을 진행
#     for n in range(6):
#         name, message = simulator.step()
#         with st.chat_message("assistant", avatar={"Pro(찬성)": "🙆‍♂️", "Con(반대)": "🙅‍♂️"}[name]):
#             st.write(message)

#

# 각 참가자에 대한 설명 생성
# agent_descriptions = {name: generate_agent_description(name, topic) for name in names_search}
#
# specified_topic = specify_topic(topic, agent_descriptions)

# agent_system_messages = {
#     name: generate_system_message(name, description, tools)
#     for (name, tools), description in zip(names_search.items(), agent_descriptions.values())
# }
# 각 토론자 에이전트 생성
# agents = [
#     DialogueAgentWithTools(
#         name=name,
#         system_message=SystemMessage(content=system_message),
#         model=ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.2),
#         tools=tools
#     )
#     for (name, tools), system_message in zip(names_search.items(), agent_system_messages.values())
# ]

# LangGraph
import operator
from typing import Annotated, Sequence
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    topic: str


from langchain_core.messages import (
    FunctionMessage,
    HumanMessage,
)

from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph


def create_agent(llm, tools, system_message: str):
    # 에이전트를 생성합니다.
    functions = [format_tool_to_openai_function(t) for t in tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join(
        [tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)


import functools


def agent_node(state, agent, name):
    result = agent.invoke(state)

    st.write(result)

    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }


def topic_agent_node(state, name):
    agent_descriptions = {name: generate_agent_description(name, topic) for name in names_search}
    st.write(agent_descriptions)
    state["topic"] = specify_topic(topic, agent_descriptions)
    st.write(state["topic"])
    state["sender"] = name
    pros_agent = create_agent(
        llm,
        [search_tool],
        system_message=agent_descriptions["Pros(찬성)"],
    )
    cons_agent = create_agent(
        llm,
        [search_tool],
        system_message=agent_descriptions["Cons(찬성)"],
    )
    return state


# def moderator_agent_node(state, name):
#
#
#     return state

llm = ChatOpenAI(model="gpt-4-1106-preview")

# Pros/Cons agent and node
topic_agent = create_agent(llm, [search_tool],
                           system_message=f"You can make a topic more specific. {topic}\nPlease make the topic more specific. Consider the participants: {agent_descriptions}. Answer in Korean.")
topic_node = functools.partial(topic_agent_node, agent=topic_agent, name="topic")

# moderator_agent = create_agent(llm, [search_tool],
#                                system_message=f"You can make a topic more specific. {topic}\nPlease make the topic more specific. Consider the participants: {agent_descriptions}. Answer in Korean.")
# moderator_node = functools.partial(agent_node, agent=moderator_agent, name="moderator")


pros_node = functools.partial(agent_node, agent=pros_agent, name="pros")

cons_node = functools.partial(agent_node, agent=cons_agent, name="cons")


def router(state):
    # 상태 정보를 기반으로 다음 단계를 결정하는 라우터 함수
    now = state["turn"]

    if now > 6:
        return "end"
    elif now % 2 != 0:
        return "pros"
    else:
        return "cons"


workflow = StateGraph(AgentState)

workflow.add_node("Topic", topic_node)
workflow.add_node("Pros", pros_node)
workflow.add_node("Cons", cons_node)
# workflow.add_node("Moderator", moderator_node)

workflow.add_edge("Topic", "Pros")
# workflow.add_edge("Topic", "Moderator")
# workflow.add_edge("Moderator", "Pros")

workflow.add_conditional_edges(
    "Pros",
    router,
    {"pros": "Pros", "cons": "Cons", "end": END},
)

workflow.add_conditional_edges(
    "Cons",
    router,
    {"pros": "Pros", "cons": "Cons", "end": END},
)

workflow.set_entry_point("Topic")
graph = workflow.compile()

if topic:
    # 사용자 메시지 추가
    # st.session_state.messages.append({"role": "user", "content": topic, "avatar": "🧑"})
    with st.chat_message("user", avatar="🧑"):
        st.write(topic)

    graph.invoke() ## how?

