    
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from typing import Callable, List
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        # 에이전트의 이름을 설정합니다.
        self.name = name
        # 시스템 메시지를 설정합니다.
        self.system_message = system_message
        # LLM 모델을 설정합니다.
        self.model = model
        # 에이전트 이름을 지정합니다.
        self.prefix = f"{self.name}: "
        # 에이전트를 초기화합니다.
        self.reset()

    def reset(self):
        """
        대화 내역을 초기화합니다.
        """
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        메시지에 시스템 메시지 + 대화내용과 마지막으로 에이전트의 이름을 추가합니다.
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join([self.prefix] + self.message_history)),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        name 이 말한 message 를 메시지 내역에 추가합니다.
        """
        self.message_history.append(f"{name}: {message}")
        
class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        # 에이전트 목록을 설정합니다.
        self.agents = agents
        # 시뮬레이션 단계를 초기화합니다.
        self._step = 0
        # 다음 발언자를 선택하는 함수를 설정합니다.
        self.select_next_speaker = selection_function

    def reset(self):
        # 모든 에이전트를 초기화합니다.
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        name 의 message 로 대화를 시작합니다.
        """
        # 모든 에이전트가 메시지를 받습니다.
        for agent in self.agents:
            agent.receive(name, message)

        # 시뮬레이션 단계를 증가시킵니다.
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. 다음 발언자를 선택합니다.
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. 다음 발언자에게 메시지를 전송합니다.
        message = speaker.send()

        # 3. 모든 에이전트가 메시지를 받습니다.
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. 시뮬레이션 단계를 증가시킵니다.
        self._step += 1

        # 발언자의 이름과 메시지를 반환합니다.
        return speaker.name, message



class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
        tools,
    ) -> None:
        # 부모 클래스의 생성자를 호출합니다.
        super().__init__(name, system_message, model)
        # 주어진 도구 이름과 인자를 사용하여 도구를 로드합니다.
        self.tools = tools

    def send(self) -> str:
        """
        메시지 기록에 챗 모델을 적용하고 메시지 문자열을 반환합니다.
        """
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_tools_agent(self.model, self.tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=self.tools, verbose=False)
        # AI 메시지를 생성합니다.
        message = AIMessage(
            content=agent_executor.invoke(
                {
                    "input": "\n".join(
                        [self.system_message.content]
                        + [self.prefix]
                        + self.message_history
                    )
                }
            )["output"]
        )

        # 생성된 메시지의 내용을 반환합니다.
        return message.content