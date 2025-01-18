"""
Copyright 2025 Calum Murray

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
from typing import Sequence, TypedDict, Annotated
import operator
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools.render import format_tool_to_openai_tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.globals import set_debug

from langgraph.prebuilt import ToolExecutor, ToolInvocation, tool_executor
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from pprint import pprint

import chainlit as cl

from tools import WordLength

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    chat_history: Sequence[BaseMessage]

@cl.on_chat_start
async def on_chat_start():
    # set up history for this chat
    chat_history = ConversationBufferMemory(return_messages=True)
    cl.user_session.set("chat_history", chat_history)

    # create the model instance
    model = ChatOpenAI(temperature=0.1, streaming=True, max_retries=5, timeout=60.)

    tools = [WordLength()]

    tool_executor = ToolExecutor(tools)

    tools = [format_tool_to_openai_tool(t) for t in tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                    "system",
                    "You are a helpful AI assistant. "
                    "Use the provided tools to progress towards answering the question. "
                    "If you do not have the final answer yet, prefix your response with CONTINUE. "
                    "You have access to the following tools: {tool_names}."
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    prompt = prompt.partial(tool_names=", ".join(tool["function"]["name"] for tool in tools))

    model = prompt | model.bind_tools(tools)

    def next_state(state: AgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        print(messages)
        print(last_message)
        if "tool_calls" in last_message.additional_kwargs:
            return "tool_call"
        elif "CONTINUE" not in last_message.content:
            return "end"
        else:
            return "model_call"

    async def call_model(state: AgentState):
        response = await model.ainvoke(state)
        return {"messages": [response]}

    async def call_tool(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        tool_calls = []
        tasks = []
        for tool_call in last_message.additional_kwargs["tool_calls"]:
            function_name = tool_call["function"]["name"]
            action = ToolInvocation(
                    tool=function_name,
                    tool_input=json.loads(tool_call["function"]["arguments"]),
            )
            tool_calls.append(action)
            tasks.append(cl.Task(title=action.tool, status=cl.TaskStatus.RUNNING))

        task_list = cl.user_session.get("task_list")
        if task_list is None:
            task_list = cl.TaskList()

        await asyncio.gather(*[task_list.add_task(task) for task in tasks])
        await task_list.send()

        responses = await asyncio.gather(*[tool_executor.ainvoke(tool_call) for tool_call in tool_calls])

        tool_messages = []
        for i in range(len(responses)):
            tool_messages.append(ToolMessage(tool_call_id=last_message.additional_kwargs["tool_calls"][i]["id"], content=str(responses[i]), name=tool_calls[i].tool))

        for task in tasks:
            task.status = cl.TaskStatus.DONE
        await task_list.send()

        cl.user_session.set("task_list", task_list)

        return {"messages": tool_messages}

    graph = StateGraph(AgentState)

    graph.add_node("model", call_model)
    graph.add_node("tool", call_tool)

    graph.set_entry_point("model")

    graph.add_conditional_edges(
            "model",
            next_state,
            {
                "tool_call": "tool",
                "model_call": "model",
                "end": END,
            },
    )

    graph.add_edge("tool", "model")

    memory = MemorySaver()

    runner = graph.compile(checkpointer=memory)

    cl.user_session.set("runner", runner)


@cl.on_message
async def main(message: cl.Message):
    runner = cl.user_session.get("runner") # Type: CompilerGraph
    if runner is None:
        print("error: runner is none")
        return

    inputs = {
            "messages": [HumanMessage(content=message.content)],
    }

    cl.user_session.set("task_list", None)

    msg = cl.Message(content="")

    chunk = ""
    async for event in runner.astream_events(inputs, {"configurable": {"thread_id": "thread-1"}}, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream" or kind == "on_chain_stream":
            data = event.get("data", {})
            response_chunk = data.get("chunk", {})
            if response_chunk is None:
                continue

            messages = response_chunk.get("messages", [AIMessage(content="")])
            content = messages[0].content
            if content and not isinstance(messages[0], ToolMessage):
                chunk += content
                if chunk.strip() not in "CONTINUE":
                    await msg.stream_token(content)
                elif chunk.strip() == "CONTINUE":
                    chunk = ""
    
    await msg.send()
