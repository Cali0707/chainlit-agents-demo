# LLM Agents with Chainlit Demo

This repo contains a simple demo of how to add tools to LLM Agents using Chainlit for the frontend,
and Langchain + Langgraph for the LLM Agent. To add more tools, add them to the `tools` directory
and add them into the tools list in `chat.py`

Note: you will need to copy `.env.default` into a new file called `.env`, and set the `OPENAI_API_KEY`
to your own api key in order for this to run.
