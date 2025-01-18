from typing import Type
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field

class WordLengthQuery(BaseModel):
    """Input for the WordLength tool"""

    word: str = Field(description="the word to get the length of")

class WordLength(BaseTool):
    """WordLength is a tool that provides the exact length of a given word or string"""

    name: str = "WordLength"
    description: str = (
        "Get the exact length (number of characters) for a given word or string. "
        "Use this whenever you need to determine the length of a word or string."
    )
    args_schema: Type[BaseModel] = WordLengthQuery

    def _run(
        self,
        word: str,
        run_manager=None,
    ) -> str:
        """Use the WordLength tool synchronously"""
        return f"length = {len(word)}"

    async def _arun(
        self,
        word: str,
        run_manager=None,
    ) -> str:
        return self._run(word)
