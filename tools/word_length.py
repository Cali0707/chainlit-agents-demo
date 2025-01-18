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
