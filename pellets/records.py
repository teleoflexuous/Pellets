from pydantic import BaseModel, Field, UUID4, Json, ConfigDict
from typing import Any
import uuid


class PromptArgumentsCreate(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
    )

    model: str = Field(description="The model to use for the prompt")
    prompt: str = Field(description="The prompt to use for the model")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="The parameters to use for the model",
    )


class PromptArgumentsRecord(PromptArgumentsCreate):
    id: int = Field(
        description="The unique identifier for the prompt arguments",
    )


class ResponseCreate(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
    )
    response: str = Field(description="The response to the prompt")
    prompt_arguments_id: int = Field(
        description="The unique identifier for the prompt arguments",
    )


class ResponseRecord(ResponseCreate):
    id: int = Field(
        description="The unique identifier for the response",
    )


class ResponseStreamRecord(ResponseCreate):
    id: int = Field(
        description="The unique identifier for the response stream",
    )
    prompt_arguments_id: int = Field(
        description="The unique identifier for the prompt arguments",
    )
    done: bool = Field(description="Whether the response stream is done")
    previous_response_id: int | None = Field(
        description="The unique identifier for the previous response",
    )
