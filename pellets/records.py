from pydantic import (
    BaseModel,
    Field,
    validator,
    UUID4,
    ConfigDict,
    SecretStr,
    create_model,
)
import uuid

from pellets.config import config

PromptArgumentsCreate = create_model(
    "PromptArgumentsCreate",
    # add from_attributes=True to configdict
    **config.prompt_arguments_schema,
    __config__=ConfigDict(from_attributes=True)
)

PromptArgumentsRecord = create_model(
    "PromptArguments",
    id=(
        UUID4,
        Field(description="The unique identifier for the prompt arguments"),
    ),
    # add from_attributes=True to configdict
    **config.prompt_arguments_schema,
    __config__=ConfigDict(from_attributes=True)
)

ResponseCreate = create_model(
    "ResponseCreate",
    # add from_attributes=True to configdict
    **config.response_schema,
    # Relationship between the prompt arguments and the response
    prompt_arguments_id=(UUID4, Field(description="The unique identifier for the prompt arguments")),
    __config__=ConfigDict(from_attributes=True)
)

ResponseRecord = create_model(
    "Response",
    id=(
        UUID4,
        Field(description="The unique identifier for the response"),
    ),
    # add from_attributes=True to configdict
    **config.response_schema,
    # Relationship between the prompt arguments and the response
    prompt_arguments_id=(UUID4, Field(description="The unique identifier for the prompt arguments")),
    __config__=ConfigDict(from_attributes=True)
)