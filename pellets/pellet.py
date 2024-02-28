# The library is expected to be able to:
# Log prompt, model, and response to a sqlite database
# One of the interfaces for performing this is a decorator
# The decorator is expected to be used on top of a async request handler
# Response can come in one piece or be streamed

import sqlite3
import json
import pydantic
import asyncio
import uuid
import inspect
from typing import Type
from itertools import chain
import requests

from pellets.config import config
from pellets.records import (
    PromptArgumentsRecord,
    ResponseRecord,
    PromptArgumentsCreate,
    ResponseCreate,
)
from pellets.logger import logger


class Pellet:
    def __init__(
        self,
        db_path: str = config.db_path,
        prompt_arguments_schema: dict = config.prompt_arguments_schema,
        response_schema: dict = config.response_schema,
    ):
        self.db_path = db_path
        self.prompt_arguments_schema = prompt_arguments_schema
        self.response_schema = response_schema
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.__create_tables()

    def __create_tables(self):
        # Tables will always contain:
        # id: uuid.uuid4()
        # prompt_arguments_id: uuid.uuid4() - foreign key to prompt_arguments in response
        # Other fields need to be pulled from the config

        # Create the prompt_arguments table
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS prompt_arguments (
                id TEXT PRIMARY KEY,
                {', '.join([f'{k} TEXT' for k in self.prompt_arguments_schema.keys()])}
            )
            """
        )
        self.conn.commit()

        # Create the response table
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS response (
                id TEXT PRIMARY KEY,
                {', '.join([f'{k} TEXT' for k in self.response_schema.keys()])},
                prompt_arguments_id TEXT,
                FOREIGN KEY (prompt_arguments_id) REFERENCES prompt_arguments(id)
            )
            """
        )
        self.conn.commit()

    def __insert_prompt_arguments(
        self,
        #   Avoid Variable not allowed in type expression
        prompt_arguments: Type[pydantic.BaseModel],
    ):
        if not isinstance(prompt_arguments, PromptArgumentsCreate):
            raise ValueError(
                "prompt_arguments must be an instance of PromptArgumentsCreate"
            )
        id = str(uuid.uuid4())
        self.cursor.execute(
            f"""
            INSERT INTO prompt_arguments
            VALUES (
                :id,
                {', '.join([f':{k}' for k in self.prompt_arguments_schema.keys()])}
            )
            """,
            {**prompt_arguments.model_dump(), "id": id},
        )
        self.conn.commit()
        return id

    def __insert_response(
        self,
        #   Avoid Variable not allowed in type expression
        response: Type[pydantic.BaseModel],
        prompt_arguments_id: str,
    ):
        if not isinstance(response, ResponseCreate):
            raise ValueError("response must be an instance of ResponseCreate")
        id = str(uuid.uuid4())
        self.cursor.execute(
            f"""
            INSERT INTO response
            VALUES (
                :id,
                {', '.join([f':{k}' for k in self.response_schema.keys()])},
                :prompt_arguments_id
            )
            """,
            {
                **response.model_dump(),
                "id": id,
                "prompt_arguments_id": prompt_arguments_id,
            },
        )
        self.conn.commit()
        return id

    def log(self, with_results: bool = False):
        def decorator(func):
            if asyncio.iscoroutinefunction(func):

                async def wrapper(*args, **kwargs):
                    logger.debug(f"args: {args}")
                    logger.debug(f"kwargs: {kwargs}")
                    original_response = await func(*args, **kwargs)
                    # this is a Response object
                    original_response = original_response.json()
                    logger.debug(f"original_response: {original_response}")

                    # build prompt_arguments from args, skip elements not in prompt_arguments_schema
                    # this will require inspect.getfullargspec(func) to get the function signature
                    # then use that to build the prompt_arguments
                    prompt_arguments = PromptArgumentsCreate(
                        **{
                            k: v
                            for k, v in zip(
                                inspect.getfullargspec(func).args,
                                chain(args, kwargs.values()),
                            )
                            if k in self.prompt_arguments_schema
                        }
                    )

                    # insert prompt_arguments into the database
                    prompt_arguments_id = self.__insert_prompt_arguments(
                        prompt_arguments
                    )

                    # build response from original_response, skip elements not in response_schema
                    response = ResponseCreate(
                        **{
                            k: v
                            for k, v in original_response.items()
                            if k in self.response_schema.keys()
                        },
                        prompt_arguments_id=prompt_arguments_id,
                    )

                    # insert response into the database
                    response_id = self.__insert_response(response, prompt_arguments_id)

                    if with_results:
                        return original_response, {
                            "prompt_arguments": PromptArgumentsRecord(
                                id=prompt_arguments_id, **prompt_arguments.model_dump()
                            ),
                            "response": ResponseRecord(
                                id=response_id,
                                **response.model_dump(),
                            ),
                        }
                    else:
                        return original_response

                return wrapper
            else:

                def wrapper(*args, **kwargs):
                    logger.debug(f"args: {args}")
                    logger.debug(f"kwargs: {kwargs}")
                    original_response = func(*args, **kwargs)
                    original_response = original_response.json()
                    logger.debug(f"original_response: {original_response}")

                    # build prompt_arguments from args, skip elements not in prompt_arguments_schema
                    # this will require inspect.getfullargspec(func) to get the function signature
                    # then use that to build the prompt_arguments
                    prompt_arguments = PromptArgumentsCreate(
                        **{
                            k: v
                            for k, v in zip(
                                inspect.getfullargspec(func).args,
                                chain(args, kwargs.values()),
                            )
                            if k in self.prompt_arguments_schema
                        }
                    )

                    # insert prompt_arguments into the database
                    prompt_arguments_id = self.__insert_prompt_arguments(
                        prompt_arguments
                    )

                    # build response from original_response, skip elements not in response_schema
                    response = ResponseCreate(
                        **{
                            k: v
                            for k, v in original_response.items()
                            if k in self.response_schema.keys()
                        },
                        prompt_arguments_id=prompt_arguments_id,
                    )

                    # insert response into the database
                    response_id = self.__insert_response(response, prompt_arguments_id)

                    if with_results:
                        return original_response, {
                            "prompt_arguments": PromptArgumentsRecord(
                                id=prompt_arguments_id, **prompt_arguments.model_dump()
                            ),
                            "response": ResponseRecord(
                                id=response_id,
                                **response.model_dump(),
                            ),
                        }
                    else:
                        return original_response

                return wrapper
            
        return decorator