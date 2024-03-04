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
from typing import Type, List, Callable, Generator, Union, Tuple, AsyncGenerator, Dict
from itertools import chain
import requests
import time

from pellets.config import config
from pellets.records import (
    PromptArgumentsRecord,
    ResponseRecord,
    PromptArgumentsCreate,
    ResponseCreate,
    ResponseStreamCreate,
    ResponseStreamRecord,
)
from pellets.utils.logger import logger


class Pellet:
    def __init__(
        self,
        db_path: str = config.db_path,
        prompt_parameters: List[str] = config.prompt_parameters,
        cache_size: int = 1000,
    ):
        self.db_path = db_path
        self.prompt_parameters = prompt_parameters
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.__create_tables()
        self.time = time
        self.cache_size = cache_size
        self.response_stream_cache: List[ResponseStreamRecord] = []

    def __create_tables(self):
        # Tables will always contain:
        # id: uuid.uuid4()
        # prompt_arguments_id: uuid.uuid4() - foreign key to prompt_arguments in response

        # Create the prompt_arguments table
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS prompt_arguments (
                id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                prompt TEXT NOT NULL,
                parameters TEXT
            )
            """
        )
        self.conn.commit()

        # Create the response table
        # prompt_arguments_id is a foreign key to prompt_arguments
        # response_id is a foreign key to response - UUID that defaults to None, but it needs to exist in the table
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS response (
                id TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                prompt_arguments_id TEXT NOT NULL,
                previous_response_id TEXT,
                done BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (prompt_arguments_id) REFERENCES prompt_arguments(id),
                FOREIGN KEY (previous_response_id) REFERENCES response(id)
            )
            """
        )
        self.conn.commit()

    def __insert_prompt_arguments(
        self,
        prompt_arguments: PromptArgumentsCreate,
    ) -> PromptArgumentsRecord:
        id = str(uuid.uuid4())
        logger.debug(f"prompt_arguments: {prompt_arguments.model_dump()}")
        self.cursor.execute(
            f"""
            INSERT INTO prompt_arguments (id, model, prompt,
            parameters) VALUES (?, ?, ?, ?)
            """,
            (
                id,
                prompt_arguments.model,
                prompt_arguments.prompt,
                json.dumps(prompt_arguments.parameters),
            ),
        )
        self.conn.commit()
        return PromptArgumentsRecord(id=id, **prompt_arguments.model_dump())

    def __insert_response(
        self,
        response: ResponseCreate,
    ) -> ResponseRecord:
        id = str(uuid.uuid4())
        self.cursor.execute(
            f"""
            INSERT INTO response (id, response, prompt_arguments_id) VALUES (?, ?, ?)
            """,
            (id, response.response, str(response.prompt_arguments_id)),
        )
        self.conn.commit()
        return ResponseRecord(id=id, **response.model_dump())

    def __log_prompt_arguments_and_response_non_streamed(
        self,
        args_dict: dict,
        original_response_json: dict,
    ) -> Tuple[PromptArgumentsRecord, ResponseRecord]:

        # parameters are all the arguments that are not model or prompt, that appear in self.prompt_parameters, put into a dictionary
        prompt_arguments = PromptArgumentsCreate(
            model=args_dict["model"],
            prompt=args_dict["prompt"],
            parameters={
                k: v
                for k, v in sorted(args_dict.items())
                if k not in ["model", "prompt"] and k in self.prompt_parameters
            },
        )

        prompt_arguments = self.__insert_prompt_arguments(prompt_arguments)
        logger.debug(f"prompt_arguments: {prompt_arguments}, args_dict: {args_dict}")
        response = ResponseCreate(
            prompt_arguments_id=prompt_arguments.id,
            previous_response_id=None,
            response=original_response_json["response"],
        )

        response = self.__insert_response(response)

        return prompt_arguments, response

    def __insert_response_stream(
        self,
    ) -> List[ResponseStreamRecord]:
        self.cursor.executemany(
            f"""
            INSERT INTO response (id, response, prompt_arguments_id,
            previous_response_id, done) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    str(response.id),
                    response.response,
                    str(response.prompt_arguments_id),
                    str(response.previous_response_id),
                    response.done,
                )
                for response in self.response_stream_cache
            ],
        )
        self.conn.commit()
        self.response_stream_cache = []

        return [
            ResponseStreamRecord(id=response.id, **response.model_dump())
            for response in self.response_stream_cache
        ]

    def __log_non_stream(self, with_results: bool = False) -> Callable:
        def decorator(func: Callable) -> Callable:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            logger.debug(f"param_names: {param_names}, func: {func}")

            async def async_wrapper(*args, **kwargs):
                original_response = await func(*args, **kwargs)
                original_response_json = await original_response.json()
                return original_response, original_response_json

            def sync_wrapper(*args, **kwargs):
                original_response = func(*args, **kwargs)
                logger.debug(
                    f"original_response: {original_response}, {type(original_response)}, args: {args}, kwargs: {kwargs}"
                )
                original_response_json = original_response.json()
                return original_response, original_response_json

            wrapper = (
                async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
            )

            def final_wrapper(*args, **kwargs) -> Union[
                requests.Response,
                Tuple[
                    requests.Response,
                    Dict[str, Union[PromptArgumentsRecord, ResponseRecord]],
                ],
            ]:
                original_response, original_response_json = wrapper(*args, **kwargs)
                args_dict = dict(zip(param_names, args))
                args_dict.update(kwargs)

                if with_results:
                    prompt_arguments, response = (
                        self.__log_prompt_arguments_and_response_non_streamed(
                            args_dict, original_response_json
                        )
                    )
                    return original_response, {
                        "prompt_arguments": prompt_arguments,
                        "response": response,
                    }
                else:
                    self.__log_prompt_arguments_and_response_non_streamed(
                        args_dict, original_response_json
                    )
                    return original_response

            return final_wrapper

        return decorator

    def __log_stream(self, with_results: bool = False) -> Callable:
        def decorator(func) -> Callable:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            if asyncio.iscoroutinefunction(func):

                async def wrapper(*args, **kwargs) -> AsyncGenerator:
                    args_dict = dict(zip(param_names, args))
                    args_dict.update(kwargs)
                    original_response_stream = func(*args, **kwargs)
                    prompt_arguments = self.__insert_prompt_arguments(
                        PromptArgumentsCreate(
                            model=args_dict["model"],
                            prompt=args_dict["prompt"],
                            parameters={
                                k: v
                                for k, v in sorted(args_dict.items())
                                if k not in ["model", "prompt"]
                                and k in self.prompt_parameters
                            },
                        )
                    ).id
                    self.response_stream_cache = []
                    previous_response_id = prompt_arguments.id
                    for original_response in original_response_stream.iter_lines():
                        original_response_json = original_response.json()
                        new_response_id = uuid.uuid4()
                        new_response = ResponseStreamCreate(
                            id=new_response_id,
                            prompt_arguments_id=prompt_arguments.id,
                            previous_response_id=previous_response_id,
                            response=original_response_json["response"],
                            done=original_response_json["done"] == "true",
                        )
                        previous_response_id = str(new_response_id)
                        self.response_stream_cache.append(new_response)
                        if (
                            len(self.response_stream_cache) >= self.cache_size
                            or original_response_json["done"] == "true"
                        ):
                            self.__insert_response_stream()

                        if with_results:
                            yield original_response, {
                                "prompt_arguments": prompt_arguments,
                                "responses": ResponseStreamRecord(
                                    id=new_response_id, **new_response.model_dump()
                                ),
                            }
                        else:
                            yield original_response

                return wrapper
            else:
                logger.debug(f"func: {func}, with_results: {with_results}")
                def wrapper(*args, **kwargs) -> Generator:
                    logger.debug(f"args: {args}, kwargs: {kwargs}")
                    original_response_stream = func(*args, **kwargs)
                    args_dict = dict(zip(param_names, args))
                    args_dict.update(kwargs)
                    prompt_arguments = self.__insert_prompt_arguments(
                        PromptArgumentsCreate(
                            model=args_dict["model"],
                            prompt=args_dict["prompt"],
                            parameters={
                                k: v
                                for k, v in sorted(args_dict.items())
                                if k not in ["model", "prompt"]
                                and k in self.prompt_parameters
                            },
                        )
                    )
                    logger.debug(
                        f"log response stream: {original_response_stream}, {type(original_response_stream)}"
                    )
                    self.response_stream_cache = []
                    previous_response_id = prompt_arguments.id
                    for original_response in original_response_stream.iter_lines():
                        original_response_json = json.loads(original_response)
                        new_response_id = str(uuid.uuid4())
                        new_response = ResponseStreamCreate(
                            id=new_response_id,
                            prompt_arguments_id=prompt_arguments.id,
                            previous_response_id=previous_response_id,
                            response=original_response_json["response"],
                            done=original_response_json["done"] == "true",
                        )
                        previous_response_id = new_response_id
                        self.response_stream_cache.append(new_response)
                        if (
                            len(self.response_stream_cache) >= self.cache_size
                            or original_response_json["done"] == "true"
                        ):
                            self.__insert_response_stream()

                        if with_results:
                            yield original_response, {
                                "prompt_arguments": prompt_arguments,
                                "response": ResponseStreamRecord(
                                    **new_response.model_dump()
                                ),
                            }
                        else:
                            yield original_response

                return wrapper

        return decorator

    def log(self, with_results: bool = False, stream: bool = False):
        if stream:
            return self.__log_stream(with_results=with_results)
        else:
            return self.__log_non_stream(with_results=with_results)

    def close(self):
        self.conn.close()
        logger.debug("Connection to sqlite database closed")
