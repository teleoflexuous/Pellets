# The library is expected to be able to:
# Log prompt, model, and response to a sqlite database
# One of the interfaces for performing this is a decorator
# The decorator is expected to be used on top of a async request handler
# Response can come in one piece or be streamed

import sqlite3
import json
import pydantic
import asyncio
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
        cache_size: int = 500,
    ):
        self.db_path = db_path
        self.prompt_parameters = prompt_parameters
        self.cache_size = cache_size
        if self.cache_size > 950:
            logger.warning(
                f"cache_size is set to {self.cache_size}. Sqlite has a limit of 999 variables in a single query and Pellets needs a bit of headroom to work with. Consider increasing cache_size to at least 950 to avoid issues."
            )

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.__create_tables()

        self.response_stream_cache: List[ResponseStreamRecord] = []

    def __create_tables(self):
        # Tables will always contain:
        # id: INTEGER PRIMARY KEY
        # prompt_arguments_id: INTEGER

        # Create the prompt_arguments table
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS prompt_arguments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response TEXT NOT NULL,
                prompt_arguments_id INTEGER NOT NULL,
                previous_response_id INTEGER,
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
        logger.debug(f"prompt_arguments: {prompt_arguments.model_dump()}")
        self.cursor.execute(
            f"""
            INSERT INTO prompt_arguments (model, prompt, parameters) VALUES (?, ?, ?)
            """,
            (
                prompt_arguments.model,
                prompt_arguments.prompt,
                json.dumps(prompt_arguments.parameters),
            ),
        )
        self.conn.commit()
        return PromptArgumentsRecord(
            id=self.cursor.lastrowid, **prompt_arguments.model_dump()
        )

    def __insert_response(
        self,
        response: ResponseCreate,
    ) -> ResponseRecord:
        self.cursor.execute(
            f"""
            INSERT INTO response (response, prompt_arguments_id) VALUES (?, ?)
            """,
            (response.response, response.prompt_arguments_id),
        )
        self.conn.commit()
        return ResponseRecord(id=self.cursor.lastrowid, **response.model_dump())

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


    def __add_to_response_stream_transaction(self, response: ResponseStreamRecord) -> int:
        self.cursor.execute(
            """ 
            INSERT INTO response (response, prompt_arguments_id, previous_response_id, done) 
            VALUES (?, ?, ?, ?) 
            """,
            (
                response.response, 
                response.prompt_arguments_id, 
                self.previous_response_id, 
                response.done,
            ),
        )
        self.previous_response_id = self.cursor.lastrowid
        return self.previous_response_id
    
    def __insert_response_stream(self, response: ResponseStreamRecord) -> ResponseStreamRecord:
        self.response_stream_cache.append(response)
        response.previous_response_id = self.previous_response_id
        self.previous_response_id = self.__add_to_response_stream_transaction(response)

        if len(self.response_stream_cache) >= self.cache_size or response.done:
            logger.debug(f"len(self.response_stream_cache): {len(self.response_stream_cache)}, response.done: {response.done}")
            self.conn.commit()
            self.cursor.execute("BEGIN TRANSACTION")
            self.response_stream_cache = []
        
        return ResponseStreamRecord(id=self.previous_response_id, **response.model_dump())

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
                    self.previous_response_id = None
                    for original_response in original_response_stream.iter_lines():
                        original_response_json = original_response.json()
                        new_response = ResponseStreamCreate(
                            prompt_arguments_id=prompt_arguments.id,
                            previous_response_id=self.previous_response_id,
                            response=original_response_json["response"],
                            done=original_response_json["done"],
                        )
                        self.response_stream_cache.append(new_response)
                        if (
                            len(self.response_stream_cache) >= self.cache_size
                            or original_response_json["done"]
                        ):
                            response_stream = self.__insert_response_stream()

                        if with_results:
                            yield original_response, {
                                "prompt_arguments": prompt_arguments,
                                
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
                    original_response = next(original_response_stream.iter_lines())
                    original_response_json = json.loads(original_response)
                    new_response = ResponseStreamCreate(
                        prompt_arguments_id=prompt_arguments.id,
                        previous_response_id=None,
                        response=original_response_json["response"],
                        done=original_response_json["done"],
                    )
                    self.previous_response_id = self.__insert_response(new_response).id
                    self.cursor.execute("BEGIN TRANSACTION")
                    for original_response in original_response_stream.iter_lines():
                        original_response_json = json.loads(original_response)
                        new_response = ResponseStreamCreate(
                            prompt_arguments_id=prompt_arguments.id,
                            previous_response_id=self.previous_response_id,
                            response=original_response_json["response"],
                            done=original_response_json["done"],
                        )
                        if with_results:
                            yield original_response, {
                                "prompt_arguments": prompt_arguments,
                                "response": self.__insert_response_stream(new_response),
                            }
                        else:
                            self.__insert_response_stream(new_response)
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
