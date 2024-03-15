# The library is expected to be able to:
# Log prompt, model, and response to a sqlite database
# One of the interfaces for performing this is a decorator
# The decorator is expected to be used on top of a async request handler
# Response can come in one piece or be streamed

import sqlite3
import json
import orjson
import pydantic
import asyncio
import inspect
from typing import Type, List, Callable, Generator, Union, Tuple, AsyncGenerator, Dict
from itertools import chain
import requests
from queue import Queue
from threading import Thread
import time


from pellets.config import config
from pellets.records import (
    PromptArgumentsRecord,
    ResponseRecord,
    PromptArgumentsCreate,
    ResponseCreate,
    ResponseStreamRecord,
)
from pellets.utils.logger import logger

def read_ahead(stream, queue):
    for line in stream.iter_lines():
        queue.put(line)
    queue.put(None)


class Pellet:
    def __init__(
        self,
        db_path: str = config.db_path,
        prompt_parameters: List[str] = config.prompt_parameters,
        response_stream_cache_size: int = 500,
    ):
        self.db_path = db_path
        self.prompt_parameters = prompt_parameters
        self.response_stream_cache_size = response_stream_cache_size

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.__create_tables()

        self.response_stream_cache: List[str] = []

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

        self.cursor.execute("BEGIN TRANSACTION")
        prompt_arguments = self.__insert_prompt_arguments(prompt_arguments)
        logger.debug(f"prompt_arguments: {prompt_arguments}, args_dict: {args_dict}")
        response = ResponseCreate(
            prompt_arguments_id=prompt_arguments.id,
            previous_response_id=None,
            response=original_response_json["response"],
        )

        response = self.__insert_response(response)
        self.conn.commit()

        return prompt_arguments, response

    def insert_response_stream_cache(
        self,
        prompt_arguments: PromptArgumentsRecord,
        previous_response_id: int = None,
        done: bool = False,
    ) -> ResponseStreamRecord:
        # Integrate str in cache into a single string
        response = "".join(self.response_stream_cache)
        self.response_stream_cache = []

        # Insert the response into the database
        self.cursor.execute(
            f"""
            INSERT INTO response (response, prompt_arguments_id, previous_response_id, done) VALUES (?, ?, ?, ?)
            """,
            (response, prompt_arguments.id, previous_response_id, done),
        )
        return ResponseStreamRecord(
            id=self.cursor.lastrowid,
            prompt_arguments_id=prompt_arguments.id,
            done=done,
            response=response,
            previous_response_id=previous_response_id,
        )

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
            previous_response_id = None
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
                    )
                    self.response_stream_cache = []
                    results = None
                    original_response = next(original_response_stream.iter_lines())
                    original_response_json = original_response.json()
                    self.response_stream_cache.append(original_response["response"])
                    for original_response in original_response_stream.iter_lines():
                        original_response_json = original_response.json()
                        if (
                            len(self.response_stream_cache)
                            >= self.response_stream_cache_size
                            or original_response_json["done"]
                        ):
                            response_stream = self.insert_response_stream_cache(
                                prompt_arguments=prompt_arguments,
                                previous_response_id=previous_response_id,
                                done=original_response_json["done"],
                            )
                            results = {
                                "prompt_arguments": prompt_arguments,
                                "response": response_stream,
                            }
                        if with_results:
                            yield original_response, results
                            results = None
                        else:
                            yield original_response

                return wrapper
            else:
                # logger.debug(f"func: {func}, with_results: {with_results}")
                def wrapper(*args, **kwargs) -> Generator:
                    # logger.debug(f"args: {args}, kwargs: {kwargs}")
                    original_response_stream = func(*args, **kwargs)
                    args_dict = dict(zip(param_names, args))
                    args_dict.update(kwargs)
                    self.cursor.execute("BEGIN TRANSACTION")
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
                    # logger.debug(
                    #     f"log response stream: {original_response_stream}, {type(original_response_stream)}"
                    # )
                    time_create_prompt_arguments = time.time()
                    self.response_stream_cache = []
                    original_response = next(original_response_stream.iter_lines())
                    # original_response_json = json.loads(original_response.decode("utf-8"))
                    # make it orjson
                    original_response_json = orjson.loads(original_response)
                    self.response_stream_cache.append(original_response_json["response"])
                    previous_response_id = None

                    queue = Queue(maxsize=100)
                    thread = Thread(target=read_ahead, args=(original_response_stream, queue))
                    thread.start()
                    
                    def process_responses(args, prompt_arguments, original_response, previous_response_id):
                        results = None                           
                        # original_response_json = json.loads(original_response.decode("utf-8"))
                        original_response_json = orjson.loads(original_response)
                        self.response_stream_cache.append(original_response_json["response"])
                        if (
                                len(self.response_stream_cache)
                                >= self.response_stream_cache_size
                                or original_response_json["done"]
                            ):
                            response_stream = self.insert_response_stream_cache(
                                    prompt_arguments=prompt_arguments,
                                    previous_response_id=previous_response_id,
                                    done=original_response_json["done"],
                                )
                            results = {
                                    "prompt_arguments": prompt_arguments,
                                    "response": response_stream,
                                }
                            logger.debug(f"results: {results}")
                            self.conn.commit()
                            previous_response_id = response_stream.id
                        return results

                    if with_results:
                        for original_response in iter(queue.get, None):
                            if original_response is None:
                                break
                            results = process_responses(args, prompt_arguments, original_response, previous_response_id)
                            yield original_response, results
                    else:
                        for original_response in iter(queue.get, None):
                            if original_response is None:
                                break
                            process_responses(args, prompt_arguments, original_response, previous_response_id)
                            yield original_response
                    
                return wrapper

        return decorator

    def log(self, with_results: bool = False, stream: bool = False):
        if stream:
            return self.__log_stream(with_results=with_results)
        else:
            return self.__log_non_stream(with_results=with_results)

    def __read_prompt_arguments(
        self,
        ids: List[int] = None,
        models: List[str] = None,
        prompts: List[str] = None,
        params_dicts: List[Dict[str, str]] = None,
    ) -> List[PromptArgumentsRecord]:
        query = f"""
        SELECT * FROM prompt_arguments
        """

        if ids:
            query += f"WHERE id IN ({', '.join([str(i) for i in ids])})"
        if models:
            query += f"WHERE model IN ({', '.join([str(m) for m in models])})"
        if prompts:
            query += f"WHERE prompt IN ({', '.join([str(p) for p in prompts])})"
        if params_dicts:
            params_dicts = [json.dumps(p) for p in sorted(params_dicts)]
            query += (
                f"WHERE parameters IN ({', '.join([str(p) for p in params_dicts])})"
            )

        self.cursor.execute(query)
        return [PromptArgumentsRecord(*row) for row in self.cursor.fetchall()]

    def __read_response(
        self,
        ids: List[int] = None,
        prompt_arguments_ids: List[int] = None,
        previous_response_ids: List[int] = None,
        done: bool = None,
    ) -> List[ResponseRecord]:
        query = f"""
        SELECT * FROM response
        """

        if ids:
            query += f"WHERE id IN ({', '.join([str(i) for i in ids])})"
        if prompt_arguments_ids:
            query += f"WHERE prompt_arguments_id IN ({', '.join([str(p) for p in prompt_arguments_ids])})"
        if previous_response_ids:
            query += f"WHERE previous_response_id IN ({', '.join([str(p) for p in previous_response_ids])})"
        if done is not None:
            query += f"WHERE done = {done}"

        self.cursor.execute(query)
        return [ResponseRecord(*row) for row in self.cursor.fetchall()]

    # def __read_response_stream(self, ids: List[int] = None, prompt_arguments_ids: List[int] = None, previous_response_ids: List[int] = None, done: bool = None) -> List[ResponseStreamRecord]:
