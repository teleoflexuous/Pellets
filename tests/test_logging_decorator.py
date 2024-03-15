# The library is expected to be able to:
# Log prompt, model, and response to a sqlite database
# One of the interfaces for performing this is a decorator
# The decorator is expected to be used on top of a async request handler
# Response can come in one piece or be streamed

# Streamed response looks like this:
# [
#   {
#     "model": "llama2",
#     "created_at": "2024-02-27T15:35:43.9583603Z",
#     "response": "\n",
#     "done": false
#   },
#   {
#     "model": "llama2",
#     "created_at": "2024-02-27T15:35:44.0325118Z",
#     "response": "The",
#     "done": false
#   },
#   (...),
#   {
#     "model": "llama2",
#     "created_at": "2024-02-27T15:36:08.5407464Z",
#     "response": "",
# #     "done": true,
#     "context": [
#         518,
#         25580,
#         (...),
#    ]
#        "total_duration": 24883114000,
#        "load_duration": 1000400,
#        "prompt_eval_duration": 299610000,
#        "eval_count": 279,
#        "eval_duration": 24582669000
#    }
# ]
# Non-streamed response looks like this:
# {
#   "model": "llama2",
#   "created_at": "2023-08-04T19:22:45.499127Z",
#   "response": "The sky is blue because it is the color of the sky.",
#   "done": true,
#   "context": [1, 2, 3],
#   "total_duration": 5043500667,
#   "load_duration": 5025959,
#   "prompt_eval_count": 26,
#   "prompt_eval_duration": 325953000,
#   "eval_count": 290,
#   "eval_duration": 4709213000
# }

# For testing, we will use a locally run LLM model
# Example request for local testing:
# curl http://localhost:11434/api/generate -d '{
#   "model": "llama2",
#   "prompt":"Why is the sky blue?"
# }'

# Tests architecture, from the most basic to the most complex:
# (We are testing the logging, not the model or the API)
# 1. Test that the decorator correctly forwards the original response
# 2. Test that the decorator correctly reads the prompt, model, and response
# 3. Test that the decorator correctly logs the prompt, model, and response to a sqlite database
# 4. Test that the decorator correctly reads the prompt, model, and response from the sqlite database
# 5. Test all together on a live server stream/non-stream and sync/async
# 6. Check overhead of the decorator

import pytest  # for the fixture, among other things
import requests  # for the HTTP requests
import json  # for parsing the response
import sqlite3  # for the database connection
import datetime  # for the timestamp
import uuid  # for the UUID
import os  # for the file handling
import asyncio  # for the async tests
from unittest.mock import AsyncMock, MagicMock  # for the mocking
import aiohttp  # for the async HTTP requests
import pickle  # for the serialization
import timeit  # for the performance testing
import cProfile  # for the performance testing
import pstats  # for the performance testing
import io  # for the performance testing

from pellets.pellet import Pellet
from pellets.records import (
    PromptArgumentsRecord,
    ResponseRecord,
    PromptArgumentsCreate,
    ResponseCreate,
    ResponseStreamRecord,
)
from pellets.utils.logger import logger
from pellets.config import config


MODEL = "llama2"
PROMPT = "Why is the sky blue?"
PARAMS = {"format": "json"}
LOCAL_PICKLE_PATH = "tests/pickles/"

file_names = {
    (
        MODEL,
        PROMPT,
        str(sorted(PARAMS.items())),
        False,
        False,
    ): "nonstreamed_sync_sky_blue_llama2_format_json.pkl",
    # (
    #     MODEL,
    #     PROMPT,
    #     str(sorted(PARAMS.items())),
    #     False,
    #     True,
    # ): "nonstreamed_async_sky_blue_llama2_format_json.pkl",
    (
        MODEL,
        PROMPT,
        str(sorted(PARAMS.items())),
        True,
        False,
    ): "streamed_sync_sky_blue_llama2_format_json.pkl",
    # (
    #     MODEL,
    #     PROMPT,
    #     str(sorted(PARAMS.items())),
    #     True,
    #     True,
    # ): "streamed_async_sky_blue_llama2_format_json.pkl",
}


# Set up the HTTP requests for local pickle management
def generate(stream, model, prompt, params):
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": stream, **params}
    response = requests.post(url, json=data)
    assert response.status_code == 200
    return response


async def async_generate(model, prompt, stream, params):
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": stream, **params}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            assert response.status == 200
            return response


# We need to have local pickles of the responses for the tests
# One for each of stream/non-stream and sync/async
# Check if they exist, if not, create them


@pytest.fixture(scope="session", autouse=True)
def test_pickle_exists():
    if not os.path.exists(LOCAL_PICKLE_PATH):
        os.makedirs(LOCAL_PICKLE_PATH)
    for model, prompt, str_params, stream, async_ in file_names.keys():
        params = dict(eval(str_params))
        logger.debug(
            f"model: {model}, prompt: {prompt}, params: {params}, stream: {stream}, async: {async_}"
        )
        if not os.path.exists(
            f"{LOCAL_PICKLE_PATH}{file_names[(model, prompt, str_params, stream, async_)]}"
        ):
            if async_:
                response = asyncio.run(async_generate(model, prompt, stream, params))
            else:
                response = generate(stream, model, prompt, params)
            with open(
                f"{LOCAL_PICKLE_PATH}{file_names[(model, prompt, str_params, stream, async_)]}",
                "wb",
            ) as f:
                pickle.dump(response, f)


@pytest.fixture
def db_path():
    return config.db_path


@pytest.fixture
def pellet():
    return Pellet(prompt_parameters=PARAMS)


@pytest.fixture
def model():
    return MODEL


@pytest.fixture
def prompt():
    return PROMPT


@pytest.fixture
def params():
    return PARAMS


@pytest.fixture
def live_generate_nonstreamed_sync():
    return generate(False, MODEL, PROMPT, PARAMS)


@pytest.fixture
def live_generate_streamed_sync():
    return generate(True, MODEL, PROMPT, PARAMS)


# Set up the mock response


@pytest.fixture
def mock_response_list():
    return [
        {"response": "\n", "done": False},
        {"response": "The", "done": False},
        {
            "response": "The sky is blue because it is the color of the sky.",
            "done": True,
        },
    ]


@pytest.fixture
def mock_generate():
    filename = file_names[(MODEL, PROMPT, str(sorted(PARAMS.items())), False, False)]
    logger.debug(f"filename: {filename}")
    local_response = pickle.load(open(f"{LOCAL_PICKLE_PATH}{filename}", "rb"))
    return local_response


@pytest.fixture
def async_mock_generate():
    async def _mock_response():
        filename = file_names[(MODEL, PROMPT, str(sorted(PARAMS.items())), False, True)]
        local_response = pickle.load(open(f"{LOCAL_PICKLE_PATH}{filename}", "rb"))
        return local_response

    return _mock_response


@pytest.fixture
def mock_generate_stream():
    filename = file_names[(MODEL, PROMPT, str(sorted(PARAMS.items())), True, False)]
    local_response = pickle.load(open(f"{LOCAL_PICKLE_PATH}{filename}", "rb"))
    return local_response


@pytest.fixture
def async_mock_generate_stream():
    async def _mock_response():
        filename = file_names[(MODEL, PROMPT, str(sorted(PARAMS.items())), True, True)]
        local_response = pickle.load(open(filename, "rb"))
        return local_response

    return _mock_response


# Test that the decorator correctly forwards the original response for a non-streamed response
def test_log_forward_original_response_nonstreamed_sync(
    pellet, mock_generate, model, prompt, params
):
    @pellet.log(with_results=False, stream=False)
    def func(model, prompt, **params):
        return mock_generate

    original_response = func(model, prompt, **params)

    # Assert the response is not empty
    assert original_response.json()["response"] != ""
    # Assert the status code
    assert original_response.status_code == mock_generate.status_code
    # Assert the headers
    assert original_response.headers == mock_generate.headers
    # Assert the JSON response
    assert original_response.json() == mock_generate.json()


@pytest.mark.asyncio
async def test_log_forward_original_response_nonstreamed_async(
    pellet, async_mock_generate, model, prompt, params
):
    return  # TODO: Async

    @pellet.log(with_results=False, stream=False)
    async def func(model, prompt, **params):
        return async_mock_generate

    original_response = await func(model, prompt, **params)

    # Assert the status code
    assert original_response.status_code == mock_response.status_code
    # Assert the headers
    assert original_response.headers == mock_response.headers
    # Assert the JSON response
    assert original_response.json() == mock_response.json()


# Test that the decorator correctly forwards the original response for a streamed response
def test_log_forward_original_response_streamed_sync(
    pellet, mock_generate_stream, model, prompt, params
):
    @pellet.log(with_results=False, stream=True)
    def func(model, prompt, **params):
        return mock_generate_stream

    original_response = func(model, prompt, **params)

    for response, expected_response in zip(
        original_response, mock_generate_stream.iter_lines()
    ):
        pass
        response = json.loads(response)
        original_response = json.loads(expected_response)
        assert response == original_response


@pytest.mark.asyncio
async def test_log_forward_original_response_streamed_async(
    pellet, async_mock_generate_stream, model, prompt, params
):
    return  # TODO: Async

    @pellet.log(with_results=False, stream=True)
    async def func(model, prompt, **params):
        return await async_mock_generate_stream

    original_response = await func(model, prompt, **params)

    assert original_response.status_code == mock_response_stream.status_code
    assert original_response.headers == mock_response_stream.headers
    for response, expected_response in zip(
        original_response, mock_response_stream.json.side_effect
    ):
        assert response.json() == expected_response


# Test that the decorator correctly reads the prompt, model, and response for a non-streamed response


def test_log_read_arguments_nonstreamed_sync(
    pellet, mock_generate, model, prompt, params
):
    @pellet.log(with_results=True, stream=False)
    def func(model, prompt, **params):
        return mock_generate

    original_response, results = func(model, prompt, **params)

    assert isinstance(results["prompt_arguments"], PromptArgumentsRecord)
    assert results["prompt_arguments"] == PromptArgumentsRecord(
        id=results["prompt_arguments"].id,
        **PromptArgumentsCreate(
            model=model, prompt=prompt, parameters=params
        ).model_dump(),
    )
    assert isinstance(results["response"], ResponseRecord)
    assert results["response"] == ResponseRecord(
        id=results["response"].id,
        **ResponseCreate(
            response=mock_generate.json()["response"],
            prompt_arguments_id=results["prompt_arguments"].id,
        ).model_dump(),
    )


@pytest.mark.asyncio
async def test_log_read_arguments_nonstreamed_async(
    pellet, model, prompt, params, async_mock_generate
):
    return  # TODO: Async

    @pellet.log(with_results=True, stream=False)
    async def func(model, prompt, **params):
        return await async_mock_generate

    original_response, results = await func(model, prompt, **params)

    assert isinstance(results["prompt_arguments"], PromptArgumentsRecord)
    assert results["prompt_arguments"] == PromptArgumentsRecord(
        id=results["prompt_arguments"].id,
        **PromptArgumentsCreate(
            model=model, prompt=prompt, parameters=params
        ).model_dump(),
    )
    assert isinstance(results["response"], ResponseRecord)
    assert results["response"] == ResponseRecord(
        id=results["response"].id,
        **ResponseCreate(
            response=response["response"],
            prompt_arguments_id=results["prompt_arguments"].id,
        ).model_dump(),
    )


# Test that the decorator correctly reads the prompt, model, and response for a streamed response


def test_log_read_arguments_streamed_sync(
    pellet, model, prompt, params, mock_generate_stream
):
    @pellet.log(with_results=True, stream=True)
    def func(model, prompt, **params):
        return mock_generate_stream

    original_responses = []
    results = {"prompt_arguments": None, "responses": []}
    for original_response, result in func(model, prompt, **params):
        if result:
            if results["prompt_arguments"]:
                assert result["prompt_arguments"] == results["prompt_arguments"]
            results["prompt_arguments"] = result["prompt_arguments"]
            results["responses"].append(result["response"])

    prompt_arguments = results["prompt_arguments"]
    responses = results["responses"]
    logger.debug(f"len(responses): {len(responses)}")

    assert isinstance(prompt_arguments, PromptArgumentsRecord)
    assert prompt_arguments == PromptArgumentsRecord(
        id=prompt_arguments.id,
        **PromptArgumentsCreate(
            model=model, prompt=prompt, parameters=params
        ).model_dump(),
    )

    for response, expected_response in zip(
        responses, mock_generate_stream.iter_lines()
    ):
        expected_response = json.loads(expected_response)
        expected_response["response"] = expected_response["response"].strip()
        original_responses.append(response.response)  # Use dot notation
        if (
            len(original_responses) >= pellet.response_stream_cache_size
            or expected_response["done"]
        ):
            assert response == ResponseStreamRecord(
                id=response.id,
                **ResponseStreamRecord(
                    prompt_arguments_id=prompt_arguments.id,
                    done=expected_response["done"],
                    response="".join(original_responses),
                ).model_dump(),
            )
            original_responses = []


@pytest.mark.asyncio
async def test_log_read_arguments_streamed_async(
    pellet, model, prompt, params, async_mock_generate_stream
):
    return  # TODO: Async

    @pellet.log(with_results=True, stream=True)
    async def func(model, prompt, **params):
        return await async_mock_generate_stream

    original_responses = []
    results = {"prompt_arguments": None, "responses": []}
    async for original_response, result in func(model, prompt, **params):
        if results:
            if results["prompt_arguments"]:
                assert result["prompt_arguments"] == results["prompt_arguments"]
            results["prompt_arguments"] = result["prompt_arguments"]
            results["responses"].append(result["response"])

    prompt_arguments = results["prompt_arguments"]
    responses = results["responses"]

    assert isinstance(prompt_arguments, PromptArgumentsRecord)
    assert prompt_arguments == PromptArgumentsRecord(
        id=prompt_arguments.id,
        **PromptArgumentsCreate(
            model=model, prompt=prompt, parameters=params
        ).model_dump(),
    )

    for response, expected_response in zip(
        responses, mock_response_stream.iter_lines()
    ):
        expected_response = json.loads(expected_response)
        original_responses.append(response.response)  # Use dot notation
        if (
            len(original_responses) >= pellet.response_stream_cache_size
            or expected_response["done"]
        ):
            assert response == ResponseStreamRecord(
                id=response.id,
                **ResponseStreamRecord(
                    prompt_arguments_id=prompt_arguments.id,
                    done=expected_response["done"],
                    response="".join(original_responses),
                ).model_dump(),
            )
            original_responses = []


# Test that the decorator correctly logs the prompt, model, and response for a non-streamed response to a sqlite database


def test_log_to_sqlite_nonstreamed_sync(
    pellet, model, prompt, params, mock_generate, db_path
):
    @pellet.log(with_results=True, stream=False)
    def func(model, prompt, **params):
        return mock_generate

    original_response, results = func(model, prompt, **params)

    created_prompt_arguments = results["prompt_arguments"]
    created_response = results["response"]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        # There was a change. Everything is indexed by id now
        f"""
        SELECT * FROM prompt_arguments WHERE id = '{created_prompt_arguments.id}'
        """
    )
    fetched_prompt_arguments_table = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    fetched_prompt_arguments = {}
    for column, value in zip(columns, fetched_prompt_arguments_table):
        fetched_prompt_arguments[column] = value
    fetched_prompt_arguments["parameters"] = json.loads(
        fetched_prompt_arguments["parameters"]
    )
    logger.debug(f"fetched_prompt_arguments: {fetched_prompt_arguments}")
    fetched_prompt_arguments = PromptArgumentsRecord.model_validate(
        fetched_prompt_arguments
    )
    assert fetched_prompt_arguments == created_prompt_arguments

    cursor.execute(
        f"""
        SELECT * FROM response WHERE id = '{created_response.id}'
        """
    )
    fetched_response_table = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    fetched_response = {}
    for column, value in zip(columns, fetched_response_table):
        fetched_response[column] = value
    fetched_response = ResponseRecord.model_validate(fetched_response)
    assert fetched_response == created_response

    conn.close()


@pytest.mark.asyncio
async def test_log_to_sqlite_nonstreamed_async(
    pellet, model, prompt, params, async_mock_generate, db_path
):
    return

    @pellet.log(with_results=True, stream=False)
    async def func(model, prompt, **params):
        return await async_mock_generate

    original_response, results = await func(model, prompt, **params)

    created_prompt_arguments = results["prompt_arguments"]
    created_response = results["response"]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT * FROM prompt_arguments WHERE id = '{created_prompt_arguments.id}'
        """
    )
    fetched_prompt_arguments_table = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    fetched_prompt_arguments = {}
    for column, value in zip(columns, fetched_prompt_arguments_table):
        fetched_prompt_arguments[column] = value
    fetched_prompt_arguments["parameters"] = json.loads(
        fetched_prompt_arguments["parameters"]
    )
    fetched_prompt_arguments = PromptArgumentsRecord.model_validate(
        fetched_prompt_arguments
    )
    assert fetched_prompt_arguments == created_prompt_arguments

    cursor.execute(
        f"""
        SELECT * FROM response WHERE id = '{created_response.id}'
        """
    )
    fetched_response_table = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    fetched_response = {}
    for column, value in zip(columns, fetched_response_table):
        fetched_response[column] = value
    fetched_response = ResponseRecord.model_validate(fetched_response)
    assert fetched_response == created_response

    conn.close()


# Test that the decorator correctly logs the prompt, model, and response for a streamed response to a sqlite database


def test_log_to_sqlite_streamed_sync(
    pellet, model, prompt, params, mock_generate_stream, db_path
):
    @pellet.log(with_results=True, stream=True)
    def func(model, prompt, **params):
        return mock_generate_stream

    results = {"prompt_arguments": None, "responses": []}
    for original_response, result in func(model, prompt, **params):
        if result:
            if results["prompt_arguments"]:
                assert result["prompt_arguments"] == results["prompt_arguments"]
            results["prompt_arguments"] = result["prompt_arguments"]
            results["responses"].append(result["response"])

    prompt_arguments = results["prompt_arguments"]
    logger.debug(f"prompt_arguments: {prompt_arguments}")
    responses = results["responses"]
    logger.debug(f"len(responses): {len(responses)}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT * FROM prompt_arguments WHERE id = '{prompt_arguments.id}'
        """
    )
    fetched_prompt_arguments_table = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    logger.debug(f"columns: {columns}")
    fetched_prompt_arguments = {}
    for column, value in zip(columns, fetched_prompt_arguments_table):
        fetched_prompt_arguments[column] = value
    fetched_prompt_arguments["parameters"] = json.loads(
        fetched_prompt_arguments["parameters"]
    )
    fetched_prompt_arguments = PromptArgumentsRecord.model_validate(
        fetched_prompt_arguments
    )
    assert fetched_prompt_arguments == prompt_arguments
    response_ids = [response.id for response in responses]

    # Prepare the placeholders for the IN clause
    placeholders = ", ".join("?" for _ in response_ids)

    # Prepare the query
    query = f"SELECT * FROM response WHERE id IN ({placeholders})"

    # Execute the query
    cursor.execute(query, response_ids)
    fetched_responses_table = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    fetched_responses = []
    for fetched_response_record in fetched_responses_table:
        fetched_response = {}
        for column, value in zip(columns, fetched_response_record):
            fetched_response[column] = value
        fetched_response = ResponseStreamRecord.model_validate(fetched_response)
        fetched_responses.append(fetched_response)
    fetched_responses = [
        ResponseStreamRecord.model_validate(fetched_response)
        for fetched_response in fetched_responses
    ]
    for fetched_response, response in zip(fetched_responses, responses):
        assert fetched_response.response == response.response
        assert fetched_response.done == response.done
        assert fetched_response.prompt_arguments_id == response.prompt_arguments_id
        assert fetched_response.previous_response_id == response.previous_response_id
        assert fetched_response == response

    conn.close()


@pytest.mark.asyncio
async def test_log_to_sqlite_streamed_async(
    pellet, model, prompt, params, async_mock_generate_stream, db_path
):
    return

    @pellet.log(with_results=True, stream=True)
    async def func(model, prompt, **params):
        return await async_mock_generate_stream

    results = {"prompt_arguments": None, "responses": []}
    async for original_response, result in func(model, prompt, **params):
        if results["prompt_arguments"]:
            assert result["prompt_arguments"] == results["prompt_arguments"]
        results["prompt_arguments"] = result["prompt_arguments"]
        results["responses"].append(result["response"])

    prompt_arguments = results["prompt_arguments"]
    responses = results["responses"]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT * FROM prompt_arguments WHERE id = '{prompt_arguments.id}'
        """
    )
    fetched_prompt_arguments_table = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    fetched_prompt_arguments = {}
    for column, value in zip(columns, fetched_prompt_arguments_table):
        fetched_prompt_arguments[column] = value
    fetched_prompt_arguments["parameters"] = json.loads(
        fetched_prompt_arguments["parameters"]
    )
    fetched_prompt_arguments = PromptArgumentsRecord.model_validate(
        fetched_prompt_arguments
    )
    assert fetched_prompt_arguments == prompt_arguments
    response_ids = [response.id for response in responses]

    # Prepare the placeholders for the IN clause
    placeholders = ", ".join("?" for _ in response_ids)

    # Prepare the query
    query = f"SELECT * FROM response WHERE id IN ({placeholders})"

    cursor.execute(query, response_ids)
    fetched_responses_table = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    fetched_responses = []
    for fetched_response_record in fetched_responses_table:
        fetched_response = {}
        for column, value in zip(columns, fetched_response_record):
            fetched_response[column] = value
        fetched_response = ResponseStreamRecord.model_validate(fetched_response)
        fetched_responses.append(fetched_response)
    fetched_responses = [
        ResponseStreamRecord.model_validate(fetched_response)
        for fetched_response in fetched_responses
    ]
    for fetched_response, response in zip(fetched_responses, responses):
        assert fetched_response.response == response.response
        assert fetched_response.done == response.done
        assert fetched_response.prompt_arguments_id == response.prompt_arguments_id
        assert fetched_response.previous_response_id == response.previous_response_id
        assert fetched_response == response

    conn.close()


# Test all together on a live server stream/non-stream and sync/async
def test_log_live_sqlite_nonstreamed_sync(pellet, model, prompt, params, db_path):
    @pellet.log(with_results=True, stream=False)
    def func(model, prompt, **params):
        return generate(stream=False, model=model, prompt=prompt, params=params)

    original_response, results = func(model, prompt, **params)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        f"""
        SELECT * FROM prompt_arguments WHERE id = '{results["prompt_arguments"].id}'
        """
    )
    fetched_prompt_arguments_table = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    fetched_prompt_arguments = {}
    for column, value in zip(columns, fetched_prompt_arguments_table):
        fetched_prompt_arguments[column] = value
    fetched_prompt_arguments["parameters"] = json.loads(
        fetched_prompt_arguments["parameters"]
    )
    fetched_prompt_arguments = PromptArgumentsRecord.model_validate(
        fetched_prompt_arguments
    )
    assert fetched_prompt_arguments == results["prompt_arguments"]

    cursor.execute(
        f"""
        SELECT * FROM response WHERE id = '{results["response"].id}'
        """
    )
    fetched_response_table = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    fetched_response = {}
    for column, value in zip(columns, fetched_response_table):
        fetched_response[column] = value
    fetched_response = ResponseRecord.model_validate(fetched_response)
    assert fetched_response == results["response"]

    conn.close()


@pytest.mark.asyncio
async def test_log_live_sqlite_nonstreamed_async(
    pellet, model, prompt, params, db_path
):
    return

    @pellet.log(with_results=True, stream=False)
    async def func(model, prompt, **params):
        return await async_generate(model, prompt, stream=False, params=params)

    original_response, results = await func(model, prompt, **params)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        f"""
        SELECT * FROM prompt_arguments WHERE id = '{results["prompt_arguments"].id}'
        """
    )
    fetched_prompt_arguments_table = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    fetched_prompt_arguments = {}
    for column, value in zip(columns, fetched_prompt_arguments_table):
        fetched_prompt_arguments[column] = value
    fetched_prompt_arguments["parameters"] = json.loads(
        fetched_prompt_arguments["parameters"]
    )
    fetched_prompt_arguments = PromptArgumentsRecord.model_validate(
        fetched_prompt_arguments
    )
    assert fetched_prompt_arguments == results["prompt_arguments"]

    cursor.execute(
        f"""
        SELECT * FROM response WHERE id = '{results["response"].id}'
        """
    )
    fetched_response_table = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    fetched_response = {}
    for column, value in zip(columns, fetched_response_table):
        fetched_response[column] = value
    fetched_response = ResponseRecord.model_validate(fetched_response)
    assert fetched_response == results["response"]

    conn.close()


def test_log_live_sqlite_streamed_sync(pellet, model, prompt, params, db_path):
    @pellet.log(with_results=True, stream=True)
    def func(model, prompt, **params):
        return generate(stream=True, model=model, prompt=prompt, params=params)

    original_responses = []
    results = {"prompt_arguments": None, "responses": []}
    for original_response, result in func(model, prompt, **params):
        original_responses.append(original_response)
        if result:
            if results["prompt_arguments"]:
                assert result["prompt_arguments"] == results["prompt_arguments"]
            results["prompt_arguments"] = result["prompt_arguments"]
            results["responses"].append(result["response"])

    prompt_arguments = results["prompt_arguments"]
    logger.debug(f"prompt_arguments: {prompt_arguments}")
    responses = results["responses"]
    logger.debug(f"len(responses): {len(responses)}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT * FROM prompt_arguments WHERE id = '{prompt_arguments.id}'
        """
    )
    fetched_prompt_arguments_table = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    fetched_prompt_arguments = {}
    for column, value in zip(columns, fetched_prompt_arguments_table):
        fetched_prompt_arguments[column] = value
    fetched_prompt_arguments["parameters"] = json.loads(
        fetched_prompt_arguments["parameters"]
    )
    fetched_prompt_arguments = PromptArgumentsRecord.model_validate(
        fetched_prompt_arguments
    )
    assert fetched_prompt_arguments == prompt_arguments
    response_ids = [response.id for response in responses]

    # Prepare the placeholders for the IN clause
    placeholders = ", ".join("?" for _ in response_ids)

    # Prepare the query
    query = f"SELECT * FROM response WHERE id IN ({placeholders})"

    cursor.execute(query, response_ids)
    fetched_responses_table = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    fetched_responses = []
    for fetched_response_record in fetched_responses_table:
        fetched_response = {}
        for column, value in zip(columns, fetched_response_record):
            fetched_response[column] = value
        fetched_response = ResponseStreamRecord.model_validate(fetched_response)
        fetched_responses.append(fetched_response)
    fetched_responses = [
        ResponseStreamRecord.model_validate(fetched_response)
        for fetched_response in fetched_responses
    ]
    for fetched_response, response in zip(fetched_responses, responses):
        assert fetched_response.response == response.response
        assert fetched_response.done == response.done
        assert fetched_response.prompt_arguments_id == response.prompt_arguments_id

    conn.close()


@pytest.mark.asyncio
async def test_log_live_sqlite_streamed_async(pellet, model, prompt, params, db_path):
    return

    @pellet.log(with_results=True, stream=True)
    async def func(model, prompt, **params):
        return await async_generate(model, prompt, stream=True, params=params)

    original_responses = []
    results = {"prompt_arguments": None, "responses": []}
    async for original_response, result in func(model, prompt, **params):
        original_responses.append(original_response)
        if result:
            if results["prompt_arguments"]:
                assert result["prompt_arguments"] == results["prompt_arguments"]
            results["prompt_arguments"] = result["prompt_arguments"]
            results["responses"].append(result["response"])

    prompt_arguments = results["prompt_arguments"]
    responses = results["responses"]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT * FROM prompt_arguments WHERE id = '{prompt_arguments.id}'
        """
    )
    fetched_prompt_arguments_table = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]
    fetched_prompt_arguments = {}
    for column, value in zip(columns, fetched_prompt_arguments_table):
        fetched_prompt_arguments[column] = value
    fetched_prompt_arguments["parameters"] = json.loads(
        fetched_prompt_arguments["parameters"]
    )
    fetched_prompt_arguments = PromptArgumentsRecord.model_validate(
        fetched_prompt_arguments
    )
    assert fetched_prompt_arguments == prompt_arguments
    response_ids = [response.id for response in responses]

    # Prepare the placeholders for the IN clause
    placeholders = ", ".join("?" for _ in response_ids)

    # Prepare the query
    query = f"SELECT * FROM response WHERE id IN ({placeholders})"

    cursor.execute(query, response_ids)
    fetched_responses_table = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    fetched_responses = []
    for fetched_response_record in fetched_responses_table:
        fetched_response = {}
        for column, value in zip(columns, fetched_response_record):
            fetched_response[column] = value
        fetched_response = ResponseStreamPieceRecord.model_validate(fetched_response)
        fetched_responses.append(fetched_response)
    fetched_responses = [
        ResponseStreamPieceRecord.model_validate(fetched_response)
        for fetched_response in fetched_responses
    ]
    for fetched_response, response in zip(fetched_responses, responses):
        assert fetched_response.response == response.response
        assert fetched_response.done == response.done
        assert fetched_response.prompt_arguments_id == response.prompt_arguments_id
        assert fetched_response.previous_response_id == response.previous_response_id
        assert fetched_response == response

    conn.close()


# Test the overhead of the decorator
def test_overhead(pellet, model, prompt, params, mock_generate, mock_generate_stream):
    # Set logging level to info
    logger.setLevel("INFO")
    # Measure the average time of the original function via timeit (non-streamed)
    original_time_non_stream = timeit.timeit(
        lambda: mock_generate, number=1000
    )  # 1000 iterations
    original_time_non_stream = f"{original_time_non_stream:.2e}"
    logger.info(f"original_time_non_stream: {original_time_non_stream}")

    # Measure the average time of the decorated function via timeit (non-streamed)
    decorated_time_non_stream = timeit.timeit(
        lambda: pellet.log(with_results=False, stream=False)(lambda: mock_generate),
        number=1000,
    )  # 1000 iterations
    decorated_time_non_stream = f"{decorated_time_non_stream:.2e}"
    logger.info(f"decorated_time_non_stream: {decorated_time_non_stream}")

    # Measure the average time of the original function via timeit (streamed), iterate over the response
    def original_streamed():
        for response in mock_generate_stream:
            pass

    original_time_stream = timeit.timeit(
        original_streamed, number=100
    )  # 1000 iterations
    original_time_stream = f"{original_time_stream:.2e}"
    logger.info(f"original_time_stream: {original_time_stream}")

    def decorated_streamed():
        @pellet.log(with_results=True, stream=True)
        def func(model, prompt, **params):
            return mock_generate_stream
        
        # profiler = cProfile.Profile()
        # profiler.enable()
        for response, result in func(model, prompt, **params):
            pass
        # profiler.disable()
        # s = io.StringIO()
        # stats = pstats.Stats(profiler, stream=s).sort_stats('time')
        # stats.print_stats()
        # logger.info(s.getvalue())

    decorated_time_stream = timeit.timeit(
        decorated_streamed, number=100
    )  # 1000 iterations
    decorated_time_stream = f"{decorated_time_stream:.2e}"
    logger.info(f"decorated_time_stream: {decorated_time_stream}")

    logger.info(
        f"overhead_non_stream: {(float(decorated_time_non_stream) - float(original_time_non_stream))/1000}"
    )
    logger.info(
        f"overhead_stream: {(float(decorated_time_stream) - float(original_time_stream))/100}"
    )
