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
# 1. Test that the decorator correctly reads the prompt and model for a non-streamed response
# 2. Test that the decorator correctly reads the response for a non-streamed response
# 3. Test that the decorator correctly reads the prompt and model for a streamed response
# 4. Test that the decorator correctly reads the response for a streamed response
# 5. Test that the decorator correctly logs the prompt, model, and response for a non-streamed response to a sqlite database
# 6. Test that the decorator correctly logs the prompt, model, and response for a streamed response to a sqlite database
# 7. Test that the decorator correctly reads the prompt, model, and response from the sqlite database

import pytest  # for the fixture, among other things
import requests  # for the HTTP requests
import json  # for parsing the response
import sqlite3  # for the database
import datetime  # for the timestamp
import uuid  # for the UUID
import os  # for the file handling
import asyncio  # for the async tests

from pellets.pellet import Pellet
from pellets.records import (
    PromptArgumentsRecord,
    ResponseRecord,
    PromptArgumentsCreate,
    ResponseCreate,
)
from pellets.logger import logger


# Set up the model, prompt, Pellet
@pytest.fixture
def model():
    return "llama2"


@pytest.fixture
def prompt():
    return "Why is the sky blue?"


@pytest.fixture
def pellet():
    return Pellet()


def generate(model, prompt, stream):
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": stream}
    response = requests.post(url, json=data)
    assert response.status_code == 200
    return response


async def async_generate(model, prompt, stream):
    url = "http://localhost:11434/api/generate"
    data = {"model": model, "prompt": prompt, "stream": stream}
    response = requests.post(url, json=data)
    assert response.status_code == 200
    return response


# Test that the decorator correctly reads the prompt and model for a non-streamed response


def test_log_read_arguments_non_streamed_sync(pellet, model, prompt, mocker):
    @pellet.log(with_results=True)
    def func(model, prompt):
        # Mock with pytests-mock
        response_mock = mocker.MagicMock()
        response_mock.json.return_value = {"response": "The sky is blue because it is the color of the sky."}
        response_mock.status_code = 200
        return response_mock

    response, results = func(model, prompt)

    assert results["prompt_arguments"] == PromptArgumentsRecord(
        id=results["prompt_arguments"].id,
        **PromptArgumentsCreate(model=model, prompt=prompt).model_dump(),
    )
    assert isinstance(results["response"], ResponseRecord)


def test_log_read_arguments_non_streamed_async(pellet, model, prompt, mocker):
    @pellet.log(with_results=True)
    async def func(model, prompt):
        # Mock with pytests-mock
        response_mock = mocker.MagicMock()
        response_mock.json.return_value = {"response": "The sky is blue because it is the color of the sky."}
        response_mock.status_code = 200
        return response_mock

    response, results = asyncio.run(func(model, prompt))
    
    assert results["prompt_arguments"] == PromptArgumentsRecord(
        id=results["prompt_arguments"].id,
        **PromptArgumentsCreate(model=model, prompt=prompt).model_dump(),
    )
    assert isinstance(results["response"], ResponseRecord)



# Test that the decorator correctly reads the response for a non-streamed response
