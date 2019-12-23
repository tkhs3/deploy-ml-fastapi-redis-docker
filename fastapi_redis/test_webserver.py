import pytest
from fastapi import FastAPI, File, HTTPException
import redis
import webserver

import PIL

@pytest.fixture
def app():
    return FastAPI()


@pytest.fixture
def db():
    return redis.StrictRedis(host="redis")


@pytest.fixture
def web(app, db):

    processor = webserver.processor
    processor.init(app=app, db=db)

    assert processor.db is not None
    assert processor.app is not None

    return processor


def test_register_route(web):
    web.register_route()

    assert web.predict is not None
    assert web.index is not None
