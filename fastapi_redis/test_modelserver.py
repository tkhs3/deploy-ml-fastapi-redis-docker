import pytest
import redis
import modelserver


@pytest.fixture
def model():
    return ResNet50(weights="imagenet")


@pytest.fixture
def db():
    return redis.StrictRedis(host="redis")


@pytest.fixture
def server(model, db):

    processor = modelserver.processor
    processor.init(model=model, db=db)

    assert processor.db is not None
    assert processor.model is not None

    return processor

