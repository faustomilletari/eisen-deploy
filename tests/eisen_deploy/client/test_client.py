import pytest
import requests
import pickle
import numpy as np
import json

from eisen_deploy.client import EisenServingClient


DATA = {
    'image': np.random.randn(2, 3, 10, 426, 240).astype(np.float32),
    'string': 'test',
    'integer': 2
}

REF_RESPONSE = {
    'prediction': np.random.randn(2, 10).astype(np.float32),
    'class': 'test'
}

REF_METADATA = {
    'inputs': [
        {'name': 'image', 'type': 'np.ndarray', 'shape': [-1, 3, 10, 426, 240]},
        {'name': 'string', 'type': 'str', 'shape': None},
        {'name': 'integer', 'type': 'int', 'shape': None}
    ],
    'outputs': [
        {'name': 'prediction', 'type': 'np.ndarray', 'shape': [-1, 10]},
        {'name': 'class', 'type': 'str', 'shape': None}
    ],
    'model_input_list': ['image', 'integer'],
    'model_output_list': ['prediction'],
    'custom': {
        'custom': True
    }
}


class MockResponse:
    def __init__(self, json, content):
        self._json = json
        self.content = content

    def json(self):
        return json.loads(self._json)


def mock_post(*args, **kwargs):
    assert kwargs['url'] == 'http://test.com/prediction/model'

    if 'data' in kwargs.keys():
        sent_dict = pickle.loads(kwargs['data'])

        for key in sent_dict.keys():
            if isinstance(sent_dict[key], np.ndarray):
                assert np.all(sent_dict[key] == DATA[key])
            else:
                assert sent_dict[key] == DATA[key]

        assert all(sent_dict.keys() == DATA.keys())

        return MockResponse(json=None, content=pickle.dumps(REF_RESPONSE))
    else:
        return MockResponse(json=json.dumps(REF_METADATA), content=None)


@pytest.fixture(autouse=True)
def mock_requests(monkeypatch):
    monkeypatch.setattr(requests, "post", mock_post)


class TestEisenServingClient:
    def test_client(self):
        client = EisenServingClient('http://test.com/prediction/model', validate_inputs=False)

        assert client.metadata == REF_METADATA

        response = client.predict(DATA)

        assert np.all(response['prediction'] == REF_RESPONSE['prediction'])
        assert response['class'] == REF_RESPONSE['class']

