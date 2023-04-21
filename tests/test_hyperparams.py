import pytest
import json

config = json.load('config.json')


def test_hyperparams():
    assert config["hidden_size"] % config["num_attention_heads"] == 0
    assert config['intermediate_size'] == 4 * config['hidden_size']
    assert config['image_size'] % config['patch_size'] == 0
