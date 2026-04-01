import json
import os
from copy import deepcopy


DEFAULT_CONFIG = {
    "paths": {
        "data": "data",
        "vectorstore": "vectorstore",
        "sources": "sources.txt",
        "chroma": "chroma"
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "embedding_model": "nomic-embed-text",
        "chat_model": "tinyllama"
    },
    "groq": {
        "model": "llama-3.1-8b-instant",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "temperature": 0.2,
        "max_tokens": 512,
        "top_p": 0.85
    },
    "retrieval": {
        "default_k": 3,
        "reject_threshold": 0.78,
        "warning_threshold": 0.75,
        "max_context_online": 2000,
        "max_context_offline": 1200
    },
    "tinyllama": {
        "temperature": 0.2,
        "num_predict": 192,
        "top_k": 10,
        "top_p": 0.85,
        "num_thread": 4,
        "repeat_penalty": 1.1
    },
    "network": {
        "internet_check_host": "8.8.8.8",
        "internet_check_port": 53,
        "internet_check_timeout": 5,
        "ollama_check_timeout": 3
    },
    "database": {
        "chunk_size": 1000,
        "chunk_overlap": 250
    }
}


def _deep_merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")

    config = deepcopy(DEFAULT_CONFIG)

    if not os.path.exists(config_path):
        return config

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = json.load(f)

    if not isinstance(user_config, dict):
        return config

    return _deep_merge(config, user_config)
