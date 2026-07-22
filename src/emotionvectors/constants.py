"""Canonical constants for the released Qwen2.5-7B emotion vectors."""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TOKENIZER_CLASS = "Qwen2TokenizerFast"
MODEL_CLASS = "Qwen2ForCausalLM"

NUM_LAYERS = 28
HIDDEN_SIZE = 3584
HIDDEN_STATE_MAPPING = "saved layer l equals outputs.hidden_states[l + 1]"
RESIDUAL_STREAM_DEFINITION = "output after each transformer layer"

EMOTIONS = (
    "anger",
    "fear",
    "disgust",
    "sadness",
    "anxiety",
    "desperation",
    "frustration",
    "hostility",
    "calmness",
    "compassion",
    "joy",
    "trust",
)

STORY_PROMPT_VERSION = "anthropic_emotional_story_single_v3_english_only"
NEUTRAL_PROMPT_VERSION = "anthropic_neutral_dialogue_single_v8"
STORY_START_TOKEN_POSITION = 40
NEUTRAL_START_TOKEN_POSITION = 1
VISUALIZATION_LAYER_INDEX = 18
VISUALIZATION_LAYER_NUMBER = 19

__all__ = [
    "EMOTIONS",
    "HIDDEN_SIZE",
    "HIDDEN_STATE_MAPPING",
    "MODEL_CLASS",
    "MODEL_ID",
    "MODEL_REVISION",
    "NEUTRAL_PROMPT_VERSION",
    "NEUTRAL_START_TOKEN_POSITION",
    "NUM_LAYERS",
    "RESIDUAL_STREAM_DEFINITION",
    "STORY_PROMPT_VERSION",
    "STORY_START_TOKEN_POSITION",
    "TOKENIZER_CLASS",
    "VISUALIZATION_LAYER_INDEX",
    "VISUALIZATION_LAYER_NUMBER",
]
