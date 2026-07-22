"""Prompt assets for independent emotional-story generation."""

from __future__ import annotations

PROMPT_VERSION = "anthropic_emotional_story_single_v3_english_only"

STORY_PROMPT_TEMPLATE = """Write one story based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Write only one self-contained story. Do not include a title, story number, label, explanation, preface, or analysis.

Keep the story between 180 and 250 words. Give it a complete narrative arc and finish the final sentence within that limit.

LANGUAGE REQUIREMENT: Write the entire story in English only. All narration, thoughts, names, quoted text, and dialogue must use English-language prose. Do not include Chinese characters, non-Latin scripts, translations, or words or sentences in another language.

The story should be a fresh start with no continuity with any previous story. Vary the character, setting, narrative perspective, actions, and phrasing across generations.

IMPORTANT: You must NEVER use the word "{emotion}" or any direct synonyms of it in the story. Instead, convey the emotion ONLY through:

- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named.

Output only the story."""


def render_story_prompt(*, topic: str, emotion: str) -> str:
    """Render the versioned single-story prompt for one independent call."""

    return STORY_PROMPT_TEMPLATE.format(topic=topic, emotion=emotion)


__all__ = ["PROMPT_VERSION", "STORY_PROMPT_TEMPLATE", "render_story_prompt"]
