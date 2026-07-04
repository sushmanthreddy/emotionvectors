"""Shared lexical references for generation quality checks and verification."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

# These are deliberately compact lexical checks, not sentiment classifiers.  Stage 1
# uses the inflected surface forms to reject explicit naming/direct synonyms while
# still allowing stories to convey an emotion indirectly.
EMOTION_SYNONYMS: Mapping[str, tuple[str, ...]] = {
    "excited": ("excitement", "eager", "eagerness", "thrill", "thrilled", "anticipation"),
    "elated": ("delight", "delighted", "overjoyed", "exhilarated", "exhilaration"),
    "ecstatic": ("euphoric", "euphoria", "rapturous", "rapture", "blissful", "bliss"),
    "enthusiastic": ("enthusiasm", "zealous", "zeal", "passionate"),
    "joyful": ("joy", "happy", "happiness", "glad", "gleeful"),
    "content": ("contented", "satisfied", "satisfaction", "fulfilled"),
    "calm": ("peaceful", "tranquil", "composed", "unruffled"),
    "serene": ("serenity", "tranquility", "placid"),
    "grateful": ("gratitude", "thankful", "appreciative"),
    "relaxed": ("relaxation", "at ease", "unworried", "laid-back"),
    "angry": ("anger", "mad", "irate", "rage", "enraged"),
    "furious": ("fury", "raging", "infuriated", "livid"),
    "terrified": ("terror", "afraid", "frightened", "horrified"),
    "anxious": ("anxiety", "worried", "worry", "uneasy", "nervous"),
    "panicked": ("panic", "frantic", "alarmed"),
    "outraged": ("outrage", "indignant", "indignation", "offended"),
    "sad": ("sadness", "grief", "grieving", "sorrow", "unhappy", "tearful"),
    "depressed": ("depression", "hopeless", "hopelessness", "despair", "despondent"),
    "gloomy": ("gloom", "bleak", "somber", "dreary"),
    "lonely": ("loneliness", "lonesome", "isolated"),
    "miserable": ("misery", "wretched", "suffering"),
    "bored": ("boredom", "uninterested", "tedious", "dull"),
    "surprised": ("surprise", "astonished", "astonishment", "startled", "amazed"),
    "proud": ("pride", "self-satisfied"),
    "hopeful": ("hope", "optimistic", "optimism"),
    "nostalgic": ("nostalgia", "wistful", "reminiscent"),
    "guilty": ("guilt", "remorse", "remorseful", "culpable"),
    "ashamed": ("shame", "embarrassed", "humiliated", "disgraced"),
    "jealous": ("jealousy", "envious", "envy", "possessive"),
    "disgusted": ("disgust", "revolted", "repulsed", "revulsion", "nauseated"),
}

NEUTRAL_GENERIC_BANS: tuple[str, ...] = (
    "emotion",
    "emotions",
    "emotional",
    "emotionally",
    "feeling",
    "feelings",
    "thank you",
    "thanks",
    "great question",
    "happy to help",
    "glad to help",
    "sorry",
)


def forbidden_terms(
    text: str, emotions: Iterable[str], *, neutral: bool = False
) -> tuple[str, ...]:
    """Return explicit emotion labels/synonyms found as complete surface forms."""

    candidates: list[str] = []
    for emotion in emotions:
        label = str(emotion).strip().lower()
        # ``content`` is a common neutral noun (document content). Its unambiguous
        # affective synonyms remain prohibited in neutral dialogues.
        if label and not (neutral and label == "content"):
            candidates.append(label)
        candidates.extend(EMOTION_SYNONYMS.get(label, ()))
    if neutral:
        candidates.extend(NEUTRAL_GENERIC_BANS)

    matches = {
        term
        for term in candidates
        if term
        and re.search(
            rf"(?<![a-z]){re.escape(term.lower())}(?![a-z])",
            text.lower(),
        )
    }
    return tuple(sorted(matches))


__all__ = ["EMOTION_SYNONYMS", "NEUTRAL_GENERIC_BANS", "forbidden_terms"]
