"""The versioned prompt used for independent neutral-dialogue generation."""

from __future__ import annotations

PROMPT_VERSION = "anthropic_neutral_dialogue_single_v8"

CHAT_PROMPT_ROLE = "user"
CHAT_SYSTEM_INSTRUCTION = (
    "Execute the user's complete generation and formatting specification literally. "
    "Do not substitute a different conversational task. Return only the requested "
    "final output."
)

GENERATION_REQUEST = (
    "Produce the final transcript now. Use exactly one self-contained Person turn "
    "and one complete AI turn. Follow the assigned task frame exactly rather than "
    "choosing a different kind of task. Use only the supplied topic and general "
    "knowledge, and follow the selected output structure literally, including any "
    "supplied first line. Silently rewrite the draft until every line is "
    "task-essential, then output only the transcript."
)

DIALOGUE_TYPES = [
    "factual_question",
    "code_or_programming",
    "work_task",
    "practical_how_to",
    "creative_neutral_task",
    "concise_information_exchange",
]

DIALOGUE_TYPE_INSTRUCTIONS = {
    "factual_question": (
        "Use the topic to ask about one stable general concept or one method for "
        "verifying a claim with an authoritative source. The answer must remain "
        "independent of unidentified people, unseen records, current local facts, "
        "and personalized judgment. Answer in no more than four concise sentences "
        "or one short list, and identify how context-dependent details can be "
        "verified."
    ),
    "code_or_programming": (
        "Use the topic to request one short, self-contained Python standard-library "
        "function that organizes, compares, validates, or summarizes supplied "
        "hypothetical data. The AI must include every import, no more than 25 lines "
        "of executable code, one small literal example, and a closed code fence. "
        "The function must not require network access, files, external packages, or "
        "an assumed dataset."
    ),
    "work_task": (
        "Use the topic to request one compact reusable checklist, table, form, or "
        "template for collecting or organizing information. The AI must produce the "
        "finished artifact with no more than eight entries. It must describe fields "
        "or procedures without inventing case facts, people, records, or outcomes."
    ),
    "practical_how_to": (
        "Use the topic to request a general procedure for documenting facts, "
        "comparing available records, locating an authoritative source, or preparing "
        "a neutral inquiry. The AI must provide three to six direct, numbered steps. "
        "Keep context-dependent actions conditional and avoid asserting case-specific "
        "rights, requirements, or outcomes."
    ),
    "creative_neutral_task": (
        "Use the topic to request six clearly proposed names, headings, categories, "
        "or design alternatives for a hypothetical artifact. The AI must deliver "
        "exactly six concise options whose proposed status is clear. The options "
        "must not be presented as real people, events, records, rules, or findings."
    ),
    "concise_information_exchange": (
        "Use the topic to request one stable definition, verification method, or "
        "two-sentence general summary. The AI must answer in no more than three "
        "concise sentences without drafting a social message or fictional scene."
    ),
}

SYSTEM_INSTRUCTION_REQUIRED = """The first line of the generated output must be exactly:

You should respond objectively and include only the information needed to complete the task.

This line is literal transcript content. Reproduce it in the output rather than merely following it silently.

It must:

- appear before the first Person turn;
- not use a \"System:\" label;
- be followed by exactly one blank line;
- remain separate from the Person's message."""

SYSTEM_INSTRUCTION_NOT_REQUIRED = (
    "Do not include a system instruction. Begin directly with the first Person turn."
)

OPTIONAL_SYSTEM_INSTRUCTION_INCLUDED = (
    "You should respond objectively and include only the information needed to "
    "complete the task."
)
OPTIONAL_SYSTEM_INSTRUCTION_NOT_INCLUDED = ""

NEUTRAL_DIALOGUE_PROMPT_TEMPLATE = """Write one dialogue based on the following topic.

Topic: {topic}

The dialogue must be between:

- Person, a human
- AI, an AI assistant

The Person should make one concrete, self-contained request that clearly matches the assigned dialogue type. Frame the topic as a general case rather than reenacting it as the speaker's personal or social experience.

The AI should provide one clear, useful, direct, and matter-of-fact response that completes the request without asking the Person for another turn.

The first speaker turn must always be Person.

Use this format:

{optional_system_instruction}

Person: [message]

AI: [response]

Write exactly one complete Person–AI exchange. Do not add a follow-up turn.

SYSTEM-INSTRUCTION REQUIREMENT:

{system_instruction_requirement}

IMPORTANT FORMATTING REQUIREMENTS:

- Write exactly one dialogue.
- Write exactly one Person turn and one AI turn.
- Always place a blank line before every speaker turn.
- Every speaker turn must begin with exactly \"Person:\" or \"AI:\".
- The first speaker label must be \"Person:\".
- Do not include a title.
- Do not include a dialogue number.
- Do not include labels such as \"[dialogue 1]\".
- Do not include an explanation, preface, analysis, or concluding note.
- Do not place the dialogue inside a Markdown code block.
- After the AI's substantive result, output nothing else—not a wrapper marker, status line, or note.

CRITICAL REQUIREMENT: The interaction must remain neutral and task-focused.

- Treat the topic as subject matter and obey the assigned task frame exactly.
- Every turn must define the request, add necessary task information, or deliver part of the result.
- The Person may state external circumstances needed for the task but must not describe an internal reaction or seek relational support.
- Start the AI turn with the result itself: the factual content, first numbered step, artifact, or code fence. Do not announce that an answer is about to follow.
- Do not use text for appreciation, encouragement, approval, reassurance, rapport, conversation management, or an offer to continue later.
- When a topic concerns a personal or interpersonal event, convert it into a factual question or practical request without turning it into a personal narrative.
- Make the assigned dialogue type unmistakable in both the request and the delivered result.
- The Person's request must be answerable immediately from supplied information and general knowledge within the assigned frame.
- Complete the requested task inside the single AI turn. If a detail is unavailable, provide a reusable method or template and identify the required input rather than asking a question.
- Deliver finished artifacts with concrete topic-relevant content. Leave fields unfilled only when the requested result is explicitly a reusable template, and identify the input required for each field.
- Do not invent external access, links, records, actions, or results. When a claim depends on missing context, state that dependency and give a verification method.
- Keep the complete transcript under 240 words and finish every answer, list, code block, and sentence.
- End immediately when the substantive answer is complete.
- Before producing the output, silently remove or rewrite anything that does not help define the request, provide information, or complete the task.

Any subject may be discussed factually. Neutrality concerns how the speakers communicate, not which subject they discuss.

LANGUAGE REQUIREMENT:

- Write the entire dialogue in English.
- Do not include Chinese characters or another non-Latin writing system.
- Do not include translations or sentences in another language.

FINAL SILENT GATE:

ASSIGNED TASK FRAME:

{dialogue_type_instruction}

Do not emit the draft until all of these are true:

- It contains exactly one self-contained request and one complete answer.
- The request and answer visibly satisfy the assigned dialogue type.
- Every sentence performs necessary task work rather than managing a social interaction.
- The answer claims no access, facts, or actions that were not supplied or generally knowable.
- The answer is complete, concise, and ends at a natural boundary.

If any condition is false, rewrite the draft silently before output.

Output only the dialogue."""


def render_neutral_dialogue_prompt(
    *,
    topic: str,
    dialogue_type: str,
    include_system_instruction: bool,
) -> str:
    """Render one complete prompt without carrying context from another job."""

    try:
        dialogue_type_instruction = DIALOGUE_TYPE_INSTRUCTIONS[dialogue_type]
    except KeyError as error:
        raise ValueError(f"Unknown dialogue type: {dialogue_type}") from error

    if include_system_instruction:
        optional_system_instruction = OPTIONAL_SYSTEM_INSTRUCTION_INCLUDED
        system_instruction_requirement = SYSTEM_INSTRUCTION_REQUIRED
    else:
        optional_system_instruction = OPTIONAL_SYSTEM_INSTRUCTION_NOT_INCLUDED
        system_instruction_requirement = SYSTEM_INSTRUCTION_NOT_REQUIRED

    return NEUTRAL_DIALOGUE_PROMPT_TEMPLATE.format(
        topic=topic,
        optional_system_instruction=optional_system_instruction,
        dialogue_type_instruction=dialogue_type_instruction,
        system_instruction_requirement=system_instruction_requirement,
    )


__all__ = [
    "CHAT_PROMPT_ROLE",
    "CHAT_SYSTEM_INSTRUCTION",
    "DIALOGUE_TYPES",
    "DIALOGUE_TYPE_INSTRUCTIONS",
    "GENERATION_REQUEST",
    "NEUTRAL_DIALOGUE_PROMPT_TEMPLATE",
    "OPTIONAL_SYSTEM_INSTRUCTION_INCLUDED",
    "OPTIONAL_SYSTEM_INSTRUCTION_NOT_INCLUDED",
    "PROMPT_VERSION",
    "SYSTEM_INSTRUCTION_NOT_REQUIRED",
    "SYSTEM_INSTRUCTION_REQUIRED",
    "render_neutral_dialogue_prompt",
]
