import re
from typing import List, Tuple

DEFAULT_MIN_WORDS = 28
DEFAULT_MAX_WORDS = 45

_SPEAKER_LINE_RE = re.compile(r"^(Speaker\s+\d+\s*:\s*)(.*)$", re.IGNORECASE)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _split_long_unit_by_words(unit: str, min_words: int, max_words: int) -> List[str]:
    tokens = unit.split()
    if len(tokens) <= max_words:
        return [unit.strip()] if unit.strip() else []

    chunks: List[str] = []
    i = 0
    while i < len(tokens):
        remaining = len(tokens) - i
        if remaining <= max_words:
            chunks.append(" ".join(tokens[i:]).strip())
            break

        start = i + min_words
        end = min(i + max_words, len(tokens))
        candidate_end = end

        # Priority 1: sentence-ending punctuation nearest to max words.
        for idx in range(end - 1, start - 1, -1):
            if re.search(r"[.!?][\"')\]]?$", tokens[idx]):
                candidate_end = idx + 1
                break
        else:
            # Priority 2: secondary punctuation nearest to max words.
            for idx in range(end - 1, start - 1, -1):
                if re.search(r"[,;:][\"')\]]?$", tokens[idx]):
                    candidate_end = idx + 1
                    break

        # Priority 3 fallback: cut near max words.
        if candidate_end <= i:
            candidate_end = end

        chunks.append(" ".join(tokens[i:candidate_end]).strip())
        i = candidate_end

    return [c for c in chunks if c]


def _parse_speaker_line(line: str) -> Tuple[str, str]:
    match = _SPEAKER_LINE_RE.match(line)
    if match:
        return match.group(1), match.group(2).strip()
    # Keep output valid for VibeVoice parser even if an input line is malformed.
    return "Speaker 0: ", line.strip()


def chunk_script_long_form(
    script: str,
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
) -> List[str]:
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    if not lines:
        return []

    chunks: List[str] = []
    current_chunk_lines: List[str] = []
    current_words = 0

    for line in lines:
        speaker_prefix, content = _parse_speaker_line(line)
        if not content:
            continue

        # Split this speaker segment into sentence-level units first.
        sentence_units = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
        if not sentence_units:
            sentence_units = [content]

        expanded_units: List[str] = []
        for unit in sentence_units:
            if _word_count(unit) > max_words:
                expanded_units.extend(_split_long_unit_by_words(unit, min_words=min_words, max_words=max_words))
            else:
                expanded_units.append(unit)

        for unit in expanded_units:
            unit_words = _word_count(unit)
            candidate_line = f"{speaker_prefix}{unit}".strip()

            if current_words + unit_words <= max_words:
                current_chunk_lines.append(candidate_line)
                current_words += unit_words
            else:
                if current_chunk_lines:
                    chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = [candidate_line]
                current_words = unit_words

    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    return [c for c in chunks if c.strip()]
