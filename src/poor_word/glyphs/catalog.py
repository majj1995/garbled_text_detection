from pathlib import Path


def _parse_line(line: str) -> tuple[str, ...]:
    fields = line.split("\t")
    if len(fields) >= 2 and fields[0].strip().isdigit():
        return (fields[1].strip(),)
    return tuple("".join(line.split()))


def load_common_chars(path: Path, expected_count: int = 3500) -> tuple[str, ...]:
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")

    characters: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        characters.extend(_parse_line(line))

    invalid = [character for character in characters if len(character) != 1]
    if invalid:
        raise ValueError("catalog entries must contain exactly one Unicode code point")
    if len(characters) != len(set(characters)):
        raise ValueError("catalog contains duplicate characters")
    if len(characters) != expected_count:
        raise ValueError(
            f"catalog must contain exactly {expected_count} characters, got {len(characters)}"
        )
    return tuple(characters)
