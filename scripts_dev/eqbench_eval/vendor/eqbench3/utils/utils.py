import json
from typing import Any, Dict


def robust_json_loads(raw_text: str) -> Dict[str, Any]:
    """
    Parse a JSON object embedded in judge output that may contain:
      • multiple ```json fenced blocks       → keep only the last
      • fence markers (``` or ```json)       → strip them
      • unescaped newlines inside strings    → repair
      • stray double-quotes inside strings   → repair

    Returns the decoded object or raises JSONDecodeError.
    """

    # ────────────────────  stage 1: isolate last ```json block  ────────────────────
    if raw_text.count("```json") > 1:
        raw_text = raw_text[raw_text.rfind("```json") + len("```json"):]

    # ────────────────────  stage 2: drop any fence lines  ────────────────────
    cleaned = "\n".join(
        line for line in raw_text.splitlines()
        if not line.lstrip().startswith("```")
    ).strip()

    # ────────────────────  stage 3: extract the first balanced JSON object  ────────────────────
    def extract_first_json(s: str) -> str | None:
        in_string = False
        escape_next = False
        depth = 0
        start = None

        for i, ch in enumerate(s):
            if in_string:
                if escape_next:
                    escape_next = False
                elif ch == "\\":
                    escape_next = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        return s[start : i + 1]
        return None

    json_str = extract_first_json(cleaned)
    if json_str is None:
        raise json.JSONDecodeError("No balanced JSON object found", cleaned, 0)

    # ────────────────────  stage 4: quick parse  ────────────────────
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass  # fall through to repair passes

    # ────────────────────  stage 5a: repair unescaped newlines  ────────────────────
    def repair_newlines(s: str) -> str:
        out, in_string, esc = [], False, False
        for ch in s:
            if in_string:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_string = False
                elif ch in "\r\n":
                    ch = "\\n"
            else:
                if ch == '"':
                    in_string = True
            out.append(ch)
        return "".join(out)

    fixed = repair_newlines(json_str)

    # ────────────────────  stage 5b: repair stray inner quotes  ────────────────────
    def repair_inner_quotes(s: str) -> str:
        out, in_string, esc = [], False, False
        i, n = 0, len(s)
        while i < n:
            ch = s[i]
            if in_string:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    # peek ahead for illegal terminator
                    j = i + 1
                    while j < n and s[j].isspace():
                        j += 1
                    if j < n and s[j] not in {",", "}", "]", ":"}:
                        out.append("\\")  # escape
                    else:
                        in_string = False
            else:
                if ch == '"':
                    in_string = True
            out.append(ch)
            i += 1
        return "".join(out)

    fixed = repair_inner_quotes(fixed)

    # ────────────────────  stage 6: final parse  ────────────────────
    return json.loads(fixed)
