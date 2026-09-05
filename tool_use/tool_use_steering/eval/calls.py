"""Structural checker for tool calls: model text -> a verdict dict.

Each model family emits calls in its own surface format, so parsing is
per-family and the family is an explicit argument, resolved from the model
id by `family_for` -- never auto-detected, so a verdict is always the
verdict of one named parser and each family is calibrated on its own
fixtures (tests/fixtures/):

  hermes      `<tool_call>\\n{"name": .., "arguments": ..}\\n</tool_call>`,
              what Qwen3 emits and eval/prompt.py's template trains.
              Calibrated against 352 checked Qwen generations; any edit here
              must keep that gate green.
  llama_json  bare `{"name": .., "parameters": ..}` objects, optionally
              several separated by `;` -- what Llama 3.1's own chat template
              instructs. Nested-brace safe. Reviewed against 300 checked
              Llama generations (2026-08-31, zero errors); promotion of
              those rows into tests/fixtures pending.
  bfcl_python `[func(a=1), pkg.func2(b="x")]` -- the BFCL prompting
              convention, used for models whose template ships no tool
              format (Gemma 2). AST-decoded like BFCL's own path.
              Fixture gate pending.

Output in no known call shape is plain text here; the judge still reads it
semantically, so call-like prose is not lost, just not a structural call.
Checks follow BFCL's AST evaluation: name offered -> no hallucinated params
-> required present. Types are recorded, not gated.

Verdict fields:
  attempt        a call was attempted in the format family
  valid          parsed AND passed all structural checks
  name_match     valid AND name == gold_names[0]  (None without gold)
  tool_call_success  valid AND name_match -- the strict-scoring bit
  invalid_reason unparseable | name_not_in_tools | hallucinated_param |
                 missing_required
  types_ok       declared-type agreement (recorded only)
  extra_text     prose outside the call block(s) (recorded only)
  n_calls        calls found (pools are single-call gold; FIRST is graded)
  call           the graded call as {"name", "args"} (None if unparseable)
"""

from __future__ import annotations

import json
import re

# Tolerant of a missing closing tag: truncation at the token budget produces
# exactly that (same tolerance as vLLM's parser).
_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
_BARE_JSON_CALL_RE = re.compile(r'\{[^{}]*"name"\s*:\s*"[^"]+"', re.S)
# llama_json attempt boundary: an opened call object counts as an attempt
# even when truncation cut it before any argument key.
_LLAMA_CALL_HINT_RE = re.compile(r'\{\s*"name"\s*:\s*"', re.S)

# bfcl_python attempt boundary: a bracketed call list is the convention;
# a bare `func(...)` at the start of the reply still reads as an attempt.
_BFCL_CALL_HINT_RE = re.compile(r"\[\s*[\w.]+\s*\(")

FAMILY_OF_MODEL = (("qwen", "hermes"), ("llama", "llama_json"),
                   ("gemma", "bfcl_python"))


def family_for(model: str) -> str:
    """The call-format family a model id parses under. Loud on unknowns:
    grading a new model with the wrong parser is a silent scoring bug."""
    lowered = model.lower()
    for key, family in FAMILY_OF_MODEL:
        if key in lowered:
            return family
    raise KeyError(f"no call-format family for model {model!r}; "
                   f"add it to calls.FAMILY_OF_MODEL")

_TYPE_MAP = {
    "string": str, "str": str,
    "integer": int, "int": int,
    "number": (int, float), "float": (int, float),
    "boolean": bool, "bool": bool,
    "array": list, "list": list, "tuple": list,
    "object": dict, "dict": dict,
}


def _fn(tool: dict) -> dict:
    return tool.get("function", tool)


def _parse(text: str, family: str):
    """-> (attempt: bool, calls: list|None, extra_text: bool)."""
    return _PARSERS[family](text)


def _parse_hermes(text: str):
    blocks = _BLOCK_RE.findall(text)
    if not blocks:
        # tag-less JSON that still has the call shape: an attempt whose
        # envelope broke (the template trains the tags)
        if _BARE_JSON_CALL_RE.search(text) and '"arguments"' in text:
            return True, None, False
        return False, None, False
    calls = []
    for block in blocks:
        try:
            obj = json.loads(block)
        except (ValueError, TypeError):
            return True, None, False
        if not isinstance(obj, dict) or "name" not in obj:
            return True, None, False
        args = obj.get("arguments")
        if args is not None and not isinstance(args, dict):
            return True, None, False
        calls.append({"name": obj["name"], "args": args or {}})
    outside = _BLOCK_RE.sub("", text).strip()
    return True, calls, bool(outside)


def _json_object_spans(text: str) -> list[tuple[int, int]]:
    """Top-level balanced {...} spans, string-literal aware."""
    spans = []
    depth, start, in_str, escaped = 0, 0, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    return spans


def _parse_llama(text: str):
    """Bare {"name": .., "parameters": ..} objects; `;`-separated when
    several. A call object opened but never closed (truncation) or not
    loadable as JSON is an attempt with no parse, mirroring hermes."""
    calls, call_spans = [], []
    for start, end in _json_object_spans(text):
        try:
            obj = json.loads(text[start:end])
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("parameters", obj.get("arguments"))
        if args is not None and not isinstance(args, dict):
            return True, None, False
        calls.append({"name": obj["name"], "args": args or {}})
        call_spans.append((start, end))
    if not calls:
        attempt = bool(_LLAMA_CALL_HINT_RE.search(text))
        return attempt, None, False
    outside = text
    for start, end in reversed(call_spans):
        outside = outside[:start] + outside[end:]
    return True, calls, bool(outside.strip("; \t\n"))


def _parse_bfcl_python(text: str):
    """BFCL prompting-convention calls: [func(a=1), pkg.Cls.func2(b="x")].
    Decoded the way BFCL's own AST path does: strip code fences, ast-parse
    the bracketed list, keyword values via literal_eval, dotted names
    reassembled from attribute chains. Anything call-shaped that fails to
    parse is an attempt with no parse; prose brackets without a call shape
    are not attempts."""
    import ast
    s = text.strip()
    s = re.sub(r"^```(?:python|json|tool_code)?\s*|\s*```$", "", s, flags=re.S).strip()
    m = _BFCL_CALL_HINT_RE.search(s)
    if not m:
        return bool(re.match(r"[\w.]+\s*\($", s.split("\n")[0].strip()[:80])
                    or re.match(r"[\w.]+\s*\(.*\)\s*$", s)), None, False
    start = m.start()
    end = s.rfind("]")
    if end <= start:                      # truncated before the close
        return True, None, False
    try:
        node = ast.parse(s[start:end + 1], mode="eval").body
    except SyntaxError:
        return True, None, False
    if not isinstance(node, ast.List) or not node.elts:
        return True, None, False
    calls = []
    for el in node.elts:
        if not isinstance(el, ast.Call) or el.args:   # positional args are
            return True, None, False                  # outside the convention
        parts, f = [], el.func
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if not isinstance(f, ast.Name):
            return True, None, False
        parts.append(f.id)
        args = {}
        for kw in el.keywords:
            if kw.arg is None:
                return True, None, False
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                return True, None, False
        calls.append({"name": ".".join(reversed(parts)), "args": args})
    outside = (s[:start] + s[end + 1:]).strip()
    return True, calls, bool(outside)


_PARSERS = {"hermes": _parse_hermes, "llama_json": _parse_llama,
            "bfcl_python": _parse_bfcl_python}


def _types_ok(args: dict, props: dict) -> bool:
    for param, value in args.items():
        declared = _TYPE_MAP.get(str(props.get(param, {}).get("type", "")).lower())
        if declared is None or value is None:
            continue
        if isinstance(value, bool) and declared is not bool:
            return False
        if not isinstance(value, declared):
            return False
    return True


def check(text: str, tools: list, gold_names: list | None = None, *,
          family: str) -> dict:
    verdict = {"attempt": False, "valid": False, "name_match": None,
               "tool_call_success": False, "invalid_reason": None,
               "types_ok": None, "extra_text": False, "n_calls": 0,
               "call": None}

    attempt, calls, extra = _parse(text, family)
    if not attempt:
        return verdict                                    # plain text
    verdict.update(attempt=True, extra_text=extra)
    if calls is None:
        verdict["invalid_reason"] = "unparseable"
        return verdict

    verdict.update(n_calls=len(calls), call=calls[0])
    call = calls[0]

    offered = {_fn(t)["name"]: _fn(t) for t in tools}
    if call["name"] not in offered:
        verdict["invalid_reason"] = "name_not_in_tools"
        return verdict
    schema = offered[call["name"]].get("parameters") or {}
    props = schema.get("properties") or {}
    if props and any(p not in props for p in call["args"]):
        verdict["invalid_reason"] = "hallucinated_param"
        return verdict
    missing = [p for p in (schema.get("required") or []) if p not in call["args"]]
    if missing:
        verdict["invalid_reason"] = "missing_required"
        return verdict

    verdict["valid"] = True
    verdict["types_ok"] = _types_ok(call["args"], props)

    # The strict bit stays out of `valid` on purpose: a well-formed call to
    # the WRONG offered tool is a real mis-call and must stay a call.
    if gold_names:
        verdict["name_match"] = call["name"] == gold_names[0]
        verdict["tool_call_success"] = bool(verdict["name_match"])
    return verdict


def used_value(verdict: dict, slot: str, value: str) -> bool | None:
    """Did the graded call use the value the user just supplied for `slot`?

    The follow-up scorer's question. None when there is no valid call to ask
    it of. Comparison is string-normalised: the value crossed a conversation
    turn as text, so "10" and 10 are the same answer.
    """
    if not verdict["valid"]:
        return None
    args = verdict["call"]["args"]
    if slot not in args:
        return False
    got = args[slot]
    return str(got).strip().lower() == str(value).strip().lower()
