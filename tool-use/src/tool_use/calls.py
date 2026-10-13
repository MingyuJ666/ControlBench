"""Structural checker for tool calls: model text -> a verdict.

Every model is prompted in BFCL's convention and told to answer
`[func_name(param=value)]`, but a model writes what it was trained to write:
Qwen3 reaches for `<tool_call>{...}</tool_call>` and Llama 3.1 for a bare
`{"name": .., "parameters": ..}` object. A call is a call whichever syntax it
arrives in, so all three are recognised and the one used is recorded. Reading
only the instructed syntax would score a correct decision as no decision at
all, and would confuse "steered away from calling" with "stopped using the
requested format".

The parser detects call-shaped text, including quoted examples. The judge
separately decides whether the response actually submits an invocation; a
structural match alone never settles the scored action.

The decision checker matches the current When2Call single-call references:
it requires exactly one call, and the name must be offered,
parameters must be declared, and every required parameter must be present.
Positional arguments bind in the order of the published schema properties.
Declared argument types are enforced, including nested values. The shared
parser still returns multiple calls for the multi-turn executor.

Verdict fields:
  attempt        a call was attempted, in any of the three syntaxes
  syntax         bfcl | hermes | llama_json, or None
  valid          parsed AND passed every structural check
  name_match     valid AND name == gold_names[0]  (None without gold)
  success        valid AND name_match -- the strict bit
  invalid_reason unparseable | name_not_in_tools | invented_param |
                 missing_required | multiple_calls | invalid_name |
                 too_many_positional | duplicate_param | type_mismatch
  types_ok       declared-type agreement (required for validity)
  extra_text     prose outside the call (recorded only)
  n_calls        calls found; single-turn evaluation requires exactly one
  call           the call as {"name", "args", "positional"}; after successful
                 validation, positional arguments are bound into args
  positional_args number of positional arguments in the original reply
"""

import ast
import json
import math
import re

# Tolerant of a missing closing tag: truncation at the token budget produces
# exactly that (the same tolerance vLLM's own parser has).
_HERMES_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
_HERMES_HINT = re.compile(r'\{[^{}]*"name"\s*:\s*"[^"]+"', re.S)
#: An opened call object counts as an attempt even when truncation cut it
#: before any argument key.
_LLAMA_HINT = re.compile(r'\{\s*"name"\s*:\s*"', re.S)
#: A bracketed call list is BFCL's convention; a bare `func(...)` opening the
#: reply still reads as an attempt.
_BFCL_HINT = re.compile(r"\[\s*[\w.]+\s*\(")
_FENCE = re.compile(r"^```(?:python|json|tool_code)?\s*|\s*```$", re.S)

_TYPES = {
    "string": str, "str": str,
    "integer": int, "int": int,
    "number": (int, float), "float": (int, float),
    "boolean": bool, "bool": bool,
    "array": list, "list": list, "tuple": list,
    "object": dict, "dict": dict,
}


def _fn(tool: dict) -> dict:
    return tool.get("function", tool)


def _json_object_spans(text: str) -> list[tuple[int, int]]:
    """Top-level balanced {...} spans, string-literal aware."""
    spans = []
    depth, start, in_string, escaped = 0, 0, False, False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append((start, index + 1))
    return spans


def _parse_bfcl(text: str):
    """Decode bracketed calls or complete bare calls using the same AST checks.

    BFCL accepts omitted outer brackets. Normalize the parsed shape rather
    than changing the response text; argument lists and strings stay intact.
    """
    stripped = _FENCE.sub("", text.strip()).strip()
    bare = bool(re.match(r"[\w.]+\s*\(", stripped))
    hint = _BFCL_HINT.search(stripped)
    if bare:
        start, end = 0, len(stripped) - 1
    elif hint is not None:
        start, end = hint.start(), stripped.rfind("]")
        if end <= start:                                # truncated before the close
            return True, None, False
    else:
        return False, None, False
    try:
        node = ast.parse(stripped[start:end + 1], mode="eval").body
    except (SyntaxError, MemoryError, RecursionError):
        return True, None, False
    if bare and isinstance(node, ast.Call):
        elements = [node]
    elif isinstance(node, ast.Tuple if bare else ast.List) and node.elts:
        elements = node.elts
    else:
        return True, None, False
    calls = []
    for element in elements:
        if not isinstance(element, ast.Call):
            return True, None, False
        parts, func = [], element.func
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if not isinstance(func, ast.Name):
            return True, None, False
        parts.append(func.id)
        positional = []
        for argument in element.args:
            try:
                positional.append(ast.literal_eval(argument))
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                return True, None, False
        args = {}
        for keyword in element.keywords:
            if keyword.arg is None or keyword.arg in args:
                return True, None, False
            try:
                args[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                return True, None, False
        calls.append({"name": ".".join(reversed(parts)), "args": args,
                      "positional": positional})
    # Python literals such as Ellipsis, sets and bytes are not tool JSON values.
    # Reject them here instead of allowing score serialization to crash later.
    try:
        json.dumps(calls, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return True, None, False
    outside = (stripped[:start] + stripped[end + 1:]).strip()
    return True, calls, outside


def _parse_hermes(text: str):
    """`<tool_call>{"name": .., "arguments": ..}</tool_call>`, what Qwen3 writes."""
    blocks = _HERMES_BLOCK.findall(text)
    if not blocks:
        if _HERMES_HINT.search(text) and '"arguments"' in text:
            return True, None, False                    # the envelope broke
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
    return True, calls, _HERMES_BLOCK.sub("", text).strip()


def _parse_llama(text: str):
    """Bare `{"name": .., "parameters": ..}` objects, `;`-separated when several."""
    calls, spans = [], []
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
        spans.append((start, end))
    if not calls:
        return bool(_LLAMA_HINT.search(text)), None, False
    outside = text
    for start, end in reversed(spans):
        outside = outside[:start] + outside[end:]
    return True, calls, outside.strip("; \t\n")


#: Tried in order. BFCL first because it is what every model was asked for;
#: the other two catch a model writing the format it was trained on.
_PARSERS = (("bfcl", _parse_bfcl), ("hermes", _parse_hermes), ("llama_json", _parse_llama))


def _parse_with_outside(text: str):
    """Return the parsed calls and the unconsumed text from the first syntax."""
    attempted = None
    for name, parser in _PARSERS:
        attempt, calls, extra = parser(text)
        if attempt and calls is not None:
            return True, calls, extra, name
        if attempt and attempted is None:
            attempted = name
    return (True, None, False, attempted) if attempted else (False, None, False, None)


def parse(text: str):
    """-> (attempt, calls, extra_text, syntax), preserving calls for execution."""
    attempt, parsed, outside, syntax = _parse_with_outside(text)
    return attempt, parsed, bool(outside), syntax


def _value_types_ok(value, schema: dict) -> bool:
    """Declared types only; descriptions and semantic constraints go to the judge.

    Never coerce a model argument (including numeric strings and null). Enum
    membership is assessed by the judge against the complete definition; some
    source tools encode integer enum choices as strings.
    """
    declared = schema.get("type")
    if isinstance(declared, list):
        return any(_value_types_ok(value, {**schema, "type": name}) for name in declared)
    name = str(declared).lower()
    if name == "null":
        return value is None
    expected = _TYPES.get(name)
    if expected is not None:
        if isinstance(value, bool) and expected is not bool:
            return False
        if not isinstance(value, expected):
            return False
        if isinstance(value, float) and not math.isfinite(value):
            return False
    if isinstance(value, dict):
        return _types_ok(value, schema.get("properties") or {})
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return all(_value_types_ok(element, schema["items"]) for element in value)
    return True


def _types_ok(args: dict, properties: dict) -> bool:
    return all(_value_types_ok(value, properties.get(parameter, {}))
               for parameter, value in args.items())


def _nested_signature_error(value, schema: dict) -> str | None:
    if isinstance(value, dict):
        properties = schema.get("properties")
        if any(key not in value for key in schema.get("required") or []):
            return "missing_required"
        if properties is not None:
            if any(key not in properties for key in value):
                return "invented_param"
            for key, child in value.items():
                error = _nested_signature_error(child, properties[key])
                if error:
                    return error
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for child in value:
            error = _nested_signature_error(child, schema["items"])
            if error:
                return error
    return None


def check(text: str, tools: list, gold_names: list | None = None) -> dict:
    verdict = {"attempt": False, "syntax": None, "valid": False, "name_match": None,
               "success": False, "invalid_reason": None, "types_ok": None,
               "extra_text": False, "n_calls": 0, "call": None, "positional_args": 0}

    attempt, parsed, outside, syntax = _parse_with_outside(text)
    if not attempt:
        return verdict
    verdict.update(attempt=True, syntax=syntax, extra_text=bool(outside))
    if not parsed:
        verdict["invalid_reason"] = "unparseable"
        return verdict

    verdict.update(n_calls=len(parsed), call=parsed[0])
    if outside:
        more_attempted, more_calls, _, _ = parse(outside)
        if more_attempted:
            verdict["n_calls"] += len(more_calls) if more_calls else 1
    if verdict["n_calls"] != 1:
        verdict["invalid_reason"] = "multiple_calls"
        return verdict

    call = parsed[0]
    if not isinstance(call["name"], str) or not call["name"].strip():
        verdict["invalid_reason"] = "invalid_name"
        return verdict
    offered = {_fn(tool)["name"]: _fn(tool) for tool in tools}
    if call["name"] not in offered:
        verdict["invalid_reason"] = "name_not_in_tools"
        return verdict
    schema = offered[call["name"]].get("parameters") or {}
    properties = schema.get("properties") or {}
    positional = call.get("positional") or []
    verdict["positional_args"] = len(positional)
    if any(parameter not in properties for parameter in call["args"]):
        verdict["invalid_reason"] = "invented_param"
        return verdict
    if len(positional) > len(properties):
        verdict["invalid_reason"] = "too_many_positional"
        return verdict
    bound = dict(zip(properties, positional))
    if bound.keys() & call["args"].keys():
        verdict["invalid_reason"] = "duplicate_param"
        return verdict
    bound.update(call["args"])
    if any(parameter not in bound for parameter in schema.get("required") or []):
        verdict["invalid_reason"] = "missing_required"
        return verdict

    verdict["call"] = {**call, "args": bound, "positional": []}
    verdict["types_ok"] = _types_ok(bound, properties)
    if not verdict["types_ok"]:
        verdict["invalid_reason"] = "type_mismatch"
        return verdict
    error = _nested_signature_error(bound, schema)
    if error:
        verdict["invalid_reason"] = error
        return verdict
    verdict["valid"] = True
    if gold_names:
        verdict["name_match"] = call["name"] == gold_names[0]
        verdict["success"] = bool(verdict["name_match"])
    return verdict


def used_value(verdict: dict, slot: str, value: str) -> bool | None:
    """Did the graded call use the value the user just supplied for `slot`?

    None when there is no valid call to ask it of. The comparison is
    string-normalised: the value crossed a conversation turn as text, so "10"
    and 10 are the same answer.
    """
    if not verdict["valid"]:
        return None
    args = verdict["call"]["args"]
    if slot not in args:
        return False
    return str(args[slot]).strip().lower() == str(value).strip().lower()
