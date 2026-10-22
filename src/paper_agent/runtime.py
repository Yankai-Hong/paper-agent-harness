"""Bounded tool loop adapted from learn-claude-code s02, MIT.

Source commit ce8f9f186058939da54c9d6fead78dfb5d0fd6c3.
Differences: injectable client, ID validation, tool allowlist, errors, budgets,
hooks, and usage accounting. No shell tools or implicit API credentials.
"""
from dataclasses import dataclass, field


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add(self, value):
        for key in self.__dataclass_fields__:
            count = getattr(value, key, 0) if not isinstance(value, dict) else value.get(key, 0)
            setattr(self, key, getattr(self, key) + int(count or 0))


@dataclass
class LoopResult:
    text: str
    usage: Usage
    calls: list = field(default_factory=list)


def tool_loop(client, model, messages, system, tools, handlers, max_calls=6, max_tools=8, hook=None):
    """Client implements Anthropic's messages.create; tests inject scripted responses."""
    import json
    usage, calls = Usage(), []
    allowed = {tool["name"] for tool in tools}
    for _ in range(max_calls):
        response = client.messages.create(model=model, system=system, messages=messages,
                                          tools=tools, max_tokens=2000)
        usage.add(response.usage)
        blocks = [b.model_dump() if hasattr(b, "model_dump") else dict(b) for b in response.content]
        requests = [b for b in blocks if b["type"] == "tool_use"]
        if len({b["id"] for b in requests}) != len(requests):
            raise ValueError("Duplicate tool_use IDs")
        messages.append({"role": "assistant", "content": blocks})
        if not requests:
            if getattr(response, "stop_reason", "end_turn") == "max_tokens":
                raise RuntimeError("Model output truncated")
            return LoopResult("\n".join(b["text"] for b in blocks if b["type"] == "text"), usage, calls)
        results = []
        for request in requests:
            failed = False
            try:
                if len(calls) >= max_tools:
                    raise RuntimeError("Tool budget exhausted")
                name = request["name"]
                if name not in allowed or name not in handlers:
                    raise ValueError("Tool not allowed in this stage")
                import jsonschema
                schema = next(t["input_schema"] for t in tools if t["name"] == name)
                jsonschema.validate(request["input"], schema)
                if hook:
                    hook("before_tool", request)
                output = handlers[name](**request["input"])
            except Exception as exc:
                failed, output = True, {"error": f"{type(exc).__name__}: {exc}"}
            result = {"type": "tool_result", "tool_use_id": request["id"],
                      "content": json.dumps(output, ensure_ascii=False), "is_error": failed}
            results.append(result)
            calls.append({"request": request, "result": result})
            if hook:
                hook("after_tool", calls[-1])
        messages.append({"role": "user", "content": results})
    raise RuntimeError("Model call budget exhausted")
