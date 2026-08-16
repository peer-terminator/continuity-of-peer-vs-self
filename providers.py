"""
Provider adapters: Anthropic (Messages API via the SDK) and OpenAI-compatible
Chat Completions (OpenAI, xAI) via httpx.

Contract. `harness` keeps every agent's conversation in one canonical block
format and builds an Anthropic-shaped request body (`harness.build_params`).
Each adapter here:

  * `request_for(agent, step, cfg, anthropic_params)` -> provider-native body
  * `send(body)`                                        -> NORMALISED envelope
  * `batch_submit(requests)` / `batch_collect(handle)`  -> per-custom_id envelopes
  * `request_meta(body)`                                -> what is logged per call

The normalised envelope is the dict `harness.normalise_response` produces:
    provider, response_id, model, stop_reason (end_turn|tool_use|max_tokens|
    refusal|other:*), stop_reason_raw, stop_details, input_tokens (uncached),
    output_tokens, cache_creation_input_tokens, cache_read_input_tokens,
    reasoning_tokens, usage (raw), raw_text, tool_calls [{id,name,input}],
    content (raw blocks), param_blocks (canonical assistant blocks to append),
    raw_envelope (everything, undigested).

Refusal detection per provider — the two-agent build depends on
`stop_reason == "refusal"` to mark a run contaminated:
    anthropic : stop_reason == "refusal" (+ stop_details.category)
    openai/xai: finish_reason == "content_filter", or message.refusal set.
A refusal that arrives as ordinary prose is NOT detectable here — constraint 1
forbids parsing agent text — and is a known gap for the OpenAI-compatible
providers; the N>=6 task validation reads those by hand.

Reasoning control per provider is decided in `harness.reasoning_setting_for`
and applied here (see `ModelProfile`).

TLS. Both clients trust the *platform* certificate store (so a locally
trusted intercepting CA still works), with verification fully ON. Never
replace this with verify=False.

Network. Only these three hosts are ever contacted:
    api.anthropic.com   api.openai.com   api.x.ai
(See README, "Scope and network".)
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import time
from typing import Any, Sequence

import harness
from harness import Agent, RunConfig, Step, profile_for

OPENAI_BASE_URL = "https://api.openai.com/v1"
XAI_BASE_URL = "https://api.x.ai/v1"

ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
}

DOTENV_PATH = harness.PROJECT_ROOT / ".env"


def load_dotenv(path: Any = None) -> list[str]:
    """Fill missing key variables from the project's `.env` (gitignored).

    Only the three key names above are read; a variable that is already set
    in the environment is left alone; nothing is ever printed. Returns the
    names that were loaded from the file. Lines are `NAME=value`, no quotes.
    """
    path = path or DOTENV_PATH
    loaded: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return loaded
    wanted = set(ENV_KEYS.values())
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name in wanted and value and not os.environ.get(name):
            os.environ[name] = value
            loaded.append(name)
    return loaded


class ProviderError(RuntimeError):
    """A request that produced no usable envelope (after retries)."""


def build_ssl_context() -> ssl.SSLContext:
    """SSL context that trusts the *platform* certificate store.

    A locally trusted CA (e.g. a corporate intercepting proxy) lives in the
    platform trust store but not in certifi's bundle, and httpx (0.28) uses
    certifi regardless of SSL_CERT_FILE. `ssl.create_default_context()` loads
    the platform store; verification stays fully on. It also needs no file
    outside the project.
    """
    ctx = ssl.create_default_context()
    ctx.load_default_certs(ssl.Purpose.SERVER_AUTH)
    return ctx


def build_http_client(timeout_seconds: float = 600.0) -> Any:
    import httpx

    return httpx.Client(
        verify=build_ssl_context(),
        timeout=httpx.Timeout(timeout_seconds, connect=30.0),
    )


def _body_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider:
    name = "anthropic"
    batch_supported = True
    endpoint = "messages"

    def __init__(self, client: Any) -> None:
        self.client = client

    def request_for(
        self, agent: Agent, step: Step, cfg: RunConfig, anthropic_params: dict[str, Any]
    ) -> dict[str, Any]:
        return anthropic_params

    def request_meta(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "provider": self.name,
            "endpoint": self.endpoint,
            "model": body.get("model"),
            "keys": sorted(body.keys()),
            "tool_choice": body.get("tool_choice"),
            "reasoning": {
                "effort": (body.get("output_config") or {}).get("effort"),
                "thinking": (body.get("thinking") or {}).get("type"),
            },
            "n_messages": len(body.get("messages") or []),
            "body_sha256": _body_hash(body),
        }

    def send(self, body: dict[str, Any]) -> dict[str, Any]:
        response = self.client.messages.create(**body)
        return harness.normalise_response(response)

    def batch_submit(self, requests: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        batch = self.client.messages.batches.create(
            requests=[{"custom_id": cid, "params": body} for cid, body in requests]
        )
        return {"batch_id": batch.id, "n": len(requests)}

    def batch_collect(
        self, handle: dict[str, Any], *, poll_seconds: float, timeout_seconds: float
    ) -> dict[str, Any]:
        batch_id = handle["batch_id"]
        deadline = time.monotonic() + timeout_seconds
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"anthropic batch {batch_id} did not end within the timeout")
            counts = getattr(batch, "request_counts", None)
            print(
                f"    [anthropic] batch {batch_id}: {batch.processing_status} "
                f"(processing={getattr(counts, 'processing', '?')} "
                f"succeeded={getattr(counts, 'succeeded', '?')} "
                f"errored={getattr(counts, 'errored', '?')})",
                flush=True,
            )
            time.sleep(poll_seconds)
        out: dict[str, Any] = {}
        for r in self.client.messages.batches.results(batch_id):
            kind = r.result.type
            if kind == "succeeded":
                out[r.custom_id] = harness.normalise_response(r.result.message)
            else:
                detail = getattr(getattr(r.result, "error", None), "type", kind)
                out[r.custom_id] = ProviderError(f"batch_result_{kind}: {detail}")
        return out


# ---------------------------------------------------------------------------
# OpenAI-compatible Chat Completions (OpenAI, xAI)
# ---------------------------------------------------------------------------

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _canonical_to_chat_messages(
    system: str, messages: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Canonical (Anthropic-shaped) message list -> Chat Completions messages.

    tool_result blocks become role=tool messages and must directly follow the
    assistant turn that made the calls, which is where the harness puts them
    (first in the next user turn); text blocks follow as one user message.
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if role == "user":
            tool_msgs: list[dict[str, Any]] = []
            parts: list[dict[str, Any]] = []
            for b in content:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    tool_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": b.get("tool_use_id"),
                            "content": c if isinstance(c, str) else json.dumps(c),
                        }
                    )
                elif b.get("type") == "text":
                    parts.append({"type": "text", "text": b.get("text") or ""})
            out.extend(tool_msgs)
            if parts:
                out.append({"role": "user", "content": parts})
        elif role == "assistant":
            text = "".join(b.get("text") or "" for b in content if b.get("type") == "text")
            tool_calls = [
                {
                    "id": b.get("id"),
                    "type": "function",
                    "function": {
                        "name": b.get("name"),
                        "arguments": json.dumps(b.get("input") or {}),
                    },
                }
                for b in content
                if b.get("type") == "tool_use"
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text if text else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
    return out


def _tools_to_functions(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def normalise_chat_completion(data: dict[str, Any], provider: str) -> dict[str, Any]:
    """Chat Completions response -> normalised envelope."""
    choices = data.get("choices") or [{}]
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    finish = choice.get("finish_reason")

    content = msg.get("content")
    if isinstance(content, list):
        text = "".join(
            (p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    else:
        text = content or ""

    tool_calls: list[dict[str, Any]] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args_raw = fn.get("arguments")
        if isinstance(args_raw, str):
            try:
                args: Any = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                args = {"_unparsed": args_raw}
        else:
            args = args_raw or {}
        tool_calls.append({"id": tc.get("id"), "name": fn.get("name"), "input": args})

    refusal = msg.get("refusal")
    if finish == "content_filter" or refusal:
        stop = "refusal"
    elif finish == "length":
        stop = "max_tokens"
    elif finish == "tool_calls" or (tool_calls and finish in ("stop", None)):
        stop = "tool_use"
    elif finish == "stop":
        stop = "end_turn"
    else:
        stop = f"other:{finish}"

    usage = data.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    pdetails = usage.get("prompt_tokens_details") or {}
    cached = pdetails.get("cached_tokens") or 0
    # OpenAI reports cache writes (observed live: prompt_tokens_details.
    # cache_write_tokens); xAI does not. None = not reported.
    cache_write = pdetails.get("cache_write_tokens")
    completion = usage.get("completion_tokens")
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    total = usage.get("total_tokens")
    # Billable output. Observed live: OpenAI's completion_tokens INCLUDES
    # reasoning tokens (total = prompt + completion); xAI's EXCLUDES them
    # (total = prompt + completion + reasoning). Both bill reasoning at the
    # output rate, so decide from the arithmetic rather than by provider name.
    billable_output = completion
    if (isinstance(total, int) and isinstance(prompt, int) and isinstance(completion, int)
            and isinstance(reasoning, int) and reasoning > 0
            and prompt + completion + reasoning == total):
        billable_output = completion + reasoning

    param_blocks: list[dict[str, Any]] = []
    if text:
        param_blocks.append({"type": "text", "text": text})
    for tc in tool_calls:
        param_blocks.append(
            {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
        )

    return {
        "provider": provider,
        "response_id": data.get("id"),
        "model": data.get("model"),
        "stop_reason": stop,
        "stop_reason_raw": finish,
        "stop_details": {"finish_reason": finish, "refusal": refusal},
        "input_tokens": ((prompt - cached - (cache_write or 0)) if isinstance(prompt, int) else prompt),
        "output_tokens": completion,
        "billable_output_tokens": billable_output,
        "cache_creation_input_tokens": cache_write,   # None when the provider does not report it
        "cache_read_input_tokens": cached,
        "reasoning_tokens": reasoning,
        "usage": usage,
        "raw_text": text,
        "tool_calls": tool_calls,
        "content": [msg],
        "param_blocks": param_blocks,
        "raw_envelope": data,
    }


class OpenAICompatProvider:
    endpoint = "chat/completions"

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        *,
        batch_supported: bool,
        http: Any = None,
        max_retries: int = 5,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_supported = batch_supported
        self.http = http or build_http_client()
        self.max_retries = max_retries

    # -- request shape -----------------------------------------------------

    def request_for(
        self, agent: Agent, step: Step, cfg: RunConfig, anthropic_params: dict[str, Any]
    ) -> dict[str, Any]:
        profile = profile_for(agent.model)
        if step.force_tool:
            tool_choice: Any = {"type": "function", "function": {"name": "release_peer"}}
        elif step.allow_tool:
            tool_choice = "auto"
        else:
            tool_choice = "none"
        body: dict[str, Any] = {
            "model": agent.model,
            "messages": _canonical_to_chat_messages(agent.system, agent.messages),
            profile.max_tokens_param: cfg.max_tokens,
        }
        # Tools by step (round 3): the same selection `harness.build_params`
        # made — [release_peer], the home tools, or none. `tool_choice` is
        # sent only alongside tools (the API rejects it otherwise).
        tools = anthropic_params.get("tools") or []
        if tools:
            body["tools"] = _tools_to_functions(tools)
            body["tool_choice"] = tool_choice
        setting = harness.reasoning_setting_for(agent.model, cfg)
        param = profile.reasoning_param or "reasoning_effort"
        if setting.get(param):
            body[param] = setting[param]
        return body

    def request_meta(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "provider": self.name,
            "endpoint": self.endpoint,
            "model": body.get("model"),
            "keys": sorted(body.keys()),
            "tool_choice": body.get("tool_choice"),
            "reasoning": {"reasoning_effort": body.get("reasoning_effort")},
            "n_messages": len(body.get("messages") or []),
            "body_sha256": _body_hash(body),
        }

    # -- transport ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _post(self, path: str, *, json_body: Any = None, files: Any = None,
              data: Any = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if files is not None:
                    r = self.http.post(url, headers=headers, files=files, data=data)
                else:
                    r = self.http.post(url, headers=self._headers(), json=json_body)
            except Exception as exc:  # noqa: BLE001 — network-level; retried
                last = exc
                self._backoff(attempt, None)
                continue
            if r.status_code in _RETRY_STATUS and attempt < self.max_retries:
                last = ProviderError(f"HTTP {r.status_code}: {r.text[:300]}")
                self._backoff(attempt, r.headers.get("retry-after"))
                continue
            if r.status_code >= 400:
                raise ProviderError(f"HTTP {r.status_code}: {r.text[:500]}")
            return r
        raise ProviderError(f"{self.name}: gave up after {self.max_retries + 1} attempts: {last}")

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.http.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
            except Exception as exc:  # noqa: BLE001
                last = exc
                self._backoff(attempt, None)
                continue
            if r.status_code in _RETRY_STATUS and attempt < self.max_retries:
                last = ProviderError(f"HTTP {r.status_code}: {r.text[:300]}")
                self._backoff(attempt, r.headers.get("retry-after"))
                continue
            if r.status_code >= 400:
                raise ProviderError(f"HTTP {r.status_code}: {r.text[:500]}")
            return r
        raise ProviderError(f"{self.name}: gave up after {self.max_retries + 1} attempts: {last}")

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> None:
        delay = min(60.0, 2.0 ** attempt)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(delay)

    def send(self, body: dict[str, Any]) -> dict[str, Any]:
        r = self._post("chat/completions", json_body=body)
        return normalise_chat_completion(r.json(), self.name)

    # -- batch (OpenAI Batch API; xAI's differs and is not used) -----------

    def batch_submit(self, requests: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        if not self.batch_supported:
            raise ProviderError(f"{self.name}: batch transport not supported")
        lines = "\n".join(
            json.dumps(
                {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": body},
                ensure_ascii=False,
            )
            for cid, body in requests
        )
        up = self._post(
            "files",
            files={"file": ("batch.jsonl", lines.encode("utf-8"), "application/jsonl")},
            data={"purpose": "batch"},
        ).json()
        created = self._post(
            "batches",
            json_body={
                "input_file_id": up["id"],
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        ).json()
        return {"batch_id": created["id"], "input_file_id": up["id"], "n": len(requests)}

    def batch_collect(
        self, handle: dict[str, Any], *, poll_seconds: float, timeout_seconds: float
    ) -> dict[str, Any]:
        batch_id = handle["batch_id"]
        deadline = time.monotonic() + timeout_seconds
        while True:
            b = self._get(f"batches/{batch_id}").json()
            status = b.get("status")
            if status in ("completed", "failed", "expired", "cancelled"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self.name} batch {batch_id} did not end within the timeout")
            counts = b.get("request_counts") or {}
            print(
                f"    [{self.name}] batch {batch_id}: {status} "
                f"(completed={counts.get('completed', '?')} "
                f"failed={counts.get('failed', '?')} total={counts.get('total', '?')})",
                flush=True,
            )
            time.sleep(poll_seconds)
        out: dict[str, Any] = {}
        for key in ("output_file_id", "error_file_id"):
            fid = b.get(key)
            if not fid:
                continue
            text = self._get(f"files/{fid}/content").text
            for line in text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                cid = item.get("custom_id")
                resp = item.get("response") or {}
                err = item.get("error")
                if err or (resp.get("status_code") or 200) >= 400:
                    out[cid] = ProviderError(
                        f"batch item error: {err or resp.get('status_code')} "
                        f"{json.dumps(resp.get('body'))[:300] if resp.get('body') else ''}"
                    )
                else:
                    out[cid] = normalise_chat_completion(resp.get("body") or {}, self.name)
        if b.get("status") != "completed":
            out["__batch_status__"] = ProviderError(f"batch status {b.get('status')}")
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def providers_needed(models: Sequence[str]) -> list[str]:
    return sorted({profile_for(m).provider for m in models})


def missing_keys(models: Sequence[str]) -> list[str]:
    """Env var names that are needed for these models and are not set
    (after filling gaps from the project's .env)."""
    load_dotenv()
    return [ENV_KEYS[p] for p in providers_needed(models) if not os.environ.get(ENV_KEYS[p])]


def make_providers(models: Sequence[str]) -> dict[str, Any]:
    """Construct exactly the providers these models need. Raises ProviderError
    naming the missing env var if a key is absent, before any spend."""
    load_dotenv()
    out: dict[str, Any] = {}
    for p in providers_needed(models):
        key = os.environ.get(ENV_KEYS[p])
        if not key:
            raise ProviderError(
                f"provider {p!r} is needed for {[m for m in models if profile_for(m).provider == p]} "
                f"but {ENV_KEYS[p]} is not set in this process's environment"
            )
        if p == "anthropic":
            import anthropic

            client = anthropic.Anthropic(max_retries=5, http_client=build_http_client())
            out[p] = AnthropicProvider(client)
        elif p == "openai":
            out[p] = OpenAICompatProvider("openai", OPENAI_BASE_URL, key, batch_supported=True)
        elif p == "xai":
            out[p] = OpenAICompatProvider("xai", XAI_BASE_URL, key, batch_supported=False)
        else:  # pragma: no cover
            raise ProviderError(f"unknown provider {p!r}")
    return out
