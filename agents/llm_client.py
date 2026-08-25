"""
agents/llm_client.py
====================
OpenAI GPT-4o-mini client with native structured outputs.

Routes calls to the correct model tier:
  - Tier 1 (analyst_model)   : data summarization (Technical, Macro, Sentiment, Calendar)
  - Tier 2 (reasoning_model) : debate + risk assessment
  - Tier 3 (decision_model)  : trader proposal + fund manager final decision

Key features:
  - Structured output via client.beta.chat.completions.parse() — handles $defs/$ref
    transparently, enforces ALL required fields, returns Pydantic object directly.
  - Plain-text calls via client.chat.completions.create() for debate dialogue.
  - Event-loop-aware client caching — resets when asyncio event loop changes.
  - Token usage + latency tracking returned with every call.
  - Exponential backoff retry on transient errors (429, 503, RateLimitError).

Usage:
    from agents.llm_client import llm
    report, meta = await llm.call_analyst("TechnicalAnalyst", sys, usr, TechnicalAnalystReport)
    text,   meta = await llm.call_reasoning_text("BullResearcher", sys, usr)
    decision, meta = await llm.call_decision("FundManager", sys, usr, FundManagerDecision)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Type, TypeVar

from openai import AsyncOpenAI, RateLimitError, APIStatusError
from pydantic import BaseModel

from agents.config import cfg
from agents.schemas import AgentCallMetadata

log = logging.getLogger("agents.llm")

T = TypeVar("T", bound=BaseModel)

# ── Retry config ──────────────────────────────────────────────────────────────
_MAX_RETRIES   = 3
_RETRY_BACKOFF = [2.0, 5.0, 10.0]   # seconds between retries


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in (429, 503):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "503", "quota", "resource exhausted", "overloaded", "rate limit"))


@dataclass
class LLMCallResult:
    """Result wrapper for any LLM call."""
    content: Any               # Pydantic model instance OR str (for NL calls)
    metadata: AgentCallMetadata


class OpenAIClient:
    """
    Async GPT-4o-mini client using the official OpenAI API.

    Structured output strategy:
      - Uses client.beta.chat.completions.parse() with a Pydantic model class.
        The SDK handles JSON schema generation, $defs/$ref resolution, and strict
        enforcement of ALL required fields automatically.
      - Returns .message.parsed which is already the Pydantic model instance.

    Event loop safety:
      - The AsyncOpenAI instance is cached per event loop ID.
        When asyncio.run() creates a new event loop (e.g. in tests), a fresh
        client is automatically created so httpx connections don't carry over
        from a closed loop.
    """

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._loop_id: int | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-init the async OpenAI client, resetting if the event loop changed."""
        try:
            current_loop = asyncio.get_event_loop()
            current_id = id(current_loop)
        except RuntimeError:
            current_id = None

        if self._client is None or self._loop_id != current_id:
            cfg.validate()
            self._client = AsyncOpenAI(
                api_key=cfg.openai_api_key,
                base_url=cfg.openai_base_url,
                timeout=60.0,
                max_retries=0,   # We handle retries ourselves
            )
            self._loop_id = current_id
        return self._client

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _call_structured(
        self,
        agent_name: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
        temperature: float,
    ) -> LLMCallResult:
        """
        Structured call using client.beta.chat.completions.parse().
        The SDK enforces ALL required fields in the Pydantic schema.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(_MAX_RETRIES):
            t0 = time.monotonic()
            try:
                client = self._get_client()
                completion = await client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=schema,      # Pass Pydantic class directly
                    temperature=temperature,
                    max_tokens=2048,
                )
                latency_ms = int((time.monotonic() - t0) * 1000)

                usage = completion.usage
                input_tokens  = getattr(usage, "prompt_tokens",     0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0

                # .message.parsed is already the Pydantic model instance
                content = completion.choices[0].message.parsed
                if content is None:
                    # Fallback: parse from raw text if .parsed is None
                    raw = completion.choices[0].message.content or ""
                    content = schema.model_validate_json(raw)

                meta = AgentCallMetadata(
                    agent_name=agent_name,
                    model_used=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cost_usd_estimate=cfg.estimate_cost(model, input_tokens, output_tokens),
                )
                log.info(
                    "[%s] %s | %d+%d tokens | %dms | $%.5f",
                    agent_name, model, input_tokens, output_tokens,
                    latency_ms, meta.cost_usd_estimate,
                )
                return LLMCallResult(content=content, metadata=meta)

            except Exception as exc:
                last_exc = exc
                latency_ms = int((time.monotonic() - t0) * 1000)
                log.warning(
                    "[%s] attempt %d/%d failed (%dms): %s",
                    agent_name, attempt + 1, _MAX_RETRIES, latency_ms, exc
                )
                if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                    await asyncio.sleep(_RETRY_BACKOFF[attempt])
                else:
                    break

        raise RuntimeError(
            f"[{agent_name}] All {_MAX_RETRIES} attempts failed. Last: {last_exc}"
        ) from last_exc

    async def _call_text(
        self,
        agent_name: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> LLMCallResult:
        """Plain-text call (no structured output). Used for debate dialogue."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(_MAX_RETRIES):
            t0 = time.monotonic()
            try:
                client = self._get_client()
                completion = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2048,
                    stream=False,
                )
                latency_ms = int((time.monotonic() - t0) * 1000)

                usage = completion.usage
                input_tokens  = getattr(usage, "prompt_tokens",     0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                content = completion.choices[0].message.content or ""

                meta = AgentCallMetadata(
                    agent_name=agent_name,
                    model_used=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    cost_usd_estimate=cfg.estimate_cost(model, input_tokens, output_tokens),
                )
                log.info(
                    "[%s] %s | %d+%d tokens | %dms | $%.5f",
                    agent_name, model, input_tokens, output_tokens,
                    latency_ms, meta.cost_usd_estimate,
                )
                return LLMCallResult(content=content, metadata=meta)

            except Exception as exc:
                last_exc = exc
                latency_ms = int((time.monotonic() - t0) * 1000)
                log.warning(
                    "[%s] attempt %d/%d failed (%dms): %s",
                    agent_name, attempt + 1, _MAX_RETRIES, latency_ms, exc
                )
                if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                    await asyncio.sleep(_RETRY_BACKOFF[attempt])
                else:
                    break

        raise RuntimeError(
            f"[{agent_name}] All {_MAX_RETRIES} attempts failed. Last: {last_exc}"
        ) from last_exc

    # ── Public API ────────────────────────────────────────────────────────────

    async def call_analyst(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
    ) -> tuple[T, AgentCallMetadata]:
        """
        Tier 1: Structured output call for analyst agents.
        Used by: Technical, Macro, Sentiment, Calendar Analysts.
        """
        result = await self._call_structured(
            agent_name=agent_name,
            model=cfg.analyst_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            temperature=0.2,
        )
        return result.content, result.metadata

    async def call_reasoning_structured(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
    ) -> tuple[T, AgentCallMetadata]:
        """
        Tier 2: Structured output call for reasoning agents.
        Used by: Debate Facilitator (DebateSummary), Risk Team (RiskAssessment).
        """
        result = await self._call_structured(
            agent_name=agent_name,
            model=cfg.reasoning_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            temperature=0.3,
        )
        return result.content, result.metadata

    async def call_reasoning_text(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, AgentCallMetadata]:
        """
        Tier 2: Plain-text call for reasoning agents.
        Used by: Bull/Bear Researchers (debate arguments — natural language).
        Higher temperature for more varied, argumentative responses.
        """
        result = await self._call_text(
            agent_name=agent_name,
            model=cfg.reasoning_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,
        )
        return result.content, result.metadata

    async def call_decision(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
    ) -> tuple[T, AgentCallMetadata]:
        """
        Tier 3: Structured output call for decision agents.
        Used by: Trader Agent, Fund Manager.
        Near-deterministic temperature for consistent final decisions.
        """
        result = await self._call_structured(
            agent_name=agent_name,
            model=cfg.decision_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            temperature=0.1,
        )
        return result.content, result.metadata


# ── Singleton ─────────────────────────────────────────────────────────────────
llm = OpenAIClient()
