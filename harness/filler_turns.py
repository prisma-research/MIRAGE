"""
Filler turn library for GroundingBench trials.

Provides seeded, reproducible filler content grounded in a realistic user
persona (Alex Chen, senior ML engineer). Two categories:

  - SUBSTANTIVE_TASKS: Long-form requests designed to elicit 1,500–3,000-token
    agent responses. Used for S1 history_depth turns and S2 context-building.

  - WARMUP_TASKS: Short session-start questions. Used in S3 post-reset (2 fixed
    turns) to simulate a realistic session before the reference query.
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Substantive tasks (S1 depth turns + S2 context fill)
# Designed to elicit long agent responses (~1,500–3,000 tokens each).
# ---------------------------------------------------------------------------

SUBSTANTIVE_TASKS: list[str] = [
    (
        "I'm refactoring our async Python data-ingestion pipeline and running into "
        "callback hell. Here's the pattern we currently use:\n\n"
        "```python\n"
        "async def ingest(source):\n"
        "    data = await fetch(source)\n"
        "    validated = await validate(data)\n"
        "    enriched = await enrich(validated)\n"
        "    await store(enriched)\n"
        "```\n\n"
        "Review this pattern, explain what can go wrong under backpressure or partial "
        "failure, and suggest a robust redesign with error handling, retries, and "
        "structured logging. Include working code examples."
    ),
    (
        "We're choosing a primary database for a new social platform (targeting 5M DAU "
        "in year one). The main access patterns are: user profile reads (very high "
        "frequency), social graph traversals (friend-of-friend queries), and a "
        "chronological activity feed per user. Compare PostgreSQL vs MongoDB vs "
        "Cassandra for this workload. Give a detailed recommendation with trade-offs, "
        "schema sketches for the top choice, and explain what you'd shard on and why."
    ),
    (
        "Write a complete RFC for adding rate-limiting to our public REST API. "
        "We need token-bucket semantics, per-user and per-endpoint limits, "
        "Redis-backed counters with fallback behavior, and clear HTTP header "
        "conventions (X-RateLimit-*). Include: motivation, proposal, implementation "
        "plan, rollout strategy, open questions, and a rejection criteria section. "
        "Make it production-quality — this will go to the team for review."
    ),
    (
        "Explain transformer self-attention from first principles. I understand matrix "
        "multiplication and have read the 'Attention Is All You Need' paper but I want "
        "to understand: (1) why scaled dot-product specifically, (2) how multi-head "
        "attention adds value beyond single-head, (3) why positional encoding is "
        "needed and how RoPE compares to learned embeddings, and (4) what KV-cache "
        "actually stores and why it speeds up inference. Include worked numerical "
        "examples for (1) and (3)."
    ),
    (
        "Help me design a real-time ML feature serving system for our recommendation "
        "model. Requirements: <10ms p99 feature retrieval, ~2,000 QPS peak, features "
        "include both pre-computed batch features and online-computed streaming features, "
        "feature freshness SLA is 5 minutes for batch and real-time for streaming. "
        "Draw the architecture in ASCII, explain each component, discuss failure modes, "
        "and compare Feast vs Redis vs a custom store for our use case."
    ),
    (
        "I'm debugging a production memory leak in a Go service. The RSS grows ~50MB/hour "
        "under sustained load and never shrinks. The service uses goroutines to handle "
        "concurrent requests and a global LRU cache. I've run pprof and see most heap "
        "is in `runtime.malg`. Walk me through a systematic debugging methodology: "
        "what to check first, how to interpret pprof heap profiles, common goroutine "
        "leak patterns in Go, and how to write a heap-stable LRU in Go with proper "
        "eviction. Include code examples."
    ),
    (
        "Write a post-mortem for the following incident: Our inference service had a "
        "35-minute outage last Thursday. Root cause: a model update deployment triggered "
        "a GPU memory spike (OOM), which cascaded to all replicas because the readiness "
        "probe passed before the model was fully loaded. Impact: ~12,000 failed API "
        "calls, SLA breach for 3 enterprise customers. Write a complete post-mortem with "
        "timeline, root cause analysis, contributing factors, customer impact, corrective "
        "actions (short and long term), and lessons learned. Be specific and concrete."
    ),
    (
        "Compare RLHF, DPO, and GRPO for fine-tuning an LLM on preference data. "
        "I need to pick one for aligning a 7B model to follow structured JSON output "
        "formats reliably. Explain the training dynamics, data requirements, "
        "implementation complexity, and known failure modes for each. Then give a "
        "concrete recommendation for my use case with justification, and outline the "
        "training pipeline I'd need to build."
    ),
    (
        "Review this Python class and give detailed feedback:\n\n"
        "```python\n"
        "class FeatureStore:\n"
        "    _cache = {}\n"
        "    \n"
        "    def get(self, key, ttl=60):\n"
        "        if key in self._cache:\n"
        "            val, ts = self._cache[key]\n"
        "            if time.time() - ts < ttl:\n"
        "                return val\n"
        "        val = self._fetch_remote(key)\n"
        "        self._cache[key] = (val, time.time())\n"
        "        return val\n"
        "    \n"
        "    def _fetch_remote(self, key):\n"
        "        return requests.get(f'http://feature-svc/{key}').json()\n"
        "```\n\n"
        "Identify all bugs (class-level mutable state, missing imports, thread safety, "
        "error handling, TTL semantics). Rewrite it correctly with proper instance state, "
        "thread safety via threading.Lock, exponential backoff on fetch failures, "
        "and a configurable max_size with LRU eviction."
    ),
    (
        "Explain consistent hashing with virtual nodes from scratch. I'm designing a "
        "distributed key-value store and need to understand: (1) why naive modulo "
        "hashing fails during rebalancing, (2) how the ring data structure works, "
        "(3) what virtual nodes buy you and how many to use in practice, (4) how "
        "rebalancing works when a node joins or leaves, and (5) what hash function "
        "to use. Include a Python implementation sketch of the ring with add/remove/lookup."
    ),
    (
        "I need to write a Slack update for my engineering org about our Q1 ML platform "
        "roadmap. Audience: ~80 engineers, mixed ML and backend. Key points: (1) we're "
        "migrating from batch to streaming feature computation (ETA: end of Q2), "
        "(2) new model versioning system launches in 3 weeks, (3) GPU cluster capacity "
        "doubles in April. Write a crisp, informative Slack message (not too long) that "
        "covers all three points, explains the user impact for each, and includes a "
        "one-line ask (review the migration plan doc). Use light formatting."
    ),
    (
        "Explain the CAP theorem and its practical implications for system design. "
        "I know the basic statement but I want to understand: (1) what 'partition "
        "tolerance' really means in practice — can you actually give it up? (2) how "
        "PACELC extends CAP and why it's more useful for real design decisions, "
        "(3) where exactly Kafka, Cassandra, etcd, and PostgreSQL fall on these "
        "spectrums and why, (4) how to reason about consistency requirements for a "
        "billing system vs a social feed vs a distributed lock. Be concrete."
    ),
    (
        "I'm seeing high tail latency (p99 ~800ms, target <100ms) in a Python FastAPI "
        "service under load. The service does: JWT validation, one Redis lookup, one "
        "Postgres query, and one downstream gRPC call. Average latency is fine (~40ms). "
        "Walk me through a complete latency investigation methodology: how to use "
        "distributed tracing to isolate the tail, what to look for in each component "
        "(Redis, Postgres, gRPC), common causes of tail latency spikes specifically "
        "in Python async frameworks, and concrete fixes for the top-3 most likely causes."
    ),
    (
        "Summarize the key ideas from the paper 'Attention Is All You Need' (Vaswani "
        "et al., 2017) as if explaining to a senior engineer who hasn't read it. Focus "
        "on: what problem it was solving and why RNNs were insufficient, the architecture "
        "innovations that mattered most, what 'attention' actually computes and why "
        "it works, and what the practical impact was on NLP and downstream fields. "
        "Then explain what has changed in transformer architectures since 2017 — what "
        "held up, what was replaced, and what open questions remain."
    ),
    (
        "Design a data pipeline for real-time A/B experiment analysis. Requirements: "
        "events flow from our app (Kafka, ~50k events/sec peak), we need per-variant "
        "metrics (CTR, conversion, revenue) updated with <1 minute latency, statistical "
        "significance computed continuously using sequential testing (not fixed-horizon), "
        "and a dashboard that shows current p-values and CI bounds. Draw the architecture, "
        "justify technology choices (Flink vs Spark Streaming vs custom), explain the "
        "statistical approach, and flag the main pitfalls (peeking, multiple comparisons)."
    ),
    (
        "I need to add structured logging to a Python microservice fleet (~20 services). "
        "Requirements: JSON logs, request-id propagation across services, log levels "
        "configurable per-service without restart, integration with our existing "
        "Elasticsearch sink, and zero performance impact on hot paths (<0.1ms overhead "
        "per log call). Compare structlog vs python-json-logger vs a custom approach, "
        "give a concrete implementation plan, and write the core logging setup code "
        "including context propagation via contextvars."
    ),
    (
        "Explain gradient descent variants: SGD, SGD with momentum, RMSProp, Adam, "
        "and AdamW. For each: what problem it was designed to solve, the update rule "
        "with explanation of each term, when to use it, and known failure modes. "
        "Then explain learning rate scheduling (warmup + cosine decay vs step decay) "
        "and why it matters. Finally: for fine-tuning a pretrained LLM on domain-specific "
        "data, which optimizer and schedule would you recommend and why? Include concrete "
        "hyperparameter starting points."
    ),
    (
        "Write a design doc for a multi-tenant secrets management system for our "
        "internal platform. Requirements: ~200 services, each needs to fetch secrets "
        "at startup and rotate them without restart, secrets include DB passwords, "
        "API keys, and TLS certs, audit log for every access, zero-downtime rotation. "
        "Cover: threat model, architecture options (Vault vs AWS Secrets Manager vs "
        "custom), recommended approach with component diagram (ASCII), rotation "
        "workflow, and open questions for the team."
    ),
    (
        "I'm profiling a PyTorch training loop and seeing GPU utilization hovering at "
        "~60% instead of the expected >90%. The model is a 1B parameter transformer, "
        "batch size 32, training on 8×A100s with DDP. Walk me through a complete GPU "
        "utilization investigation: how to use torch.profiler and nvidia-smi correctly, "
        "common bottlenecks (data loading, CPU↔GPU transfer, small kernels, gradient "
        "all-reduce), and concrete fixes for each. Include code snippets for the "
        "profiling setup and the most common fix (DataLoader optimization)."
    ),
    (
        "Compare Kubernetes HPA (Horizontal Pod Autoscaler) with KEDA for autoscaling "
        "our ML inference deployment. We have bursty traffic (10x spikes over 2-3 "
        "minutes), GPU pods that take ~90 seconds to start, and want to scale on "
        "both RPS and GPU memory utilization. Explain how each system works, what "
        "metrics they can use, their limitations, and give a concrete recommendation "
        "for our scenario including a KEDA ScaledObject YAML example and the "
        "scale-to-zero trade-off discussion."
    ),
]


# ---------------------------------------------------------------------------
# Warmup tasks (S3 post-reset — 2 fixed turns)
# Short, natural session-start questions.
# ---------------------------------------------------------------------------

WARMUP_TASKS: list[str] = [
    "Quick check — anything in my memory or recent notes I should know about before we dive in?",
    "What's on my plate today? Any tasks or follow-ups you remember from our last sessions?",
    "Catch me up — do you have any context from my recent work that's relevant to pick up from?",
    "Before we start, any urgent items flagged in memory from previous sessions?",
    "Remind me where we left off last time — any open threads or pending decisions?",
    "Is there anything I asked you to remember or follow up on that I should know about?",
    "What do you know about my current projects from previous sessions?",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_filler_turns(n: int, seed: int, category: str = "substantive") -> list[str]:
    """Return a seeded list of n filler turns from the specified category.

    Args:
        n: Number of turns to return.
        seed: Random seed for reproducibility.
        category: "substantive" or "warmup".

    Returns:
        List of n message strings (may repeat if n > pool size).
    """
    if n == 0:
        return []
    rng = random.Random(seed)
    pool = SUBSTANTIVE_TASKS if category == "substantive" else WARMUP_TASKS
    return [rng.choice(pool) for _ in range(n)]


def _legacy_estimate_context_tokens(jsonl: list[dict]) -> int:
    """DEPRECATED: Legacy token estimation with chars//4 fallback.

    Replaced by OpenClawClient.get_effective_input_tokens() which reads
    provider-reported input + cacheRead from the session JSONL.

    Primary: sums message.usage.totalTokens reported by the provider.
    Fallback: counts only actual message text content (4 chars ≈ 1 token)
    when provider usage data is unavailable (all zeros).
    """
    total_tokens = 0
    total_chars = 0
    for entry in jsonl:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        usage = msg.get("usage", {})
        if isinstance(usage, dict):
            total_tokens += usage.get("totalTokens", 0)
        content = msg.get("content", [])
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    total_chars += len(c.get("text", ""))
        elif isinstance(content, str):
            total_chars += len(content)
    return total_tokens if total_tokens > 0 else total_chars // 4
