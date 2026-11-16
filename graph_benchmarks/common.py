from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from graph_memory.llm import AdaptiveLimiter, LLMClient, LLMConfig

THINKING = {"off": False, "on": True, "default": None}


def dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def add_llm_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default=os.environ.get("GRAPH_MEMORY_MODEL", "qwen3"))
    ap.add_argument("--judge-model", default="", help="default: --model")
    ap.add_argument("--thinking", choices=sorted(THINKING), default="off",
                    help="off sends chat_template_kwargs enable_thinking=false (Qwen3 on vLLM / SGLang); "
                         "default sends nothing (OpenAI and others)")
    ap.add_argument("--llm-concurrency", type=int, default=16)


def llm_config(args: argparse.Namespace, judge: bool = False, **kw: Any) -> LLMConfig:
    model = (args.judge_model or args.model) if judge else args.model
    base = LLMConfig(model=model, enable_thinking=THINKING[args.thinking], temperature=0.0,
                     max_concurrency=args.llm_concurrency)
    return replace(base, **kw)


def clients(args: argparse.Namespace, specs: dict[str, tuple[bool, dict]], log_dir: Path) -> dict[str, LLMClient]:
    limiter = AdaptiveLimiter(args.llm_concurrency)
    out = {}
    for name, (judge, kw) in specs.items():
        c = LLMClient(llm_config(args, judge, **kw), log_path=log_dir / f"{name}_calls.jsonl")
        c.limiter = limiter
        out[name] = c
    return out


def totals(*cs: LLMClient) -> dict[str, int]:
    keys = set().union(*(c.totals for c in cs))
    return {k: sum(c.totals.get(k, 0) for c in cs) for k in sorted(keys)}
