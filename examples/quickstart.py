import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_memory import ChatMemory, LLMClient, LLMConfig

MODEL = os.environ.get("GRAPH_MEMORY_MODEL", "qwen3")
THINKING = False if MODEL.lower().startswith("qwen") else None

SESSIONS = [
    ("2024/03/01 (Fri) 10:00", [
        {"role": "user", "content": "I take Quelmithin for my lactose intolerance. If my health condition changes, "
                                    "my medication will be a multivitamin."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "My pull requests are reviewed by Dana. The reviewer is assigned by my team lead, "
                                    "Priya, so if the team lead changes, the reviewer would likely change too."},
        {"role": "assistant", "content": "Got it."}]),
    ("2024/03/09 (Sat) 09:30", [
        {"role": "user", "content": "Update: my doctor now says I have high blood pressure. Also, Marco replaced "
                                    "Priya as my team lead."},
        {"role": "assistant", "content": "Thanks for the update."}]),
]
QUESTIONS = ["What medication am I taking?", "Who is my code reviewer now?"]
ANSWER = ("Answer the user's question based ONLY on the memory below. If the memory says a value is unknown now, "
          "say so.\n\nMemory:\n{memory}\n\nQuestion: {question}")


async def main():
    cfg = LLMConfig(model=MODEL, enable_thinking=THINKING, temperature=0.2, max_tokens=8192)
    async with LLMClient(cfg) as llm:
        mem = ChatMemory(llm, cache=".graph_memory_cache")
        for when, turns in SESSIONS:
            await mem.add_session(turns, when)
        mem.save("memory.json")
        for q in QUESTIONS:
            memory = mem.recall(q, budget=1500)
            res = await llm.chat([{"role": "user", "content": ANSWER.format(memory=memory, question=q)}],
                                 temperature=0.0, max_tokens=300)
            print(f"Q: {q}\nMemory:\n{memory}\nA: {res.content.strip()}\n")


if __name__ == "__main__":
    asyncio.run(main())
