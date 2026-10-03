"""Run with the API stopped: python -m wafer_llm_query.build_knowledge."""
import asyncio
import json
from .knowledge_store import build

if __name__ == "__main__":
    print(json.dumps(asyncio.run(build()), ensure_ascii=False, indent=2))
