"""Full run: ErrorMap Stage 1 + Stage 2 on all 4,551 records.

Runs from the same CWD as the pilot so LiteLLM disk cache reuses
the 100 already-processed records (zero API spend on those).
"""
import asyncio, os, sys
sys.path.insert(0, '/tmp/ErrorMap-fork/src')
from error_map import ErrorMap

LITELLM_CONFIG = {
    "model": "openai/azure/gpt-5.5",
    "api_base": os.environ["OPENAI_API_BASE"],
    "api_key": os.environ["OPENAI_API_KEY"],
    "max_completion_tokens": 8192,
    "reasoning_effort": "high",
}

DATASETS = ["appworld_test_normal", "browsecompplus", "swebench",
            "tau2_airline", "tau2_retail", "tau2_telecom"]

async def main():
    em = ErrorMap(
        inference_type="litellm",
        litellm_config=LITELLM_CONFIG,
        datasets=DATASETS,
        data_path="/tmp/em_pilot/data_full",
        output_dir="/tmp/em_pilot/output_full",
        exp_id="full_gpt55_v4",
        ratio=1.0,
        max_workers=100,
        use_correct_predictions=True,
    )
    result = await em.run()
    print(f"\nDONE: {result}")

asyncio.run(main())
