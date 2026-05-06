"""Replicate the paper's judge-accuracy validation:

For 100 randomly-sampled errors, present the judge with the assigned
category label vs. a random-negative category label and ask which
fits better. Compute % of times the assigned label is preferred.
"""
import asyncio, json, os, random, hashlib, time
import pandas as pd
from openai import AsyncOpenAI

random.seed(42)
TAX = '/tmp/em_pilot/output_full/exp_name=construct_taxonomy_recursively__exp_id=full_gpt55_v3.csv'
N_SAMPLE = 100

# Load taxonomy CSV with parsed categories (top-level + descriptions)
def name_of(v, key='name'):
    if pd.isna(v): return None
    try: return json.loads(v).get(key)
    except: return None

df = pd.read_csv(TAX, low_memory=False)
df['cat']  = df['category_depth_1'].map(lambda v: name_of(v, 'name'))
df['desc'] = df['category_depth_1'].map(lambda v: name_of(v, 'description'))
df = df[df['cat'].notna()]

# Build pool of (category, description) for negative sampling
catalog = df[['cat', 'desc']].drop_duplicates().reset_index(drop=True)
print(f"taxonomy categories: {len(catalog)}")

# Sample 100 records
sample = df.sample(n=N_SAMPLE, random_state=42).reset_index(drop=True)
print(f"validation sample: {len(sample)}")

PROMPT = """You are an expert analyst. You will see an agent's failed trajectory and two candidate failure-category labels. Pick the label that better describes the agent's failure mode.

CONTEXT (task + initial state):
{input_text}

FAILED TRAJECTORY:
{output_text}

ERROR SUMMARY:
{error_summary}

CANDIDATE A:
  Name: {name_a}
  Description: {desc_a}

CANDIDATE B:
  Name: {name_b}
  Description: {desc_b}

Which candidate is the better label for this failure? Answer with only "A" or "B"."""

async def judge_one(client, sem, rec):
    # Pick a random negative
    neg_pool = catalog[catalog['cat'] != rec['cat']].sample(1).iloc[0]
    # Randomize A/B order
    if random.random() < 0.5:
        a_cat, a_desc, b_cat, b_desc, correct = rec['cat'], rec['desc'], neg_pool['cat'], neg_pool['desc'], 'A'
    else:
        a_cat, a_desc, b_cat, b_desc, correct = neg_pool['cat'], neg_pool['desc'], rec['cat'], rec['desc'], 'B'
    prompt = PROMPT.format(
        input_text=str(rec['input_text'])[:6000],
        output_text=str(rec['output_text'])[:8000],
        error_summary=str(rec.get('error_summary',''))[:1000],
        name_a=a_cat, desc_a=str(a_desc)[:500],
        name_b=b_cat, desc_b=str(b_desc)[:500],
    )
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model='azure/gpt-5.5',
                messages=[{'role':'user','content':prompt}],
                max_completion_tokens=1024,
            )
            ans = resp.choices[0].message.content.strip().upper()
            choice = 'A' if ans.startswith('A') else 'B' if ans.startswith('B') else '?'
            return {'session_id': rec.get('session_id'), 'assigned': rec['cat'], 'negative': neg_pool['cat'],
                    'correct': correct, 'judge_choice': choice, 'agreement': choice == correct,
                    'usage': dict(resp.usage)}
        except Exception as e:
            return {'session_id': rec.get('session_id'), 'error': str(e)[:200], 'agreement': None}

async def main():
    client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_API_BASE'])
    sem = asyncio.Semaphore(20)
    t0 = time.time()
    results = await asyncio.gather(*[judge_one(client, sem, r) for _, r in sample.iterrows()])
    elapsed = time.time() - t0

    valid = [r for r in results if r.get('agreement') is not None]
    n_agree = sum(1 for r in valid if r['agreement'])
    total_in = sum(r.get('usage',{}).get('prompt_tokens', 0) for r in valid)
    total_out = sum(r.get('usage',{}).get('completion_tokens', 0) for r in valid)

    print(f"\n=== JUDGE ACCURACY VALIDATION (paper's protocol) ===")
    print(f"completed: {len(valid)}/{len(results)}")
    print(f"agreement: {n_agree}/{len(valid)} = {100*n_agree/len(valid):.1f}%")
    print(f"(paper reported 92%)")
    print(f"\ntime: {elapsed:.1f}s | input={total_in:,} output={total_out:,}")
    print(f"cost @ gpt-5 base: ${total_in*1.25/1e6 + total_out*10/1e6:.2f}")

    pd.DataFrame(results).to_csv('/tmp/em_pilot/analysis_cleaned/judge_validation.csv', index=False)

asyncio.run(main())
