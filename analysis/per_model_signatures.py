"""Render per-model and per-agent failure signature plots.

Paper-style: each model/agent gets its own bar chart showing its
distinctive failure category distribution (top 15).
"""
import json, os, hashlib
import pandas as pd
import matplotlib.pyplot as plt

OUT = '/tmp/em_pilot/analysis_cleaned'

# Re-derive the joined cleaned data (same logic as cleanup_v3.py)
TAX = '/tmp/em_pilot/output_full/exp_name=construct_taxonomy_recursively__exp_id=full_gpt55_v3.csv'
SE  = '/tmp/em_pilot/output_full/exp_name=single_error__exp_id=full_gpt55_v3.csv'

SYNONYM_MERGE = {
    'Information Gathering Error':       'Evidence Retrieval Omission',
    'Factuality Error':                  'Unsupported Fabrication',
    'Reasoning Calculation Error':       'Quantitative Reasoning Error',
    'Action Targeting Error':            'Entity Resolution Error',
    'Error Recovery Error':              'Validation Recovery Failure',
    'Verification Error':                'Validation Recovery Failure',
    'Tool Result Interpretation Error':  'Evidence Interpretation Error',
    'Constraint Handling Error':         'Constraint Interpretation Error',
    'Tool Selection Error':              'Tool Invocation Error',
    'Output Formatting Error':           'Tool Invocation Error',
    'Clarification Seeking Error':       'User Communication Error',
    'State Tracking Error':              'State/Goal Tracking Error',
    'Context Retention Error':           'State/Goal Tracking Error',
    'Goal Tracking Error':               'State/Goal Tracking Error',
}
TOPLEVEL_MERGE = {
    'Search Adaptation Failure':         'Search Recovery & Adaptation',
    'Search Recovery Failure':           'Search Recovery & Adaptation',
    'Escalation Timing Error':           'Escalation Handling',
    'Escalation Protocol Error':         'Escalation Handling',
}

def name_of(v):
    if pd.isna(v): return None
    try: return json.loads(v)['name']
    except: return None

def canonical(d1, d2, d3):
    if d1 != 'Other' and d1 is not None:
        return TOPLEVEL_MERGE.get(d1, d1)
    if d2 and d2 != 'Other':
        return SYNONYM_MERGE.get(d2, d2)
    if d3 and d3 != 'Other':
        return SYNONYM_MERGE.get(d3, d3)
    return 'Other'

tax = pd.read_csv(TAX, low_memory=False)
se  = pd.read_csv(SE,  low_memory=False)
tax['d1'] = tax['category_depth_1'].map(name_of)
tax['d2'] = tax['category_depth_2'].map(name_of)
tax['d3'] = tax['category_depth_3'].map(name_of)
tax['category'] = [canonical(a,b,c) for a,b,c in zip(tax['d1'], tax['d2'], tax['d3'])]

def hh(s): return hashlib.md5(str(s).encode()).hexdigest()[:12]
se_f = se[se['score']==0].copy()
se_f['oth'] = se_f['output_text'].map(hh)
tax['oth'] = tax['output_text'].map(hh)
joined = tax.merge(se_f[['example_id','model','oth','agent']], on=['example_id','model','oth'], how='left')

# Overall ranking for consistent ordering
overall = joined['category'].value_counts().head(15).index.tolist()

def plot_signature(group_col, val, sub, ax, color):
    cnt = sub['category'].value_counts()
    pct = (cnt / len(sub) * 100).reindex(overall, fill_value=0)
    ax.barh(pct.index[::-1], pct.values[::-1], color=color)
    ax.set_xlim(0, 12)
    ax.set_title(f'{val}\n(n={len(sub)})', fontsize=10)
    for i, v in enumerate(pct.values[::-1]):
        if v > 0: ax.text(v+0.1, i, f'{v:.1f}%', va='center', fontsize=7)

# Per-model panel (5 models)
models = sorted(joined['model'].dropna().unique())
fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 6), sharey=True)
for ax, m in zip(axes, models):
    plot_signature('model', m.replace('openai_','').replace('-2025-12-11',''),
                   joined[joined['model']==m], ax, 'steelblue')
fig.suptitle('Per-model failure signatures (% of model\'s errors per category)', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/per_model_signatures.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"wrote {OUT}/per_model_signatures.png")

# Per-agent panel (5 agents)
agents = sorted(joined['agent'].dropna().unique())
fig, axes = plt.subplots(1, len(agents), figsize=(5*len(agents), 6), sharey=True)
for ax, a in zip(axes, agents):
    plot_signature('agent', a, joined[joined['agent']==a], ax, 'darkorange')
fig.suptitle('Per-agent failure signatures (% of agent\'s errors per category)', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/per_agent_signatures.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"wrote {OUT}/per_agent_signatures.png")

# Per-benchmark panel (6 benchmarks)
benches = sorted(joined['dataset'].dropna().unique())
fig, axes = plt.subplots(1, len(benches), figsize=(5*len(benches), 6), sharey=True)
for ax, b in zip(axes, benches):
    plot_signature('dataset', b, joined[joined['dataset']==b], ax, 'seagreen')
fig.suptitle('Per-benchmark failure signatures', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/per_benchmark_signatures.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"wrote {OUT}/per_benchmark_signatures.png")
