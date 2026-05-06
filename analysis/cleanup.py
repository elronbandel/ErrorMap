"""Manual cleanup pass over v3 taxonomy.

Produces a 'cleaned' category column per record and re-runs the same
analysis (prevalence + cross-tabs + per-axis breakdowns).

Mapping rules:
1. If category_depth_1 != 'Other', use it directly (with optional rename).
2. If category_depth_1 == 'Other', promote category_depth_2 (with synonym
   merging into an existing top-level when applicable).
3. Records where both depth_1 and depth_2 are 'Other' stay as 'Other'.
"""
import json, os, hashlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT = '/tmp/em_pilot/analysis_cleaned'
os.makedirs(OUT, exist_ok=True)

TAX = '/tmp/em_pilot/output_full/exp_name=construct_taxonomy_recursively__exp_id=full_gpt55_v3.csv'
SE  = '/tmp/em_pilot/output_full/exp_name=single_error__exp_id=full_gpt55_v3.csv'

tax = pd.read_csv(TAX, low_memory=False)
se  = pd.read_csv(SE,  low_memory=False)

def name_of(v):
    if pd.isna(v): return None
    try: return json.loads(v)['name']
    except: return None

tax['d1'] = tax['category_depth_1'].map(name_of)
tax['d2'] = tax['category_depth_2'].map(name_of)
tax['d3'] = tax['category_depth_3'].map(name_of)

# --- merge mapping for synonyms / promotions ---
# applied in order; a name absent from this dict keeps its identity
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

# top-level overlaps to merge
TOPLEVEL_MERGE = {
    'Search Adaptation Failure':         'Search Recovery & Adaptation',
    'Search Recovery Failure':           'Search Recovery & Adaptation',
    'Escalation Timing Error':           'Escalation Handling',
    'Escalation Protocol Error':         'Escalation Handling',
}

def canonical(d1, d2, d3):
    """Pick the canonical category for a record."""
    if d1 != 'Other' and d1 is not None:
        return TOPLEVEL_MERGE.get(d1, d1)
    if d2 and d2 != 'Other':
        return SYNONYM_MERGE.get(d2, d2)
    if d3 and d3 != 'Other':
        return SYNONYM_MERGE.get(d3, d3)
    return 'Other'

tax['category'] = [canonical(d1, d2, d3) for d1, d2, d3 in zip(tax['d1'], tax['d2'], tax['d3'])]

# Join agent + session_id from single_error via (example_id, model, output_text-hash)
def hh(s): return hashlib.md5(str(s).encode()).hexdigest()[:12]
se_failures = se[se['score'] == 0].copy()
se_failures['oth'] = se_failures['output_text'].map(hh)
tax['oth'] = tax['output_text'].map(hh)
joined = tax.merge(
    se_failures[['example_id', 'model', 'oth', 'agent', 'session_id']],
    on=['example_id', 'model', 'oth'], how='left'
)

print(f"records: {len(joined)} | with agent: {joined['agent'].notna().sum()}")

# Prevalence
prev = joined['category'].value_counts().reset_index()
prev.columns = ['category', 'count']
prev['pct'] = 100 * prev['count'] / len(joined)
prev.to_csv(f'{OUT}/prevalence.csv', index=False)
print(f"\nunique categories after cleanup: {len(prev)}")
print(f"\n=== TOP CATEGORIES (cleaned) ===")
for _, r in prev.iterrows():
    print(f"  {r['category']:<40} {r['count']:>4}  ({r['pct']:.1f}%)")

# Plot prevalence
fig, ax = plt.subplots(figsize=(12, max(6, len(prev)*0.35)))
ax.barh(prev['category'][::-1], prev['count'][::-1])
for i, (c, p) in enumerate(zip(prev['count'][::-1], prev['pct'][::-1])):
    ax.text(c+5, i, f' {c} ({p:.1f}%)', va='center', fontsize=8)
ax.set_xlabel('Records')
ax.set_title(f'Cleaned failure category prevalence (n={len(joined)})')
plt.tight_layout()
plt.savefig(f'{OUT}/prevalence.png', dpi=120)
plt.close()

# Cross-tabs
def make_heatmap(group_col, title, out_name):
    sub = joined[joined[group_col].notna()]
    ct = pd.crosstab(sub['category'], sub[group_col])
    ct.to_csv(f'{OUT}/crosstab_{out_name}.csv')
    pct = ct.div(ct.sum(axis=0), axis=1) * 100
    pct.to_csv(f'{OUT}/crosstab_{out_name}_pct.csv')
    fig, ax = plt.subplots(figsize=(max(8, len(ct.columns)*1.2), max(8, len(ct.index)*0.35)))
    sns.heatmap(pct.loc[prev['category'].tolist()], annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': '% of column total'}, ax=ax)
    ax.set_title(f'{title}  (column-normalized %)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/crosstab_{out_name}.png', dpi=120)
    plt.close()
    return ct

make_heatmap('model', 'Failure category × model', 'model')
make_heatmap('dataset', 'Failure category × benchmark', 'benchmark')
make_heatmap('agent', 'Failure category × agent', 'agent')

# Per-axis breakdowns
def per_axis(group_col, name, top_k=8):
    sub = joined[joined[group_col].notna()]
    rows = []
    for g, gdf in sub.groupby(group_col):
        cnt = gdf['category'].value_counts()
        for cat, n in cnt.head(top_k).items():
            rows.append({group_col: g, 'category': cat, 'count': n,
                         'pct': 100*n/len(gdf)})
    return pd.DataFrame(rows)

for col, nm in [('dataset','benchmark'), ('agent','agent'), ('model','model')]:
    df_b = per_axis(col, nm)
    df_b.to_csv(f'{OUT}/breakdown_{nm}.csv', index=False)

print(f"\nWrote {OUT}/")
print(f"\nOther coverage: {prev[prev['category']=='Other']['pct'].iloc[0]:.1f}%")
