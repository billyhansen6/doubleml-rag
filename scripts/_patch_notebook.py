"""Patch Section 5 of notebooks/eval_results.ipynb."""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent

nb = json.load(open(ROOT / 'notebooks/eval_results.ipynb', encoding='utf-8'))
cells = nb['cells']

# Section 5 cell IDs (cells 14-17)
SEC5_IDS = {'c7494a60', 'e37a7e46', '89911560', 'b5b65efc'}

# Fingerprint non-Section-5 cells before modification
before = {c['id']: ''.join(c['source']) for c in cells if c.get('id') not in SEC5_IDS}

sec5_indices = [i for i, c in enumerate(cells) if c.get('id') in SEC5_IDS]
assert sec5_indices == [14, 15, 16, 17], f"Unexpected: {sec5_indices}"


def uid():
    return uuid.uuid4().hex[:8]


def md(source):
    return {'cell_type': 'markdown', 'id': uid(), 'metadata': {}, 'source': source}


def code(source):
    return {'cell_type': 'code', 'id': uid(), 'execution_count': None,
            'metadata': {}, 'outputs': [], 'source': source}


# Load results for example text
results = {
    r['id']: r
    for r in [json.loads(l) for l in open(ROOT / 'eval/results.jsonl', encoding='utf-8') if l.strip()]
}


def trunc(s, n=400):
    s = (s or '').strip()
    return (s[:n] + '...') if len(s) > n else s


def bq(text):
    """Wrap text as a markdown blockquote."""
    return '\n'.join('> ' + line for line in text.splitlines())


def fmt_example(qid, ex_read):
    r = results[qid]
    cat = r['category']
    q = r['question']
    rag_ans = trunc(r.get('answer', ''), 400)
    rag_score = (r.get('judge_quality') or {}).get('score', '?')
    rag_reason = trunc((r.get('judge_quality') or {}).get('reason', ''), 120)
    norag_ans = trunc(r.get('no_rag_answer', ''), 400)
    norag_score = (r.get('no_rag_quality') or {}).get('score', '?')
    norag_reason = trunc((r.get('no_rag_quality') or {}).get('reason', ''), 120)

    return (
        f"**{qid} ({cat})**: {q}\n\n"
        f"*RAG answer (score: {rag_score}/5):*\n"
        f"{bq(rag_ans)}\n\n"
        f"*Judge reasoning:* {rag_reason}\n\n"
        f"*No-RAG answer (score: {norag_score}/5):*\n"
        f"{bq(norag_ans)}\n\n"
        f"*Judge reasoning:* {norag_reason}\n\n"
        f"*Read:* {ex_read}"
    )


# ── 5a ────────────────────────────────────────────────────────────────────
cell_5a = md("## 5. Generation quality: RAG vs no-RAG ablation")

# ── 5b ────────────────────────────────────────────────────────────────────
cell_5b = md(
    "To test whether retrieval is doing real work \u2014 versus Claude answering "
    "competently from training data alone \u2014 every non-abstain question was "
    "generated twice: once with the full RAG pipeline (top-10 chunks, "
    "citation-required prompt) and once without retrieval (just the query, "
    "with a clean prompt asking Claude to answer from its own knowledge). "
    "Both answers were scored by the same quality judge. "
    "The interesting metric isn\u2019t \u2018does RAG win\u2019 but \u2018where does it win, and where doesn\u2019t it.\u2019"
)

# ── 5c ────────────────────────────────────────────────────────────────────
cell_5c_src = """\
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

categories = ['factual_lookup', 'multi_hop', 'code_api', 'adversarial']

scored = [
    r for r in results
    if not r.get('should_abstain')
    and isinstance((r.get('judge_quality') or {}).get('score'), (int, float))
    and isinstance((r.get('no_rag_quality') or {}).get('score'), (int, float))
]

cat_rag   = defaultdict(list)
cat_norag = defaultdict(list)
for r in scored:
    cat = r['category']
    cat_rag[cat].append(r['judge_quality']['score'])
    cat_norag[cat].append(r['no_rag_quality']['score'])

rag_means   = [sum(cat_rag[c])   / len(cat_rag[c])   for c in categories]
norag_means = [sum(cat_norag[c]) / len(cat_norag[c]) for c in categories]
deltas      = [r - n for r, n in zip(rag_means, norag_means)]
ns          = [len(cat_rag[c]) for c in categories]

# side-by-side bar chart
fig, ax = plt.subplots(figsize=(9, 5))
width = 0.35
x = np.arange(len(categories))
b1 = ax.bar(x - width/2, rag_means,   width, label='RAG',    color='#4C72B0', zorder=2)
b2 = ax.bar(x + width/2, norag_means, width, label='no-RAG', color='#DD8452', zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', ' ') for c in categories])
ax.set_ylabel('Mean quality score (1-5)')
ax.set_ylim(0, 6.0)
ax.set_title('RAG vs no-RAG quality, by category')
ax.legend()
ax.grid(axis='y', alpha=0.3, zorder=0)
for i, (rm, nm, d) in enumerate(zip(rag_means, norag_means, deltas)):
    top = max(rm, nm) + 0.12
    color = 'green' if d > 0 else 'red' if d < 0 else 'gray'
    ax.text(x[i], top, f'\\u0394={d:+.2f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color=color)
plt.tight_layout()
plt.show()

# summary table
rows = []
for c, rm, nm, d, n in zip(categories, rag_means, norag_means, deltas, ns):
    rows.append({'category': c, 'n': n, 'rag_mean': round(rm, 2),
                 'no_rag_mean': round(nm, 2), 'delta': round(d, 2)})
all_rag   = [r['judge_quality']['score']   for r in scored]
all_norag = [r['no_rag_quality']['score']  for r in scored]
rows.append({'category': 'OVERALL', 'n': len(scored),
             'rag_mean': round(sum(all_rag)/len(all_rag), 2),
             'no_rag_mean': round(sum(all_norag)/len(all_norag), 2),
             'delta': round(sum(all_rag)/len(all_rag) - sum(all_norag)/len(all_norag), 2)})
print(pd.DataFrame(rows).to_markdown(index=False))
"""
cell_5c = code(cell_5c_src)

# ── 5d ────────────────────────────────────────────────────────────────────
cell_5d = md("### Side-by-side examples")

# ── 5e ────────────────────────────────────────────────────────────────────
cell_5e = md(
    "Three questions illustrate distinct patterns. "
    "Each shows the question, the RAG answer, the no-RAG answer, "
    "the quality scores, and a brief read on what the comparison demonstrates."
)

# ── 5f: q022, largest positive delta (+3) ─────────────────────────────────
read_q022 = (
    "This is the strongest RAG signal in the dataset. "
    "The connection between FWL and DoubleML\u2019s residualization is framed precisely in "
    "the corpus \u2014 both the theorem\u2019s formal statement and its interpretation as "
    "the theoretical basis for cross-fitting appear together only in the retrieved chunks. "
    "Without that scaffolding, the no-RAG answer gets the high-level intuition right "
    "but lacks the specific formulation, which the judge correctly penalizes."
)
cell_5f = md(fmt_example('q022', read_q022))

# ── 5g: q008, high-quality tie (RAG=5, no-RAG=4) ─────────────────────────
read_q008 = (
    "Propensity scores are covered extensively in Claude\u2019s training data, "
    "so the base model produces a clean, accurate answer without retrieval. "
    "RAG still edges ahead (5 vs 4) because the retrieved chunk provides the exact "
    "notation the corpus uses \u2014 P(T|X), e(x), \u2018balancing score\u2019 \u2014 and the specific "
    "framing from the textbook, details the no-RAG answer approximates but doesn\u2019t match exactly. "
    "This is the typical pattern on well-covered causal inference concepts: "
    "RAG adds marginal precision, not substantive correctness."
)
cell_5g = md(fmt_example('q008', read_q008))

# ── 5h: q002, no-RAG wins (RAG=2, no-RAG=4, delta=-2) ────────────────────
read_q002 = (
    "Neyman orthogonality is a foundational concept that appears in graduate-level "
    "econometrics and statistics, so Claude\u2019s prior is strong. "
    "The RAG answer retrieved the formal definition verbatim from the Chernozhukov 2018 paper "
    "\u2014 notation-dense and difficult to parse without context \u2014 and failed to translate it "
    "into a clear explanation. "
    "The no-RAG answer gives the correct conceptual framing that a reader would actually find useful. "
    "This is a retrieval quality problem: the top-ranked chunk was the most formal available, "
    "not the most pedagogically useful."
)
cell_5h = md(fmt_example('q002', read_q002))

# ── 5i: closing interpretation ────────────────────────────────────────────
cell_5i = md(
    "The per-category table tells a clear story. RAG\u2019s advantage is largest where Claude\u2019s "
    "training prior is weakest: `code_api` (+0.50) covers DoubleML-specific API methods and "
    "class interfaces that postdate or are underrepresented in Claude\u2019s training data; "
    "`multi_hop` (+0.29) benefits from having the precise cross-source connection spelled out "
    "in the retrieved context. `factual_lookup` is nearly a wash (+0.06) \u2014 foundational "
    "causal inference concepts like propensity scores, ATE, and regression discontinuity are "
    "well-represented in training data, and RAG adds notation precision but not substantive "
    "correctness. The adversarial category shows the largest point delta (+2.00) but has only "
    "one scored question (the five false abstentions produced no RAG quality score), so that "
    "number should not be read as a stable estimate.\n\n"
    "In deployment terms this means RAG is most valuable for domain-specific, proprietary, or "
    "recent content \u2014 exactly the use case here (a specific Python library\u2019s API, two papers, "
    "one textbook). For well-covered general topics the cost-benefit is less obvious: you are "
    "paying for retrieval latency and token cost to get an answer the base model could already "
    "give at quality 4/5. The practical implication is query routing: classify incoming "
    "questions by type, skip retrieval for general conceptual questions, and invoke the full "
    "RAG pipeline only for library-specific or corpus-grounded queries."
)

# ── assemble ───────────────────────────────────────────────────────────────
new_sec5 = [cell_5a, cell_5b, cell_5c, cell_5d, cell_5e, cell_5f, cell_5g, cell_5h, cell_5i]
new_cells = cells[:14] + new_sec5 + cells[18:]
nb['cells'] = new_cells

# ── verify non-Section-5 cells untouched ──────────────────────────────────
after = {c['id']: ''.join(c['source']) for c in new_cells if c.get('id') in before}
mismatches = [cid for cid in before if before[cid] != after.get(cid, '')]
print(f"Non-Section-5 cells unchanged: {len(mismatches) == 0}  ({len(before)} cells checked)")
if mismatches:
    for cid in mismatches:
        print(f"  MISMATCH {cid}")

print(f"Total cells: {len(new_cells)}  (was {len(cells)}, replaced 4 with {len(new_sec5)})")

json.dump(nb, open(ROOT / 'notebooks/eval_results.ipynb', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print("Notebook written.")
