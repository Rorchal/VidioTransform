# ctxgc offline eval — 30 synthetic cases, 7 question categories

Budget = fraction of the full-context token count. Cells = fraction of questions whose answer string survives in the compressed context (mean over cases and questions).

## Answer retention (all questions)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.43 | 0.61 | 0.86 | 0.90 |
| random | 0.57 | 0.79 | 0.87 | 0.85 |
| uniform | 0.49 | 0.57 | 0.85 | 0.88 |
| gc_binary | 0.43 | 0.61 | 0.86 | 0.90 |
| graded | 0.37 | 0.80 | 0.98 | 0.98 |
| graded+oracle | 0.40 | 0.79 | 0.98 | 0.98 |
| graded-facts | 0.51 | 0.81 | 0.98 | 0.98 |
| graded-tomb | 0.42 | 0.80 | 0.98 | 0.98 |
| graded-frame | 0.33 | 0.77 | 0.98 | 0.98 |
| graded-cascade | 0.37 | 0.80 | 0.98 | 0.98 |
| graded-symbolic | 0.37 | 0.80 | 0.98 | 0.98 |
| graded+llm | 0.42 | 0.79 | 0.98 | 0.98 |
| graded+llm+heur | 0.41 | 0.78 | 0.98 | 0.98 |
| graded+llmsum | 0.39 | 0.84 | 1.00 | 1.00 |
| graded+llm+llmsum | 0.42 | 0.86 | 1.00 | 1.00 |

## Retrievable (answer kept OR a stub pointing at its node kept)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.43 | 0.61 | 0.86 | 0.90 |
| random | 0.62 | 0.81 | 0.88 | 0.86 |
| uniform | 0.87 | 1.00 | 1.00 | 1.00 |
| gc_binary | 0.43 | 0.61 | 0.86 | 0.90 |
| graded | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+oracle | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-facts | 0.99 | 1.00 | 1.00 | 1.00 |
| graded-tomb | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-frame | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-cascade | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-symbolic | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llm | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llm+heur | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llmsum | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llm+llmsum | 1.00 | 1.00 | 1.00 | 1.00 |

## Stale value still presented as live (lower is better)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.07 | 0.97 | 1.00 | 1.00 |
| random | 0.50 | 0.84 | 1.00 | 0.92 |
| uniform | 0.00 | 0.77 | 1.00 | 1.00 |
| gc_binary | 0.07 | 0.97 | 1.00 | 1.00 |
| graded | 0.00 | 0.00 | 0.00 | 0.00 |
| graded+oracle | 0.00 | 0.00 | 0.00 | 0.00 |
| graded-facts | 0.00 | 0.00 | 0.00 | 0.00 |
| graded-tomb | 0.30 | 1.00 | 1.00 | 1.00 |
| graded-frame | 0.00 | 0.00 | 0.00 | 0.00 |
| graded-cascade | 0.00 | 0.00 | 0.00 | 0.00 |
| graded-symbolic | 0.00 | 0.00 | 0.00 | 0.00 |
| graded+llm | 0.00 | 0.00 | 0.00 | 0.00 |
| graded+llm+heur | 0.00 | 0.00 | 0.00 | 0.00 |
| graded+llmsum | 0.00 | 0.00 | 0.00 | 0.00 |
| graded+llm+llmsum | 0.00 | 0.00 | 0.00 | 0.00 |

## Tokens actually used (fraction of full; budget in header)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.10 | 0.20 | 0.33 | 0.40 |
| random | 0.10 | 0.20 | 0.31 | 0.42 |
| uniform | 0.10 | 0.16 | 0.28 | 0.29 |
| gc_binary | 0.10 | 0.20 | 0.33 | 0.40 |
| graded | 0.11 | 0.20 | 0.31 | 0.42 |
| graded+oracle | 0.11 | 0.20 | 0.31 | 0.42 |
| graded-facts | 0.10 | 0.19 | 0.31 | 0.41 |
| graded-tomb | 0.10 | 0.20 | 0.31 | 0.42 |
| graded-frame | 0.11 | 0.20 | 0.31 | 0.42 |
| graded-cascade | 0.11 | 0.20 | 0.31 | 0.42 |
| graded-symbolic | 0.11 | 0.20 | 0.31 | 0.42 |
| graded+llm | 0.11 | 0.20 | 0.31 | 0.42 |
| graded+llm+heur | 0.11 | 0.20 | 0.31 | 0.42 |
| graded+llmsum | 0.11 | 0.20 | 0.32 | 0.43 |
| graded+llm+llmsum | 0.11 | 0.20 | 0.32 | 0.43 |

## Retention by question category @ 10% budget

| method | chatter | constraint | deep | instruction | near | rejected | updated |
|---|---|---|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.10 | 0.00 | 0.12 | 1.00 | 0.67 | 0.00 | 1.00 |
| random | 0.82 | 0.66 | 0.23 | 0.86 | 0.51 | 0.39 | 0.99 |
| uniform | 1.00 | 0.23 | 0.00 | 0.63 | 0.33 | 1.00 | 1.00 |
| gc_binary | 0.10 | 0.00 | 0.12 | 1.00 | 0.67 | 0.00 | 1.00 |
| graded | 1.00 | 1.00 | 0.00 | 0.57 | 0.03 | 0.77 | 0.23 |
| graded+oracle | 1.00 | 1.00 | 0.00 | 0.70 | 0.03 | 1.00 | 0.23 |
| graded-facts | 1.00 | 1.00 | 0.00 | 1.00 | 0.13 | 1.00 | 0.67 |
| graded-tomb | 1.00 | 1.00 | 0.00 | 0.73 | 0.08 | 0.83 | 0.37 |
| graded-frame | 1.00 | 1.00 | 0.00 | 0.50 | 0.01 | 0.77 | 0.03 |
| graded-cascade | 1.00 | 1.00 | 0.00 | 0.57 | 0.03 | 0.77 | 0.23 |
| graded-symbolic | 1.00 | 1.00 | 0.00 | 0.57 | 0.03 | 0.77 | 0.23 |
| graded+llm | 1.00 | 1.00 | 0.00 | 0.80 | 0.03 | 1.00 | 0.27 |
| graded+llm+heur | 1.00 | 1.00 | 0.00 | 0.80 | 0.03 | 1.00 | 0.23 |
| graded+llmsum | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.70 | 0.20 |
| graded+llm+llmsum | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.20 |

## Retention by question category @ 20% budget

| method | chatter | constraint | deep | instruction | near | rejected | updated |
|---|---|---|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.60 | 0.00 | 0.50 | 1.00 | 0.69 | 0.47 | 1.00 |
| random | 0.93 | 0.78 | 0.44 | 1.00 | 0.85 | 0.76 | 1.00 |
| uniform | 1.00 | 0.87 | 0.00 | 0.87 | 0.33 | 1.00 | 1.00 |
| gc_binary | 0.60 | 0.00 | 0.50 | 1.00 | 0.69 | 0.47 | 1.00 |
| graded | 1.00 | 1.00 | 0.03 | 1.00 | 0.97 | 1.00 | 1.00 |
| graded+oracle | 1.00 | 1.00 | 0.08 | 1.00 | 0.92 | 1.00 | 1.00 |
| graded-facts | 1.00 | 1.00 | 0.03 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-tomb | 1.00 | 1.00 | 0.03 | 1.00 | 0.97 | 1.00 | 1.00 |
| graded-frame | 1.00 | 1.00 | 0.03 | 1.00 | 0.88 | 1.00 | 1.00 |
| graded-cascade | 1.00 | 1.00 | 0.05 | 1.00 | 0.98 | 1.00 | 1.00 |
| graded-symbolic | 1.00 | 1.00 | 0.03 | 1.00 | 0.97 | 1.00 | 1.00 |
| graded+llm | 1.00 | 1.00 | 0.10 | 1.00 | 0.91 | 1.00 | 1.00 |
| graded+llm+heur | 1.00 | 1.00 | 0.07 | 1.00 | 0.90 | 1.00 | 1.00 |
| graded+llmsum | 1.00 | 1.00 | 0.22 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llm+llmsum | 1.00 | 1.00 | 0.28 | 1.00 | 1.00 | 1.00 | 1.00 |

## Retention by question category @ 35% budget

| method | chatter | constraint | deep | instruction | near | rejected | updated |
|---|---|---|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 1.00 | 0.60 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| random | 1.00 | 0.90 | 0.50 | 1.00 | 0.96 | 0.88 | 1.00 |
| uniform | 1.00 | 1.00 | 0.37 | 1.00 | 0.93 | 1.00 | 1.00 |
| gc_binary | 1.00 | 0.60 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+oracle | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-facts | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-tomb | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-frame | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-cascade | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-symbolic | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llm | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llm+heur | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llmsum | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llm+llmsum | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Retention by question category @ 60% budget

| method | chatter | constraint | deep | instruction | near | rejected | updated |
|---|---|---|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| random | 0.92 | 0.91 | 0.51 | 0.98 | 0.94 | 0.89 | 0.98 |
| uniform | 1.00 | 1.00 | 0.42 | 1.00 | 1.00 | 1.00 | 1.00 |
| gc_binary | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+oracle | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-facts | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-tomb | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-frame | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-cascade | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-symbolic | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llm | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llm+heur | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llmsum | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llm+llmsum | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Model-free edge heuristics vs oracle edges (mark phase)

- precision 0.10, recall 0.53 (≈128 predicted vs 24 true edges per case, format-given tool/reads edges excluded)
- outdated value tombstoned: 1.00 (attributed to the exact updating node: 0.37); rejected proposal tombstoned: 0.63 (exact: 0.63); false tombstones per case: 0.00
- user-sentence role accuracy (goal/constraint/user/chatter): 0.57

## Model-built edges (deepseek:deepseek-flash:low) vs oracle edges (mark phase)

- precision 0.41, recall 0.77 (≈46 predicted vs 24 true edges per case, format-given tool/reads edges excluded)
- outdated value tombstoned: 1.00 (attributed to the exact updating node: 1.00); rejected proposal tombstoned: 1.00 (exact: 1.00); false tombstones per case: 0.00
- user-sentence role accuracy (goal/constraint/user/chatter): 0.83
- summaries for the *llmsum methods: deepseek:deepseek-flash:off
