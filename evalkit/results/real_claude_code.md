# ctxgc on Claude Code session logs

1 trajectories, 2 cut points; prefix ≈ 77411 tokens, ≈ 128 carried identifiers per cut. Budget = fraction of the prefix's full token count.


## Carried identifiers still present (the agent uses them later)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.40 | 0.51 | 0.60 | 0.74 |
| random | 0.26 | 0.42 | 0.54 | 0.80 |
| uniform | 0.10 | 0.10 | 0.10 | 0.71 |
| gc_binary | 0.40 | 0.51 | 0.60 | 0.74 |
| graded | 0.22 | 0.44 | 0.54 | 0.66 |
| graded-agent | 0.33 | 0.45 | 0.57 | 0.67 |
| graded-pinL0 | 0.22 | 0.44 | 0.54 | 0.66 |
| graded+llmsum | 0.26 | 0.44 | 0.58 | 0.65 |
| graded-agent+llmsum | 0.37 | 0.49 | 0.58 | 0.69 |

## Retrievable (present, or a stub for a node holding it survives)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.40 | 0.51 | 0.80 | 1.00 |
| random | 0.41 | 0.71 | 0.84 | 0.91 |
| uniform | 1.00 | 1.00 | 1.00 | 1.00 |
| gc_binary | 0.40 | 0.51 | 0.80 | 1.00 |
| graded | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-agent | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-pinL0 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llmsum | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-agent+llmsum | 1.00 | 1.00 | 1.00 | 1.00 |

## File paths named in later commands still present (1 − likely re-reads)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.49 | 0.64 | 0.70 | 0.88 |
| random | 0.63 | 0.68 | 0.83 | 0.96 |
| uniform | 0.42 | 0.42 | 0.42 | 0.91 |
| gc_binary | 0.49 | 0.64 | 0.70 | 0.88 |
| graded | 0.79 | 0.88 | 0.85 | 0.91 |
| graded-agent | 0.76 | 0.79 | 0.85 | 0.91 |
| graded-pinL0 | 0.79 | 0.88 | 0.85 | 0.91 |
| graded+llmsum | 0.82 | 0.91 | 0.91 | 0.91 |
| graded-agent+llmsum | 0.82 | 0.85 | 0.85 | 0.91 |

## Tokens actually used (fraction of full)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.10 | 0.20 | 0.35 | 0.60 |
| random | 0.10 | 0.20 | 0.35 | 0.60 |
| uniform | 0.04 | 0.04 | 0.04 | 0.43 |
| gc_binary | 0.10 | 0.20 | 0.35 | 0.60 |
| graded | 0.10 | 0.20 | 0.34 | 0.58 |
| graded-agent | 0.10 | 0.20 | 0.35 | 0.60 |
| graded-pinL0 | 0.10 | 0.20 | 0.34 | 0.58 |
| graded+llmsum | 0.10 | 0.20 | 0.35 | 0.59 |
| graded-agent+llmsum | 0.10 | 0.20 | 0.35 | 0.60 |
