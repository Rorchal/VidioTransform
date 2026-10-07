# ctxgc on Claude Code session logs

1 trajectories, 2 cut points; prefix ≈ 102032 tokens, ≈ 101 carried identifiers per cut. Budget = fraction of the prefix's full token count.


## Carried identifiers still present (the agent uses them later)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.43 | 0.54 | 0.66 | 0.79 |
| random | 0.38 | 0.57 | 0.69 | 0.94 |
| uniform | 0.11 | 0.11 | 0.11 | 0.77 |
| gc_binary | 0.43 | 0.54 | 0.66 | 0.79 |
| graded | 0.25 | 0.46 | 0.63 | 0.78 |
| graded-agent | 0.36 | 0.53 | 0.68 | 0.80 |
| graded-pinL0 | 0.25 | 0.46 | 0.63 | 0.78 |
| graded+llmsum | 0.28 | 0.43 | 0.65 | 0.78 |
| graded-agent+llmsum | 0.37 | 0.52 | 0.68 | 0.83 |

## Retrievable (present, or a stub for a node holding it survives)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.43 | 0.54 | 0.88 | 0.99 |
| random | 0.52 | 0.70 | 0.78 | 0.96 |
| uniform | 1.00 | 1.00 | 1.00 | 1.00 |
| gc_binary | 0.43 | 0.54 | 0.88 | 0.99 |
| graded | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-agent | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-pinL0 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llmsum | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-agent+llmsum | 1.00 | 1.00 | 1.00 | 1.00 |

## File paths named in later commands still present (1 − likely re-reads)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.55 | 0.69 | 0.73 | 0.90 |
| random | 0.63 | 0.88 | 0.95 | 0.99 |
| uniform | 0.46 | 0.46 | 0.46 | 0.97 |
| gc_binary | 0.55 | 0.69 | 0.73 | 0.90 |
| graded | 0.94 | 0.97 | 0.97 | 0.97 |
| graded-agent | 0.90 | 0.94 | 0.97 | 1.00 |
| graded-pinL0 | 0.94 | 0.97 | 0.97 | 0.97 |
| graded+llmsum | 0.94 | 0.97 | 0.97 | 0.97 |
| graded-agent+llmsum | 0.97 | 0.97 | 0.97 | 1.00 |

## Tokens actually used (fraction of full)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.10 | 0.20 | 0.35 | 0.60 |
| random | 0.10 | 0.20 | 0.35 | 0.60 |
| uniform | 0.04 | 0.04 | 0.04 | 0.50 |
| gc_binary | 0.10 | 0.20 | 0.35 | 0.60 |
| graded | 0.10 | 0.20 | 0.34 | 0.59 |
| graded-agent | 0.10 | 0.20 | 0.35 | 0.60 |
| graded-pinL0 | 0.10 | 0.20 | 0.34 | 0.59 |
| graded+llmsum | 0.10 | 0.20 | 0.35 | 0.60 |
| graded-agent+llmsum | 0.10 | 0.20 | 0.35 | 0.60 |
