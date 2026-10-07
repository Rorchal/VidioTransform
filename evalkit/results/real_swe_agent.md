# ctxgc on real SWE-agent trajectories (nebius/SWE-agent-trajectories)

60 trajectories, 65 cut points; prefix ≈ 7838 tokens, ≈ 5 carried identifiers per cut. Budget = fraction of the prefix's full token count.


## Carried identifiers still present (the agent uses them later)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.67 | 0.84 | 0.89 | 0.98 |
| random | 0.52 | 0.73 | 0.85 | 0.95 |
| uniform | 0.24 | 0.34 | 0.60 | 0.77 |
| gc_binary | 0.67 | 0.81 | 0.87 | 0.96 |
| graded | 0.53 | 0.66 | 0.81 | 0.91 |
| graded-agent | 0.60 | 0.70 | 0.82 | 0.92 |
| graded-pinL0 | 0.09 | 0.22 | 0.53 | 0.84 |
| graded+llmsum | 0.57 | 0.73 | 0.90 | 0.96 |
| graded-agent+llmsum | 0.67 | 0.81 | 0.90 | 0.96 |

## Retrievable (present, or a stub for a node holding it survives)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.67 | 0.84 | 0.89 | 0.98 |
| random | 0.52 | 0.73 | 0.85 | 0.95 |
| uniform | 1.00 | 1.00 | 1.00 | 1.00 |
| gc_binary | 0.67 | 0.81 | 0.87 | 0.96 |
| graded | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-agent | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-pinL0 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llmsum | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-agent+llmsum | 1.00 | 1.00 | 1.00 | 1.00 |

## File paths named in later commands still present (1 − likely re-reads)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.66 | 0.80 | 0.81 | 0.93 |
| random | 0.82 | 0.94 | 0.99 | 1.00 |
| uniform | 0.12 | 0.40 | 0.85 | 0.95 |
| gc_binary | 0.67 | 0.77 | 0.80 | 0.92 |
| graded | 0.86 | 0.95 | 0.99 | 0.99 |
| graded-agent | 0.91 | 0.97 | 0.99 | 0.99 |
| graded-pinL0 | 0.05 | 0.34 | 0.75 | 0.99 |
| graded+llmsum | 0.87 | 0.99 | 0.99 | 0.99 |
| graded-agent+llmsum | 0.92 | 0.97 | 0.99 | 0.99 |

## Identifiers from the opening request the agent reuses later, still present

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.60 | 0.67 | 0.76 | 0.91 |
| random | 0.69 | 0.78 | 0.92 | 0.97 |
| uniform | 0.42 | 0.57 | 0.68 | 0.88 |
| gc_binary | 0.60 | 0.67 | 0.77 | 0.91 |
| graded | 0.59 | 0.84 | 0.94 | 0.97 |
| graded-agent | 0.78 | 0.90 | 0.95 | 0.99 |
| graded-pinL0 | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+llmsum | 0.65 | 0.84 | 0.93 | 0.97 |
| graded-agent+llmsum | 0.82 | 0.92 | 0.96 | 0.97 |

## Tokens actually used (fraction of full)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.10 | 0.20 | 0.35 | 0.59 |
| random | 0.10 | 0.20 | 0.35 | 0.59 |
| uniform | 0.09 | 0.15 | 0.25 | 0.39 |
| gc_binary | 0.10 | 0.20 | 0.35 | 0.59 |
| graded | 0.10 | 0.20 | 0.34 | 0.57 |
| graded-agent | 0.10 | 0.20 | 0.34 | 0.58 |
| graded-pinL0 | 0.23 | 0.26 | 0.36 | 0.56 |
| graded+llmsum | 0.10 | 0.20 | 0.34 | 0.57 |
| graded-agent+llmsum | 0.10 | 0.20 | 0.35 | 0.58 |
