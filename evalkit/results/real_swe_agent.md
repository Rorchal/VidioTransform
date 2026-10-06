# ctxgc on real SWE-agent trajectories (nebius/SWE-agent-trajectories)

60 trajectories, 90 cut points; prefix ≈ 8408 tokens, ≈ 5 carried identifiers per cut. Budget = fraction of the prefix's full token count.


## Carried identifiers still present (the agent uses them later)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.74 | 0.83 | 0.89 | 0.98 |
| random | 0.65 | 0.78 | 0.89 | 0.95 |
| uniform | 0.27 | 0.38 | 0.63 | 0.78 |
| gc_binary | 0.72 | 0.83 | 0.88 | 0.98 |
| graded | 0.22 | 0.45 | 0.72 | 0.84 |
| graded-agent | 0.53 | 0.72 | 0.79 | 0.88 |
| graded-pinL0 | 0.09 | 0.24 | 0.58 | 0.83 |

## Retrievable (present, or a stub for a node holding it survives)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.74 | 0.83 | 0.89 | 0.98 |
| random | 0.65 | 0.78 | 0.89 | 0.95 |
| uniform | 1.00 | 1.00 | 1.00 | 1.00 |
| gc_binary | 0.72 | 0.83 | 0.88 | 0.98 |
| graded | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-agent | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-pinL0 | 1.00 | 1.00 | 1.00 | 1.00 |

## File paths named in later commands still present (1 − likely re-reads)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.78 | 0.83 | 0.83 | 0.95 |
| random | 0.78 | 0.92 | 0.94 | 0.98 |
| uniform | 0.35 | 0.57 | 0.82 | 0.96 |
| gc_binary | 0.74 | 0.81 | 0.81 | 0.94 |
| graded | 0.34 | 0.72 | 0.92 | 0.98 |
| graded-agent | 0.89 | 0.90 | 0.94 | 0.98 |
| graded-pinL0 | 0.07 | 0.33 | 0.69 | 0.95 |

## Identifiers from the opening request the agent reuses later, still present

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.69 | 0.76 | 0.81 | 0.87 |
| random | 0.68 | 0.83 | 0.92 | 0.96 |
| uniform | 0.28 | 0.39 | 0.57 | 0.83 |
| gc_binary | 0.69 | 0.76 | 0.81 | 0.87 |
| graded | 0.20 | 0.50 | 0.75 | 0.94 |
| graded-agent | 0.50 | 0.71 | 0.87 | 0.97 |
| graded-pinL0 | 0.97 | 0.98 | 1.00 | 1.00 |

## Tokens actually used (fraction of full)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.10 | 0.20 | 0.35 | 0.60 |
| random | 0.10 | 0.20 | 0.35 | 0.60 |
| uniform | 0.09 | 0.15 | 0.25 | 0.37 |
| gc_binary | 0.10 | 0.20 | 0.35 | 0.60 |
| graded | 0.10 | 0.20 | 0.34 | 0.56 |
| graded-agent | 0.10 | 0.20 | 0.35 | 0.58 |
| graded-pinL0 | 0.23 | 0.26 | 0.36 | 0.56 |
