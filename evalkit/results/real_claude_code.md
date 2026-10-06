# ctxgc on Claude Code session logs

1 trajectories, 2 cut points; prefix ≈ 71142 tokens, ≈ 98 carried identifiers per cut. Budget = fraction of the prefix's full token count.


## Carried identifiers still present (the agent uses them later)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.25 | 0.38 | 0.48 | 0.68 |
| random | 0.33 | 0.45 | 0.62 | 0.83 |
| uniform | 0.11 | 0.11 | 0.11 | 0.63 |
| gc_binary | 0.25 | 0.38 | 0.48 | 0.68 |
| graded | 0.23 | 0.41 | 0.54 | 0.65 |
| graded-agent | 0.31 | 0.46 | 0.58 | 0.75 |
| graded-pinL0 | 0.23 | 0.41 | 0.54 | 0.65 |

## Retrievable (present, or a stub for a node holding it survives)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.25 | 0.38 | 0.95 | 1.00 |
| random | 0.33 | 0.65 | 0.78 | 0.96 |
| uniform | 1.00 | 1.00 | 1.00 | 1.00 |
| gc_binary | 0.25 | 0.38 | 0.95 | 1.00 |
| graded | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-agent | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-pinL0 | 1.00 | 1.00 | 1.00 | 1.00 |

## File paths named in later commands still present (1 − likely re-reads)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.42 | 0.54 | 0.58 | 0.90 |
| random | 0.60 | 0.79 | 0.88 | 0.95 |
| uniform | 0.40 | 0.40 | 0.40 | 0.90 |
| gc_binary | 0.42 | 0.54 | 0.58 | 0.90 |
| graded | 0.85 | 0.85 | 0.85 | 0.90 |
| graded-agent | 0.85 | 0.85 | 0.90 | 1.00 |
| graded-pinL0 | 0.85 | 0.85 | 0.85 | 0.90 |

## Tokens actually used (fraction of full)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.10 | 0.20 | 0.35 | 0.60 |
| random | 0.10 | 0.20 | 0.35 | 0.60 |
| uniform | 0.04 | 0.04 | 0.04 | 0.41 |
| gc_binary | 0.10 | 0.20 | 0.35 | 0.60 |
| graded | 0.10 | 0.20 | 0.34 | 0.59 |
| graded-agent | 0.10 | 0.20 | 0.35 | 0.60 |
| graded-pinL0 | 0.10 | 0.20 | 0.34 | 0.59 |
