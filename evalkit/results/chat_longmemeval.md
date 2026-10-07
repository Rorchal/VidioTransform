# ctxgc on LongMemEval_s (histories cut to ≤120 turns: evidence sessions + distractors)

60 questions; history ≈ 32152 tokens, ≈ 228 nodes per question (the question itself is the final user turn). Budget = fraction of the full history's token count.


## Answer string present in the compressed context

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 0.45 | 0.45 | 0.45 | 0.45 |
| truncate | 0.13 | 0.23 | 0.28 | 0.35 |
| random | 0.11 | 0.17 | 0.25 | 0.34 |
| uniform | 0.10 | 0.18 | 0.37 | 0.38 |
| gc_binary | 0.13 | 0.23 | 0.28 | 0.35 |
| graded | 0.17 | 0.38 | 0.42 | 0.45 |
| graded+lex | 0.23 | 0.38 | 0.45 | 0.45 |
| graded-agent+lex | 0.30 | 0.37 | 0.42 | 0.45 |

## Labelled evidence turns kept verbatim or as a paragraph summary (L0/L1)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.17 | 0.30 | 0.45 | 0.65 |
| random | 0.12 | 0.24 | 0.37 | 0.68 |
| uniform | 0.00 | 0.03 | 0.95 | 1.00 |
| gc_binary | 0.17 | 0.30 | 0.45 | 0.65 |
| graded | 0.33 | 0.88 | 1.00 | 1.00 |
| graded+lex | 0.39 | 0.91 | 1.00 | 1.00 |
| graded-agent+lex | 0.49 | 0.92 | 1.00 | 1.00 |

## Labelled evidence turns kept verbatim (L0)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.17 | 0.30 | 0.45 | 0.65 |
| random | 0.12 | 0.24 | 0.37 | 0.68 |
| uniform | 0.00 | 0.00 | 0.00 | 0.00 |
| gc_binary | 0.17 | 0.30 | 0.45 | 0.65 |
| graded | 0.03 | 0.54 | 0.92 | 0.98 |
| graded+lex | 0.06 | 0.60 | 0.98 | 1.00 |
| graded-agent+lex | 0.22 | 0.68 | 0.92 | 0.98 |

## Retrievable (answer present, or a stub for an evidence turn survives)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 0.98 | 0.98 | 0.98 | 0.98 |
| truncate | 0.23 | 0.40 | 0.63 | 0.80 |
| random | 0.48 | 0.66 | 0.80 | 0.95 |
| uniform | 0.97 | 0.98 | 0.98 | 0.98 |
| gc_binary | 0.23 | 0.40 | 0.63 | 0.80 |
| graded | 0.98 | 0.98 | 0.98 | 0.98 |
| graded+lex | 0.98 | 0.98 | 0.98 | 0.98 |
| graded-agent+lex | 0.98 | 0.98 | 0.98 | 0.98 |

## Tokens actually used (fraction of full)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.10 | 0.20 | 0.35 | 0.60 |
| random | 0.10 | 0.20 | 0.35 | 0.60 |
| uniform | 0.09 | 0.14 | 0.25 | 0.26 |
| gc_binary | 0.10 | 0.20 | 0.35 | 0.60 |
| graded | 0.10 | 0.20 | 0.34 | 0.59 |
| graded+lex | 0.10 | 0.20 | 0.35 | 0.59 |
| graded-agent+lex | 0.10 | 0.20 | 0.35 | 0.59 |

## Answer present by question type @ 20% budget

| method | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|
| full | 0.80 | 0.20 | 0.70 | 0.00 | 0.70 | 0.30 |
| truncate | 0.60 | 0.00 | 0.30 | 0.00 | 0.30 | 0.20 |
| random | 0.30 | 0.10 | 0.20 | 0.00 | 0.30 | 0.15 |
| uniform | 0.30 | 0.10 | 0.10 | 0.00 | 0.40 | 0.20 |
| gc_binary | 0.60 | 0.00 | 0.30 | 0.00 | 0.30 | 0.20 |
| graded | 0.80 | 0.20 | 0.30 | 0.00 | 0.70 | 0.30 |
| graded+lex | 0.80 | 0.20 | 0.30 | 0.00 | 0.70 | 0.30 |
| graded-agent+lex | 0.80 | 0.20 | 0.20 | 0.00 | 0.70 | 0.30 |

## Evidence at L0/L1 by question type @ 20% budget

| method | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.39 | 0.13 | 0.30 | 0.40 | 0.44 | 0.17 |
| random | 0.22 | 0.29 | 0.17 | 0.31 | 0.18 | 0.25 |
| uniform | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 |
| gc_binary | 0.39 | 0.13 | 0.30 | 0.40 | 0.44 | 0.17 |
| graded | 0.96 | 1.00 | 0.40 | 0.98 | 0.94 | 0.98 |
| graded+lex | 0.96 | 1.00 | 0.60 | 0.98 | 0.94 | 0.98 |
| graded-agent+lex | 1.00 | 0.91 | 0.70 | 0.98 | 1.00 | 0.97 |
