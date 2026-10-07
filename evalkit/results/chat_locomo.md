# ctxgc on LoCoMo (two-person conversations; speaker A as user, B as assistant)

50 questions; history ≈ 24290 tokens, ≈ 1162 nodes per question (the question itself is the final user turn). Budget = fraction of the full history's token count.


## Answer string present in the compressed context

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 0.36 | 0.36 | 0.36 | 0.36 |
| truncate | 0.04 | 0.10 | 0.18 | 0.20 |
| random | 0.06 | 0.10 | 0.17 | 0.28 |
| uniform | 0.04 | 0.06 | 0.10 | 0.20 |
| gc_binary | 0.04 | 0.10 | 0.18 | 0.20 |
| graded | 0.02 | 0.08 | 0.14 | 0.26 |
| graded+lex | 0.02 | 0.08 | 0.14 | 0.26 |
| graded-agent+lex | 0.02 | 0.10 | 0.18 | 0.26 |

## Labelled evidence turns kept verbatim or as a paragraph summary (L0/L1)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.10 | 0.16 | 0.28 | 0.49 |
| random | 0.07 | 0.16 | 0.31 | 0.57 |
| uniform | 0.00 | 0.00 | 0.00 | 0.00 |
| gc_binary | 0.10 | 0.16 | 0.28 | 0.49 |
| graded | 0.03 | 0.05 | 0.11 | 0.33 |
| graded+lex | 0.03 | 0.05 | 0.18 | 0.37 |
| graded-agent+lex | 0.03 | 0.10 | 0.28 | 0.47 |

## Labelled evidence turns kept verbatim (L0)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.10 | 0.16 | 0.28 | 0.49 |
| random | 0.07 | 0.16 | 0.31 | 0.57 |
| uniform | 0.00 | 0.00 | 0.00 | 0.00 |
| gc_binary | 0.10 | 0.16 | 0.28 | 0.49 |
| graded | 0.03 | 0.03 | 0.03 | 0.07 |
| graded+lex | 0.03 | 0.03 | 0.03 | 0.09 |
| graded-agent+lex | 0.03 | 0.05 | 0.21 | 0.28 |

## Retrievable (answer present, or a stub for an evidence turn survives)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.14 | 0.26 | 0.38 | 0.58 |
| random | 0.28 | 0.46 | 0.60 | 0.79 |
| uniform | 0.20 | 0.34 | 0.62 | 1.00 |
| gc_binary | 0.14 | 0.26 | 0.38 | 0.58 |
| graded | 1.00 | 1.00 | 1.00 | 1.00 |
| graded+lex | 1.00 | 1.00 | 1.00 | 1.00 |
| graded-agent+lex | 1.00 | 1.00 | 1.00 | 1.00 |

## Tokens actually used (fraction of full)

| method | 10% | 20% | 35% | 60% |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.10 | 0.20 | 0.35 | 0.60 |
| random | 0.10 | 0.20 | 0.35 | 0.60 |
| uniform | 0.10 | 0.20 | 0.35 | 0.56 |
| gc_binary | 0.10 | 0.20 | 0.35 | 0.60 |
| graded | 0.12 | 0.20 | 0.35 | 0.60 |
| graded+lex | 0.12 | 0.20 | 0.35 | 0.60 |
| graded-agent+lex | 0.12 | 0.20 | 0.35 | 0.60 |

## Answer present by question type @ 20% budget

| method | cat1 | cat2 | cat3 | cat4 |
|---|---|---|---|---|
| full | 0.33 | 0.20 | 0.20 | 0.48 |
| truncate | 0.17 | 0.00 | 0.20 | 0.09 |
| random | 0.12 | 0.05 | 0.20 | 0.09 |
| uniform | 0.00 | 0.00 | 0.20 | 0.09 |
| gc_binary | 0.17 | 0.00 | 0.20 | 0.09 |
| graded | 0.00 | 0.00 | 0.20 | 0.13 |
| graded+lex | 0.00 | 0.00 | 0.20 | 0.13 |
| graded-agent+lex | 0.08 | 0.00 | 0.20 | 0.13 |

## Evidence at L0/L1 by question type @ 20% budget

| method | cat1 | cat2 | cat3 | cat4 |
|---|---|---|---|---|
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| truncate | 0.06 | 0.25 | 0.32 | 0.13 |
| random | 0.15 | 0.22 | 0.18 | 0.13 |
| uniform | 0.00 | 0.00 | 0.00 | 0.00 |
| gc_binary | 0.06 | 0.25 | 0.32 | 0.13 |
| graded | 0.02 | 0.04 | 0.17 | 0.04 |
| graded+lex | 0.02 | 0.06 | 0.17 | 0.04 |
| graded-agent+lex | 0.03 | 0.30 | 0.17 | 0.04 |
