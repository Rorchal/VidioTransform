# Model-answered QA on LongMemEval_s (histories ≤100000 turns)

90 questions; methods truncate, graded, graded-agent+lex; budgets [0.2, 0.35].

Answered by claude-cli-bare:claude-haiku-4-5-20251001, judged by claude-cli:claude-opus-5-5.

| context | avg tokens | accuracy | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|---|---|
| full@1.0 | 126250 | 0.62 | 0.87 | 0.53 | 0.93 | 0.20 | 0.80 | 0.40 |
| truncate@0.2 | 25211 | 0.29 | 0.67 | 0.33 | 0.27 | 0.07 | 0.20 | 0.20 |
| truncate@0.35 | 44133 | 0.38 | 0.67 | 0.33 | 0.47 | 0.27 | 0.33 | 0.20 |
| graded@0.2 | 25130 | 0.56 | 0.80 | 0.60 | 0.13 | 0.33 | 0.93 | 0.53 |
| graded@0.35 | 43940 | 0.66 | 0.80 | 0.87 | 0.53 | 0.20 | 0.93 | 0.60 |
| graded-agent+lex@0.2 | 25128 | 0.58 | 0.80 | 0.80 | 0.20 | 0.13 | 1.00 | 0.53 |
| graded-agent+lex@0.35 | 43970 | 0.68 | 0.80 | 0.80 | 0.87 | 0.27 | 0.93 | 0.40 |
