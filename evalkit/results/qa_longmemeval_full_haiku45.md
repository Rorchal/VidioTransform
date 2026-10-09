# Model-answered QA on LongMemEval_s (histories ≤100000 turns)

90 questions; methods truncate, graded, graded-agent+lex; budgets [0.2, 0.35].

| context | avg tokens | accuracy | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|---|---|
| full@1.0 | 126250 | 0.59 | 0.73 | 0.53 | 0.93 | 0.40 | 0.60 | 0.33 |
| truncate@0.2 | 25211 | 0.19 | 0.60 | 0.07 | 0.27 | 0.00 | 0.07 | 0.13 |
| truncate@0.35 | 44133 | 0.28 | 0.67 | 0.07 | 0.40 | 0.13 | 0.20 | 0.20 |
| graded@0.2 | 25130 | 0.50 | 0.80 | 0.40 | 0.13 | 0.27 | 0.87 | 0.53 |
| graded@0.35 | 43940 | 0.57 | 0.73 | 0.60 | 0.60 | 0.33 | 0.80 | 0.33 |
| graded-agent+lex@0.2 | 25128 | 0.49 | 0.73 | 0.53 | 0.20 | 0.07 | 0.87 | 0.53 |
| graded-agent+lex@0.35 | 43970 | 0.63 | 0.73 | 0.67 | 0.80 | 0.33 | 0.80 | 0.47 |
