# Model-answered QA on LongMemEval_s (histories ≤120 turns)

120 questions; methods truncate, graded, graded-agent+lex; budgets [0.2, 0.35].

| context | avg tokens | accuracy | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|---|---|
| full@1.0 | 30983 | 0.62 | 0.75 | 0.55 | 0.95 | 0.20 | 0.90 | 0.40 |
| truncate@0.2 | 6172 | 0.23 | 0.65 | 0.05 | 0.30 | 0.05 | 0.20 | 0.15 |
| truncate@0.35 | 10818 | 0.32 | 0.80 | 0.15 | 0.40 | 0.00 | 0.35 | 0.20 |
| graded@0.2 | 6134 | 0.51 | 0.65 | 0.45 | 0.45 | 0.15 | 0.95 | 0.40 |
| graded@0.35 | 10684 | 0.56 | 0.70 | 0.50 | 0.70 | 0.15 | 0.90 | 0.40 |
| graded-agent+lex@0.2 | 6141 | 0.49 | 0.65 | 0.65 | 0.30 | 0.00 | 0.95 | 0.40 |
| graded-agent+lex@0.35 | 10718 | 0.63 | 0.75 | 0.65 | 0.75 | 0.25 | 0.95 | 0.45 |
