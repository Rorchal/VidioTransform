# Model-answered QA on LongMemEval_s (histories ≤100000 turns)

30 questions; methods truncate, graded, graded-agent+lex; budgets [0.2, 0.35].

| context | avg tokens | accuracy | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|---|---|
| full@1.0 | 126085 | 0.47 | 0.60 | 0.20 | 1.00 | 0.00 | 0.60 | 0.40 |
| truncate@0.2 | 25179 | 0.23 | 0.60 | 0.00 | 0.00 | 0.20 | 0.40 | 0.20 |
| truncate@0.35 | 44077 | 0.40 | 1.00 | 0.00 | 0.40 | 0.40 | 0.60 | 0.00 |
| graded@0.2 | 25099 | 0.47 | 0.80 | 0.20 | 0.40 | 0.20 | 0.60 | 0.60 |
| graded@0.35 | 43899 | 0.53 | 0.80 | 0.20 | 0.80 | 0.20 | 0.60 | 0.60 |
| graded-agent+lex@0.2 | 25097 | 0.33 | 0.80 | 0.20 | 0.40 | 0.00 | 0.60 | 0.00 |
| graded-agent+lex@0.35 | 43929 | 0.47 | 0.80 | 0.00 | 1.00 | 0.40 | 0.60 | 0.00 |
