# Model-answered QA on LoCoMo

50 questions; methods truncate, graded, graded-agent+lex; budgets [0.2, 0.35].

| context | avg tokens | accuracy | cat1 | cat2 | cat3 | cat4 |
|---|---|---|---|---|---|---|
| full@1.0 | 24273 | 0.76 | 0.67 | 0.70 | 0.40 | 0.91 |
| truncate@0.2 | 4812 | 0.14 | 0.17 | 0.10 | 0.00 | 0.17 |
| truncate@0.35 | 8432 | 0.24 | 0.17 | 0.30 | 0.00 | 0.30 |
| graded@0.2 | 4798 | 0.10 | 0.17 | 0.00 | 0.00 | 0.13 |
| graded@0.35 | 8406 | 0.14 | 0.25 | 0.00 | 0.00 | 0.17 |
| graded-agent+lex@0.2 | 4804 | 0.20 | 0.25 | 0.20 | 0.00 | 0.22 |
| graded-agent+lex@0.35 | 8417 | 0.34 | 0.25 | 0.30 | 0.00 | 0.48 |
