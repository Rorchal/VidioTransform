#!/usr/bin/env bash
# Usage: run_cli.sh <work_root> <instance_id> <model> [default|nosys]
# Runs headless Claude Code on one task (same prompt template as the subagent run) from <work_root>/<iid>,
# so its transcript lands in ~/.claude/projects/<slug of that dir>/<session>.jsonl. Summary -> run.json.
# nosys: empty system prompt, only the six file/shell tool definitions, no skills listing.
set -uo pipefail
root=$1; iid=$2; model=$3; mode=${4:-default}
extra=()
[ "$mode" = nosys ] && extra=(--system-prompt "" --tools "Bash,Read,Edit,Write,Grep,Glob" --disable-slash-commands)
d=$root/$iid
if [[ $iid == sympy* ]]; then
  proj="SymPy"; tests="SymPy's tests run with: cd <repo> && <venv python> bin/test <path/to/test_file.py> (pytest is not installed)."
else
  proj="Django"; tests="Django's test suite runs with: cd <repo>/tests && <venv python> runtests.py <test_label> (SQLite, no extra setup)."
fi
prompt="You are a software engineer fixing a bug in an open-source repository.

Repository: $d/repo ($proj, checked out at the commit where the issue was reported)
Python: $d/venv/bin/python already has this repo installed in editable mode. Use it for every Python command. $tests

Task: make changes to non-test source files in the repository so the issue in the PR description below is fixed.
- Do not modify existing tests or configuration files. You may create scratch scripts to reproduce and verify.
- Do not create git commits. Do not pip install anything or use the network. Work only inside $d. Do the work yourself; do not spawn subagents.
- When finished, reply with a short summary: files changed, and how you verified the fix.

<pr_description>
$(cat "$d/problem.md")
</pr_description>"
cd "$d"
timeout 3600 claude -p "$prompt" --model "$model" --output-format json \
  --allowedTools "Bash Read Edit Write Grep Glob" \
  --disallowedTools "Agent Task WebFetch WebSearch" \
  --strict-mcp-config --max-turns 150 "${extra[@]}" < /dev/null > "$d/run.json" 2> "$d/run.err"
echo "$iid exit=$? $(python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print("turns",d.get("num_turns"),"err",d.get("is_error"),"cost",round(d.get("total_cost_usd") or 0,3),"models",list((d.get("modelUsage") or {}).keys()))' "$d/run.json" 2>&1 | tail -1)"
