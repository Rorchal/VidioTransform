#!/usr/bin/env bash
# Usage: setup_task.sh <work_root> <instance_id> <base_sha> <python_version>
# Creates <work_root>/<iid>/repo (shallow checkout at base_sha) and <work_root>/<iid>/venv (repo installed -e).
set -euo pipefail
root=$1; iid=$2; sha=$3; py=$4
owner=${iid%%__*}; rest=${iid#*__}; name=${rest%-*}
d=$root/$iid
mkdir -p "$d"
if [ ! -d "$d/repo/.git" ]; then
  git init -q "$d/repo"
  git -C "$d/repo" remote add origin "https://github.com/$owner/$name.git"
  for i in 1 2 3; do git -C "$d/repo" fetch -q --depth 1 origin "$sha" && break || sleep $((2**i)); done
  git -C "$d/repo" checkout -q FETCH_HEAD
fi
[ -d "$d/venv" ] || uv venv -q -p "$py" "$d/venv"
uv pip install -q -p "$d/venv/bin/python" -e "$d/repo" >"$d/install.log" 2>&1
if [ "$owner" = sympy ]; then mod=sympy; else mod=django; fi
"$d/venv/bin/python" -c "import $mod; print('$iid', $mod.__version__)"
