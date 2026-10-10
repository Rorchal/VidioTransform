# Haiku 4.5 × 50 道 SWE-bench Verified：真实轨迹 + 压缩重放

用 Claude Code（headless，Claude Haiku 4.5，不带系统提示词）在 50 道题上真跑，拿到真实轨迹和 API 用量，
再用 ctxcache 按真实 token 重放各压缩策略，和同样 50 道题上的 Gemini 3.5 Flash（mini-SWE-agent）并排比较。

**这是不压缩的基线**：agent 跑的时候上下文没有做任何处理。压缩策略的效果来自离线重放，
解题率是不压缩时的解题率；「压缩后还能不能解出来」仍然没有真跑验证。

## 怎么跑的

- **题目**：django 35 道、sympy 15 道（`tasks50.json`）。先选了 10 道 Gemini 解出的长轨迹题，另 40 道从 django/sympy 里随机抽，Gemini 没解出的也在内。
- **agent**：Claude Code 2.1.295，
  `claude -p <任务> --model claude-haiku-4-5 --system-prompt "" --tools "Bash,Read,Edit,Write,Grep,Glob" --disable-slash-commands --strict-mcp-config --max-turns 150`。
  首次调用的固定开销约 8k（6 个工具的定义），默认系统提示词下是约 28k。任务提示词见 `tools/run_cli.sh`。
- **环境**：没有 docker。每题浅克隆到 base commit，用 uv 建 venv 并以可编辑方式安装：Python 3.9；Django 5.x 用 3.11；Django < 2.2 用 3.8（uv 没有 3.7）。
- **打分**：SWE-bench Verified 的 test_patch、FAIL_TO_PASS / PASS_TO_PASS，日志用 swebench 自带的解析器。
  - 测试文件会先恢复成原版，再打官方测试补丁。
  - 每题先用官方修复跑一遍作对照，本地环境里本来就失败的测试不算 agent 的错。
  - 官方修复自己都过不了 FAIL_TO_PASS 的题算环境坏，不计入解题率（3 道：`django-12419`、`django-12209`、`django-7530`）。
- **花费**：$21.29（中位数 $0.39 / 题）。

## 结果

解题率（47 道可用题）：Haiku 4.5 **34（72%）**，Gemini 3.5 Flash 40（85%）。
- 两者都解出 33 道；只有 Haiku 解出 1 道，只有 Gemini 解出 7 道。
- Gemini 的数字来自官方 docker 评测，环境和这里不同，只能参考。
- 50 次运行里有 16 次违反指令改了测试文件，打分时都恢复了原版。

真实上下文（每题中位数，API 用量）：

| | Gemini | Haiku 4.5 |
|---|---|---|
| 模型调用次数 | 45 | 44.5 |
| 固定开销（首次调用） | 1.3k | 8.2k |
| 上下文峰值 | 36.3k | 58.6k |
| 累计发送 | 928k | 1732k |

上下文由什么组成（全部 50 题合计，真实 token）：

| | Gemini | Haiku 4.5 |
|---|---|---|
| 读文件输出 | 24% | 26% |
| 运行输出 | 20% | 14% |
| 搜索输出 | 25% | 7% |
| 其他工具输出 | 16% | 4% |
| agent 文字 / 命令 / 改动内容 | 12% | 14% |
| agent 思考 | — | 21% |
| 固定开销 | 4% | 14% |

Haiku 的上下文里约一半是思考、固定开销和 agent 自己的文字，现有策略都不动它们；
能被瘦身的运行和搜索输出只占约 22%，Gemini 是约 50%。所以同样的策略在 Haiku 上省得少、淘汰得多。

压缩重放（`results/summary.txt` 有 16k / 32k / 48k / 64k 全表）。sent 为中位数，Δ 是相对不处理的变化；
压缩、重读（括号内为内容还没过期就被扔掉的次数）、超预算都是每题平均：

| 预算 | 策略 | Gemini | Haiku 4.5 |
|---|---|---|---|
| 32k | raw | 960k，超 15.5 次 | 1691k，超 31.6 次 |
| 32k | slim | −40%，超 3.3 次 | −17%，超 25.9 次 |
| 32k | compact | −21%，压缩 1.3 次 | −31%，压缩 4.9 次，重读 0.16 |
| 32k | ours 90/70 | −41%，压缩 0.5 次，重读 0.02 | −39%，压缩 8.8 次，重读 1.00（0.82），仍超 3.3 次 |
| 48k | ours 90/70 | −40%，压缩 0.1 次，重读 0 | −22%，压缩 1.2 次，重读 0.24（0.06） |
| 48k | ours 60/40 | −42%，压缩 0.4 次，重读 0.06 | −40%，压缩 8.4 次，重读 0.96（0.80） |

- Gemini：`ours` 在 16k 到 64k 都稳定省 40% 到 50%，几乎不重读。
- Haiku 4.5：32k 太紧，有些单次读文件就上万 token，淘汰后很快又要用。48k 时 `ours 90/70` 能省 22%、基本不重读；想省 40% 就要接受每题约 1 次「内容没过期就被扔掉」的重读。
- 想在 Haiku 上省得更多，下一个杠杆是清掉旧的思考（Claude API 的 context editing 支持），重放里还没模拟。

## 复现

```bash
# 重放（不需要重跑模型）
mkdir trajs && tar -xzf trajs_haiku45_nosys_50.tar.gz -C trajs
PYTHONPATH=../.. python3 -m ctxcache.replay trajs out.json 16000,32000,48000,64000
```

`trajs_haiku45_nosys_50.tar.gz` 是清洗过的 Claude Code 日志：
- **保留**：消息内容、token 用量、`cwd`。
- **去掉**：附件（账号 / 会话 / 环境信息、系统提示词快照）、思考块签名。
- **替换**：本机路径和会话 ID 换成等长的占位符，所以文本长度不变。

清洗前后重放结果一致；4 档预算的全表里只有 1 个格子差 0.1%。

重跑整个实验：`tools/` 里依次是选题（`pick_tasks.py`、`pick_more.py`）、建环境（`setup_task.sh`、`extract_problems.py`）、
跑 agent（`run_cli.sh`）、打分（`eval_task.py`，需要 `pip install swebench pyarrow` 和 SWE-bench Verified 的 parquet）、
汇总（`final50.py`）、清洗日志（`sanitize_cc.py`）。各脚本开头写了参数。
