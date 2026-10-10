# ctxcache — agent 上下文压缩的离线重放器

把一条 agent 轨迹（每步：命令 → 工具输出）按不同策略重放，算出每一步会发给模型多大的上下文，
以及策略把文件内容扔掉之后 agent 要多读几次文件（miss）。不调模型，不花钱。

## 跑

```bash
# 1. 数据：441 条 SWE-bench Verified 真实轨迹（mini-SWE-agent）
git clone --depth 1 https://github.com/john-b-yang/20260901_mini-v2.4.2_gemini-3-5-flash.git minitraj

# 2. 重放（五种策略 × 三档预算），约 20 秒
PYTHONPATH=. python3 -m ctxcache.replay minitraj/trajs out/replay.json 8000,16000,32000

# 3. 看一条轨迹的曲线
PYTHONPATH=. python3 -m ctxcache.plot minitraj/trajs/django__django-10554.traj.json 16000 out/one.png

# 4. 换成你自己的 Claude Code 日志（格式是内部的、会变，适配器是尽力解析）
PYTHONPATH=. python3 -m ctxcache.replay ~/.claude/projects/<项目目录>/ out/mine.json 50000,100000

# 加 --chars 回到旧的「字符数 / 4」估算（对照用）
PYTHONPATH=. python3 -m ctxcache.replay minitraj/trajs out/replay.json 8000,16000,32000 --chars
```

## token 怎么算

默认用轨迹里记录的 **API 真实用量**，两种格式都有：mini-SWE-agent 每次调用的 `prompt_tokens`，
Claude Code 每条回复的 `usage`。第 k+1 次调用收到的上下文减去第 k 次的，就是第 k 步新加的内容；
其中模型自己的输出（output / completion tokens）算 agent 一侧，剩下的算工具输出。
各策略裁剪时的比例（留最后 15 行、只留文件名……）仍按文本算，再乘到真实 token 上。

- Claude 的思考在工具循环里会被原样带回，但日志里看不到原文，单独记为思考 token；除 compact 外各策略都保留它。
- 最后一次调用、或上下文变小的调用没有可用的差值，退回「字符数 / 4 × 本条轨迹的真实/估算中位比」。
- 每次运行会打印校准行：不做任何处理（raw）时重放出的每次调用上下文 vs 真实值。441 条 Gemini 轨迹和
  Haiku 4.5 的 Claude Code 日志上都是误差中位数 0.0%。

旧的「字符数 / 4」大约**低估一半**：Gemini 长轨迹不做处理的峰值，估算 24.6k，真实 48.6k。

## Claude Code 日志

- 带 `cd 目录 &&`、`变量=…;`、`timeout N` 前缀的命令先剥掉前缀再分类，`…/venv/bin/python` 当 `python`；
  否则这类命令全部被当成 other，`slim` 会把读文件的输出也只留最后 15 行。
- 路径按每条记录的 `cwd`（和命令里的 `cd`）还原，统一成相对会话根目录的写法，
  所以 Read/Edit 的绝对路径和 bash 里的相对路径指向同一个文件。mini 轨迹按 `/testbed` 同样处理。
- Edit / Write / MultiEdit 的内容计入 agent 一侧（slim 视为已落盘，只记 15 token）。
- 并行工具调用算一次模型调用：累计发送按调用计，不按步计。
- 模型偶尔把 Read 的 `offset` / `limit` 传成列表，解析时取第一个数，不再报错。
- 预算要包含固定开销：默认系统提示词 + 全部工具约 28k；`--system-prompt ""` 加 6 个工具约 8k。

## 策略

| 名字 | 做什么 |
|---|---|
| raw | 什么都不做 |
| tail10 | 只留最近 10 步的工具输出 |
| compact | 模拟 Claude Code 的 /compact：到预算 90% 时，把 8 步以前的全扔，放一段 1200 token 的摘要，保留最近读过的 5 个文件。每次压缩额外算一次全量发送（摘要请求本身要读一遍上下文） |
| slim | 只做「进门瘦身」：测试输出只留退出码 + 最后 15 行；grep 只留命中的文件名；同一文件重复读的重叠行不再进第二遍；命令里的内联脚本当作已落盘只记 15 token；agent 的「我接下来去看 X」类叙述丢掉 |
| ours | slim + 改文件后相关内容标过期 + 到预算 hi% 时按活跃度淘汰到 lo%（先淘汰过期的，再淘汰「10 步没碰、agent 最近没提、总共碰过不到 3 次」的，还不够才动活跃的）。文件内容被淘汰后留一行 25 token 的记录 |

miss 的定义：agent 要改某个文件，它之前读过这个文件，但策略已经把内容扔了、它又没重读 → 记一次 miss，并模拟它重读一次（把内容加回上下文）。
fresh miss：被扔掉时内容还没过期（文件没被改过）——这是真正的损失；过期的内容扔了问题不大。
`git checkout` / `rm` 不算需要内容的改动。

## 结果（441 条，预算 16k，真实 token；长轨迹 = ≥50 步的 129 条）

```
 budget policy        peak k  sent k  compacts  misses  fresh  over
  16000 raw             48.6    1586      0.00    0.00   0.00  44.8   ← 45 步超预算，真跑会崩
  16000 tail10          17.2     621      0.00    0.40   0.12   9.6
  16000 compact         14.4     782      9.76    0.25   0.07   0.2   ← 每次 compact 是一次模型调用
  16000 slim            28.3     937      0.00    0.00   0.00  31.4   ← 光瘦身远远不够
  16000 ours 90/70      14.3     605      4.28    0.26   0.07   0.0
  16000 ours 60/40       9.5     416      5.75    0.43   0.19   0.0
```

peak = 单步最大上下文；sent = 所有步的上下文加总（你付的钱）；compacts = 压缩次数；over = 超预算的步数。
完整表（含全部 441 条）在 `replay_table.txt`；旧的「字符数 / 4」版本在 `replay_table_chars.txt`。

瘦身省在哪（全部轨迹合计，字符数 / 4 估算）：

```
部分        原始 k   瘦身后 k   保留
read        2222     1798      81%   ← 只靠合并重复区间
run         1657      540      33%
search      1377      375      27%
agent 自己  1215      265      22%   ← 内联脚本 + 叙述
合计        7259     3541      49%
```

## Claude Code + Haiku 4.5（50 道题）

`experiments/haiku45_swe/` 里有用 Haiku 4.5 在 50 道 SWE-bench Verified 题上真跑的轨迹、打分和重放结果。要点：

- 不压缩时解出 72%（34 / 47 道可用题），同题 Gemini 3.5 Flash 85%。
- 真实上下文峰值中位数 58.6k，累计发送约为 Gemini 的 1.9 倍。
- 上下文里约一半是思考（21%）、固定开销和 agent 自己的文字，现有策略都不动它们，所以压缩效果明显弱于 Gemini：
  预算 48k 时 `ours 90/70` 省 22%、基本不重读；省到 40% 要接受每题约 1 次重读。预算 32k 对它太紧。

## 文件

- `ctxcache/model.py` 两种输入格式的解析（mini-SWE-agent `.traj.json`、Claude Code `.jsonl`）
- `ctxcache/cache.py` 五种策略、过期、淘汰、miss 模拟
- `ctxcache/replay.py` 批量跑 + 打表
- `ctxcache/plot.py` 单条轨迹曲线
- `scripts/agent_parse.py` 从 shell 命令里猜动作类型（read/search/run/edit）和涉及的文件

## 已知限制

- 自带数据只有一个 agent（纯 bash）、一个模型（Gemini 3.5 Flash）的轨迹。
- 真实 token 只覆盖「不做处理」那条线；各策略裁剪掉的部分仍按文本比例折算。
- 思考 token 按「全部保留」算（Claude 工具循环内的实际行为）；清掉旧思考的策略没有模拟。
- compact 是结构代理，不是真的 LLM 摘要；它丢掉决策和排除项造成的损失没算进去，只算了文件内容的 miss。
- 「淘汰后 agent 多读一次」是模拟，没有真跑模型验证解决率。50 道题的真跑（`experiments/haiku45_swe/`）是不压缩的基线；
  要验证压缩后的解决率，得在 agent 循环里实时套用策略，需要自己的 agent 循环（直接调 API），Claude Code 本身做不到。
