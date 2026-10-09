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
```

token 一律按「字符数 / 4」估算，沙盒里装不上分词器。相对比较不受影响。

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

## 结果（441 条，预算 16k；长轨迹 = ≥50 步的 129 条）

```
 budget policy        peak k  sent k  compacts  misses  fresh  over
  16000 raw             24.6     814      0.00    0.00   0.00  25.1   ← 25 步超预算，真跑会崩
  16000 tail10           9.7     366      0.00    0.40   0.12   0.6
  16000 compact         14.3     564      2.19    0.05   0.02   0.0   ← 每次 compact 是一次模型调用
  16000 slim            13.2     460      0.00    0.00   0.00   4.6   ← 光瘦身还会超预算
  16000 ours 90/70      13.2     458      0.67    0.03   0.00   0.0
  16000 ours 60/40       9.5     388      1.75    0.15   0.02   0.0
```

peak = 单步最大上下文；sent = 所有步的上下文加总（你付的钱）；compacts = 压缩次数；over = 超预算的步数。

瘦身省在哪（全部轨迹合计）：

```
部分        原始 k   瘦身后 k   保留
read        2222     1798      81%   ← 只靠合并重复区间
run         1657      540      33%
search      1377      375      27%
agent 自己  1215      265      22%   ← 内联脚本 + 叙述
合计        7259     3541      49%
```

## 文件

- `ctxcache/model.py` 两种输入格式的解析（mini-SWE-agent `.traj.json`、Claude Code `.jsonl`）
- `ctxcache/cache.py` 五种策略、过期、淘汰、miss 模拟
- `ctxcache/replay.py` 批量跑 + 打表
- `ctxcache/plot.py` 单条轨迹曲线
- `scripts/agent_parse.py` 从 shell 命令里猜动作类型（read/search/run/edit）和涉及的文件

## 已知限制

- 只有一个 agent（纯 bash）、一个模型（Gemini 3.5 Flash）的轨迹。Claude Code 的日志工具是带类型的，不用猜，应该更准。
- compact 是结构代理，不是真的 LLM 摘要；它丢掉决策和排除项造成的损失没算进去，只算了文件内容的 miss。
- 「淘汰后 agent 多读一次」是模拟，没有真跑模型验证解决率。下一步：挑 50 道题真跑。
