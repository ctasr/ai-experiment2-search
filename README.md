# 实验二：搜索算法

实现题目一、二的A*最短路，以及题目三、四的Minimax和Alpha-beta剪枝。使用Python 3.10或更新版本。四题算法仅依赖标准库，绘图另需安装`requirements.txt`中的Matplotlib。

## 运行

在PowerShell中依次执行：

```powershell
git clone https://github.com/ctasr/ai-experiment2-search.git
cd ai-experiment2-search
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_experiment.py
.\.venv\Scripts\python.exe extended_experiments.py
.\.venv\Scripts\python.exe advanced_experiments.py
```

在macOS或Linux上，首条命令可用`python3 -m venv .venv`，后续将Windows解释器路径替换为`.venv/bin/python`。
绘制中文图表需要系统具备中文字体。代码优先选择已安装的宋体、Noto Sans CJK SC、苹方或文泉驿；若均未安装，会提示补装中文字体。

`run_experiment.py`运行题目样例、基线与对照实验，生成`results/`中的比较表、两张距离矩阵、题目四标准字段的`official_answers.json`、A*与排序策略扫描数据，以及`figures/`中的图。`deterministic_results.json`排除了计时，用于逐次重跑核对结果。A*扫描对两题各生成100张固定种子的地图；排序扫描在固定种子42下取200组宝藏价值。测试还用30张宝藏位置、地形和价值均变化的随机地图，以独立最短路和穷举博弈核对结果。`extended_experiments.py`比较A*同分节点顺序、随机树的剪枝顺序，以及六宝藏场景中的精确Minimax缓存。`advanced_experiments.py`比较普通A*、预处理后A*和预处理后直接查表，也比较双步宝藏排序。两个补充脚本分别保存CSV、汇总JSON、不含计时的确定性JSON与图。重复运行会覆盖生成结果，但不会修改题目代码或测试。

## 文件与题目接口

| 文件 | 内容 |
| --- | --- |
| `search_algorithms.py` | 题一`shortest_maze_path(start, goal, grid)`返回最短坐标路径，无路返回`[]`；题二`minimum_terrain_cost(start, goal, grid)`返回最小总代价，无路返回`-1`。带`_search`后缀的函数额外返回路径、代价与搜索计数。 |
| `game_search.py` | 题三`alpha_beta_best_score(root)`返回根节点MAX的最佳得分；`TreeNode`与题目示例一致。`tree_minimax`用于未剪枝对照。 |
| `treasure_game.py` | 题四`compute_distances()`用A*计算六点距离矩阵，`treasure_game_answer(distances, depth)`按题目字段返回`best_target`、`best_value`，`alpha_beta=False`切换为Minimax。`solve_treasure_game()`另返回详细统计。 |
| `tests/` | 32项测试，覆盖官方样例、边界输入、独立算法核对，以及跨深度配置配对和查表对照的测量流程。 |
| `plot_style.py` | 为三组实验图选择当前系统已安装的中文字体。 |
| `run_experiment.py` | 固定场景、测量和结果导出，另生成题一扩展过程、题二累计代价和题三剪枝过程图。 |
| `extended_experiments.py` | 自设场景与优化策略的补充对照，附正确性检查和可复核数据。 |
| `advanced_experiments.py` | 多查询中普通A*、预处理后A*、直接查表的对照，以及双步排序的收益与开销。 |
| `requirements.txt` | 绘图依赖及版本。 |
| `results/` | `comparison.csv`和`details.json`记录主实验，`distance_matrix*.csv`保存距离矩阵，`official_answers.json`保存题四答案。`*_sweep.csv`、`optimization_*`和`advanced_*`保存各组对照，三份`*deterministic*.json`用于核对非计时结果，`test_log.txt`保存测试日志。 |
| `figures/` | 搜索过程图、节点数对照、深度决策、缓存效果和多查询耗时等实验图，由三个实验脚本生成。 |

题二进入`E`代价为0；题四进入`A/B/T`格代价为1。两题分别实现，不能混用。A*扩展相同`f`的节点时优先较大的`g`，再按入队顺序决定；网格邻居固定按上、右、下、左访问。博弈实验的`given`顺序为T1至T4，`greedy`只调整访问顺序，不改变效用或可选动作。同分首步可能随访问顺序变化，最佳收益差不变。

`results/details.json`中，题一的`zero_expansion_trace`和`manhattan_expansion_trace`按出队顺序记录节点的`g/h/f`、父节点及更新的邻居；题三的`cutoff_trace`记录剪枝时的α/β界和被跳过的分支。`figures/maze_search_process.png`和`figures/tree_pruning_process.png`将这些过程画出来，不改变算法输出。

计数口径：`expanded`指从优先队列取出且不是过期记录的节点数，包含终点；`visited_nodes`指进入博弈递归函数的次数，包含根与叶及缓存命中；`cutoffs`为提前停止一个节点剩余子节点的次数。`details.json`另记录被跳过的分支，题三用从根起的子节点索引表示，题四用宝藏选择序列表示。`comparison.csv`中的时间是同场景算法预热后轮换运行101次的中位数与四分位距。`optimization_game.csv`和`advanced_*.csv`采用11次轮换测量。短时计时受系统负载影响，应同时看确定性的节点数与计时波动，不能只凭一次耗时判断优劣。`details.json`及两份拓展汇总JSON记录相关源码哈希，`details.json`还记录题目PDF的哈希。

补充实验中的精确缓存只用于未剪枝的Minimax，缓存的是相同局面下的未来收益差，`cache_hits`和`cache_entries`分别记录命中次数与额外存储规模。Alpha-beta剪枝返回的数值有时只是界，不能未经区分直接当作精确值缓存。A*的`deeper`、`fifo`、`shallower`只改变同`f`节点的访问顺序，不改变最优代价。高障碍密度下可达地图太少，图中不足10张的档位不画，以免把无路搜索误解成寻路提速。

缓存深度对照复用相同40组价值配置。地图对照另记录`map_sha256`与`distinct_maps`，其中无障碍迷宫的40次重复只有1张不同地图，不计作40张独立地图。

多查询对照使用相同地图与起点，三方案预热后轮换运行11次。两种预处理方案每次独立运行反向Dijkstra建表，并计入同次总用时。`direct_lookup_batch`直接读取表中的最小总代价，适用于地图和终点固定、只需要代价的查询。`lookahead`排序估计对手下一回合的即时应对，只改变Alpha-beta的访问顺序。各方案的最优值均逐例核对，保留收益不明显或总用时更高的情况。

题目四样例的六点顺序为A、B、T1、T2、T3、T4。H=2时最佳首步为T2、收益差为-1；H=4时为T3、收益差为3。实验报告另行提交。
