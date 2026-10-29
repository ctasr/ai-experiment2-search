# 实验二：搜索算法

实现迷宫和加权地形的A*寻路、博弈树与宝藏争夺的Minimax和Alpha-beta搜索，并比较启发函数、访问顺序、缓存和多次查询的效率。

## 运行

使用Python 3.10或更新版本，在PowerShell中依次执行：

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

macOS或Linux使用`python3 -m venv .venv`，后续将解释器路径替换为`.venv/bin/python`。算法只依赖标准库，绘图需要Matplotlib和已安装的中文字体，如宋体或Noto Sans CJK SC。

## 文件说明

```text
ai-experiment2-search/
├── search_algorithms.py       题一、题二的A*及反向最短路表
├── game_search.py             题三的Minimax和Alpha-beta
├── treasure_game.py           题四的距离矩阵、状态和博弈搜索
├── run_experiment.py          四题样例、基线、对照与参数扫描
├── extended_experiments.py    同分节点顺序、随机树和缓存对照
├── advanced_experiments.py    多查询与双步排序对照
├── plot_style.py              绘图字体设置
├── tests/                     32项样例、边界与独立算法核对测试
├── requirements.txt           绘图依赖及版本
├── results/                   原始数据、实验配置和结果汇总
└── figures/                   报告中的实验图
```

题一接口`shortest_maze_path(start, goal, grid)`返回最短坐标路径，无路返回`[]`。题二`minimum_terrain_cost(start, goal, grid)`返回最小总代价，无路返回`-1`。题三`alpha_beta_best_score(root)`返回根节点MAX的得分。题四`compute_distances()`计算六点距离矩阵，`treasure_game_answer(distances, depth)`返回`best_target`和`best_value`。

`results/`内各组文件的作用如下：

| 文件 | 作用 |
| --- | --- |
| `comparison.csv`、`details.json` | 主实验对照、搜索轨迹、配置和运行环境 |
| `distance_matrix*.csv`、`official_answers.json` | 两张地图的距离矩阵及题四样例答案 |
| `astar_sweep.csv`、`ordering_sweep.csv` | 启发函数和宝藏访问顺序的逐例对照 |
| `optimization_*.csv`、`optimization_summary.json` | 同分顺序、随机树和缓存实验的原始数据及汇总 |
| `advanced_*.csv`、`advanced_summary.json` | 多查询和双步排序的原始数据及汇总 |
| `deterministic_results.json`、`optimization_deterministic.json`、`advanced_deterministic.json` | 排除计时的结果，供重跑后核对 |

## 复现说明

随机种子、实验配置和源码哈希记录在`details.json`和两份`*_summary.json`中。重新运行实验会更新`results/`和`figures/`中的输出。

`expanded`统计A*的有效出队次数，包含终点。`visited_nodes`统计博弈递归访问次数，包含根、叶和缓存命中。主实验预热后轮换计时101次，缓存和多查询等计时对照轮换11次，记录中位数与四分位距。计时受系统负载影响，三份排除计时的JSON可用于比较重跑前后的最优值和节点数。
