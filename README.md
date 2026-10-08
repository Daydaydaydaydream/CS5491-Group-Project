# CS5491 神经谱容量项目

CS5491 课程组项目：研究 Neural Spectral Capacity（NSC）能否通过引入 Transformer 残差路径中的层序信息，改进架构排序与固定资源预算下的架构选择。

本仓库以 [Optima-CityU/neural-spectral-capacity](https://github.com/Optima-CityU/neural-spectral-capacity) 为方法基线。核心上游实现 `src/cs5491_nsc/nsc_utils.py` 及对应 MIT 许可证已随本仓库提供；完整上游 checkout 位于 `references/neural-spectral-capacity/`（独立 Git 仓库，不入库）。课程 starter 的压缩包与解压版归档在 `course/starter/`。

本项目是课程研究工程，不是上游官方仓库。

## 研究问题

标准 NSC 根据架构规格计算每个权重矩阵的 MP 谱容量，再对选定矩阵求和。由于求和不依赖层顺序，具有相同层配置但 FFN 宽度分配顺序不同的模型会得到相同分数。本项目研究一个残差上下文启发式：用前序 FFN 分支的估计残差负载，调整后续 FFN 容量的贡献。

我们分别检查两个问题：

1. 评分能否对受控的层顺序变化作出响应？
2. 这种敏感性是否改善课程归档架构上的排序或预算选择？

受控示例中的分数变化只说明评分对顺序敏感，不等同于预测性能提升。

## 方法

上游方法对矩阵容量采用

\[
\psi_{MP}(m,n,s)=N\int_{\lambda_-}^{\lambda_+}\log(1+Ms^2\lambda)f_{MP}(\lambda;\gamma)\,d\lambda,
\quad N=\min(m,n),\ M=\max(m,n),\ \gamma=N/M.
\]

课程基线按每层注意力头和 FFN 投影分组，固定初始化标准差 `s = 0.02`：

```text
A_l = 3 H_l psi_MP(d, d / H_l, s) + psi_MP(d, d, s)
F_l = 2 psi_MP(d, f_l, s)
```

项目评分加入 FFN 负载代理 `q_l = d f_l s^4 / 2`：

```text
S(alpha) = sum_l [A_l + F_l / (1 + alpha * sum_{j<l} q_j)]
```

`alpha = 0` 是加性消融，对应课程基线的原始 NSC 求和。负载代理假设两层独立高斯线性映射、中间 ReLU、各向同性输入，并忽略归一化和可学习缩放；它是项目中的待检验启发式，不是上游论文提出或证明的公式。

上游附录已研究 sum、mean、harmonic、geometric、min 等层间对称聚合。因此本项目的研究差异应限定为**权重由前序残差负载决定的层序依赖**，并在实验中纳入 harmonic/min 聚合作为对照。详见 [`docs/github-survey.md`](docs/github-survey.md)。

## 项目结构

```text
src/cs5491_nsc/          可安装的项目包：MP 工具与课程评分实现
tests/                   自动化测试（pytest）
examples/                受控 FFN 宽度换序示例
scripts/                 可复现的研究操作脚本
course/starter/          课程 starter 压缩包和原始评测程序（逐字保留，不参与格式化）
data/                    课程面板数据说明
results/                 实验结果记录约定
docs/                    课程主题、项目计划、立项书与调研文档
references/              上游项目参考与版本说明
```

## 环境与运行

需要 Python 3.9 或更高版本（开发环境使用 3.12）。核心代码只依赖 NumPy 和 SciPy，无需 GPU 或 LLM API。

```bash
python -m venv .venv
source .venv/bin/activate# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

也可以使用 Makefile（`PYTHON` 默认为 `python`）：

```bash
make install
make check      # lint + 格式检查 + 测试，与 CI 一致
make demo       # 受控 FFN 换序示例
make starter    # 课程 starter 自检
```

### 受控换序示例

```bash
python -m examples.reorder_demo
```

预期输出（原始 NSC 对两种顺序完全相同，残差上下文评分则不同）：

```text
Original additive NSC:
  [256, 512]: 121.820170
  [512, 256]: 121.820170
Residual-context score (alpha=1):
  [256, 512]: 121.697831
  [512, 256]: 121.692809
```

### 课程 starter

starter 自带评测程序，只依赖 Python 标准库。运行自检与开发集基线：

```bash
cd course/starter/nsc-starter
python3 check_starter.py
python3 evaluate.py --split development --output development_results.json
```

### 上游 NSC 通用接口

```python
from cs5491_nsc import psi_mp, xavier_sigma

sigma = xavier_sigma(512, 2048)
ffn_capacity = 2 * psi_mp(2048, 512, sigma)
```

课程比较固定使用 `s = 0.02`，评分接口已据此提供：

```python
from cs5491_nsc import TransformerLayerSpec, original_nsc_score, score_transformer

layers = [
    TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=256),
    TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=512),
]
baseline = original_nsc_score(layers)
proposed = score_transformer(layers, alpha=1.0)
```

## 开发规范

- `make lint` 与 `make format` 使用 Ruff。`course/starter/` 为逐字归档的课程材料，已在配置中排除，不参与检查与格式化。
- 提交前安装 pre-commit 钩子：`python -m pip install pre-commit && pre-commit install`。
- `src/cs5491_nsc/nsc_utils.py` 为上游 vendored 实现，修改前先确认是否应在上游仓库进行。
- CI（`.github/workflows/ci.yml`）在 Python 3.10 与 3.12 上运行 lint、测试与 starter 自检。

## 课程实验流程

1. 运行课程 starter 自检并复现原始评分、资源计数和开发集基线；将研究原型适配到 starter 接口。
2. 对比原始 NSC、位置加权教学示例、残差上下文评分、加性消融、参数量、参考 FLOPs 及 harmonic/min 聚合。
3. 报告 Spearman ρ、Kendall τb、并列数，以及同宽且参数量相差不超过 10% 的比较数量。
4. 在 10M、20M、30M、50M 参数上限下评估 `k = 1, 3` 的最佳困惑度和相对遗憾，并对比均匀随机选择。
5. 只用开发集选择 `alpha` 和方法设置；冻结方案后再运行最终集。保存精确命令、配置与提交标识。

课程 starter 和固定 200 架构面板已归档在 `course/starter/nsc-starter/`。在运行 starter 自检、完成评分接口适配并复现开发集基线前，不报告正式课程评估结果。上游仓库列出的其他基准数据、模型权重和训练资源仍属外部资源，并未随本项目提供。课程归档每个架构只有一个结果，因此本项目不估计训练种子不确定性。

## 参考资料

文档索引见 [`docs/README.md`](docs/README.md)。

- 课程项目主题说明：[`docs/topic-neural-spectral-capacity.pdf`](docs/topic-neural-spectral-capacity.pdf)
- 项目规划：[`docs/project-plan.md`](docs/project-plan.md)
- 立项书框架：[`docs/project-proposal-framework-cn.md`](docs/project-proposal-framework-cn.md)
- 类似项目调研：[`docs/github-survey.md`](docs/github-survey.md)
- 上游项目关系：[`docs/upstream-project.md`](docs/upstream-project.md)
- Zhu, Chenyu, Ruoyu Zhao, and Zhichao Lu. “Neural Spectral Capacity: Measuring and Designing Architectures from Network Specification Alone.” NeurIPS 2026 poster. [论文与项目仓库](https://github.com/Optima-CityU/neural-spectral-capacity) · [上游 `nsc_utils.py`](https://github.com/Optima-CityU/neural-spectral-capacity/blob/7f003cbfc650a69b4456d85602357f4d115c3d69/nsc/nsc_utils.py)

引用论文时可使用：

```bibtex
@inproceedings{zhu2026nsc,
  title={Neural Spectral Capacity: Measuring and Designing Architectures from Network Specification Alone},
  author={Zhu, Chenyu and Zhao, Ruoyu and Lu, Zhichao},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026}
}
```