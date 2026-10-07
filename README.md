# CS5491 神经谱容量项目

本仓库是 CS5491 课程项目的独立工程，研究 Neural Spectral Capacity（NSC）能否通过考虑 Transformer 残差路径中的层序信息，改进架构排序与固定资源预算下的架构选择。

项目以 [Optima-CityU/neural-spectral-capacity](https://github.com/Optima-CityU/neural-spectral-capacity) 为研究和方法基线。核心 `nsc/nsc_utils.py` 与对应 MIT 许可证已纳入本地评分模块；完整上游 checkout 位于 `references/neural-spectral-capacity/`。本仓库保留课程研究范围、代码和实验流程，不是上游官方仓库。

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

项目评分加入 FFN 残差负载代理 `q_l = d f_l s^4 / 2`：

```text
S(alpha) = sum_l [A_l + F_l / (1 + alpha * sum_{j<l} q_j)]
```

`alpha = 0` 是加性消融，对应课程基线的原始 NSC 求和。负载代理假设两层独立高斯线性映射、中间 ReLU、各向同性输入，并忽略归一化和可学习缩放；它是项目中的待检验启发式，不是上游论文提出或证明的公式。

上游附录已研究 sum、mean、harmonic、geometric、min 等层间对称聚合。因此，本项目的研究差异应限定为**权重由前序残差负载决定的层序依赖**，并在实验中纳入 harmonic/min 聚合作为对照。

## 项目结构

```text
nsc/                    MP 谱容量与初始化标准差
src/student_score.py    课程基线分组、残差上下文评分与消融
examples/               受控 FFN 宽度换序示例
data/                    课程面板数据说明
results/                实验结果记录约定
doc/                    课程主题、中文版立项书与上游调研
```

## 环境与运行

需要 Python 3.9 或更高版本。当前核心代码依赖 NumPy 和 SciPy：

```bash
python -m pip install -r requirements.txt
python -m examples.reorder_demo
```

上游 NSC 的通用初始化示例：

```python
from nsc.nsc_utils import psi_mp, xavier_sigma

sigma = xavier_sigma(512, 2048)
ffn_capacity = 2 * psi_mp(2048, 512, sigma)
```

课程比较固定使用 `s = 0.02`，评分接口已据此提供：

```python
from src.student_score import TransformerLayerSpec, original_nsc_score, score_transformer

layers = [
    TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=256),
    TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=512),
]
baseline = original_nsc_score(layers)
proposed = score_transformer(layers, alpha=1.0)
```

## 课程实验流程

1. 接入课程 starter，复现原始评分、资源计数和开发集基线。
2. 对比原始 NSC、位置加权教学示例、残差上下文评分、加性消融、参数量、参考 FLOPs 及 harmonic/min 聚合。
3. 报告 Spearman ρ、Kendall τb、并列数，以及同宽且参数量相差不超过 10% 的比较数量。
4. 在 10M、20M、30M、50M 参数上限下评估 `k = 1, 3` 的最佳困惑度和相对遗憾，并对比均匀随机选择。
5. 只用开发集选择 `alpha` 和方法设置；冻结方案后再运行最终集。保存精确命令、配置与提交标识。

课程 starter、固定 200 架构面板与评测脚本目前尚未放入本仓库。接入前不报告课程实验结果。上游仓库也将多个基准数据、模型权重和训练资源列为外部资源，不能将其视为已随源码提供。课程归档每个架构只有一个结果，因此本项目不估计训练种子不确定性。

## 参考资料

- Zhu, Chenyu, Ruoyu Zhao, and Zhichao Lu. “Neural Spectral Capacity: Measuring and Designing Architectures from Network Specification Alone.” NeurIPS 2026 poster. [论文与项目仓库](https://github.com/Optima-CityU/neural-spectral-capacity) · [上游 `nsc_utils.py`](https://github.com/Optima-CityU/neural-spectral-capacity/blob/7f003cbfc650a69b4456d85602357f4d115c3d69/nsc/nsc_utils.py)
- CS5491 课程项目主题说明：[`doc/topic-neural-spectral-capacity.pdf`](doc/topic-neural-spectral-capacity.pdf)
- 上游仓库的结构、基准和数据来源摘要：[`doc/upstream-project.md`](doc/upstream-project.md)

引用论文时可使用：

```bibtex
@inproceedings{zhu2026nsc,
  title={Neural Spectral Capacity: Measuring and Designing Architectures from Network Specification Alone},
  author={Zhu, Chenyu and Zhao, Ruoyu and Lu, Zhichao},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026}
}
```
