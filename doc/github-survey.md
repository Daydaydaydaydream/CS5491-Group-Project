# NSC 残差上下文项目 — 类似 GitHub 项目调研

调研日期：2026-10-07。目的：确认本项目在公开代码生态中的位置、可用素材，以及研究空白的成立性。

## 1. 结论摘要

本项目使用的 NSC（Neural Spectral Capacity）**已有官方开源实现**，作者与论文一致（City University of Hong Kong，Chenyu Zhu、Ruoyu Zhao、Zhichao Lu，NeurIPS 2026 poster）。这直接决定了项目的定位：

- **基线可完整复现**：`nsc/nsc_utils.py` 中的 `psi_mp` 仅依赖 numpy/scipy，可直接引入作为 starter 缺失期间的对照基线。
- **研究空白成立**：官方仓库对**聚合函数、初始化方差、MP 有限尺寸收敛、ψ_MP 凹凸性**四项均做了消融，但**没有任何针对"层顺序 / 位置加权"的实现**。本项目的残差上下文加权属于官方未覆盖的维度。
- **风险**：官方附录 `ablation_aggregation.py` 已测试 `harmonic-mean/layer`、`min/layer` 等**非加性聚合**。本项目的 `F_l / (1 + α·Σ_{j<l} q_j)` 同样是非加性加权，**必须与官方这批非加性聚合做区分**，否则会被质疑重复官方消融。

## 2. 直接基础：官方 NSC 仓库

| 项目 | 内容 |
|---|---|
| 仓库 | `https://github.com/Optima-CityU/neural-spectral-capacity` |
| 许可 | MIT |
| 论文 | arXiv:2609.23087，NeurIPS 2026（poster） |
| 规模 | 5 star，0 fork，更新至 2026-09-26 |
| 核心文件 | `nsc/nsc_utils.py`（49 行，`psi_mp` / `xavier_sigma`，带缓存） |

方法核心（与本项目 README 描述一致）：对权重矩阵 $W \in \mathbb{R}^{m \times n}$，

$$\psi(W) = \ln\det(I + W^{\top}W) = \sum_i \ln(1 + \sigma_i^2)$$

随机初始化下由 Marchenko–Pastur 定律给出闭式 $\psi_{\mathrm{MP}}(m, n, s)$，仅依赖形状与初始化方差。NSC 为全架构上所有权重矩阵的 $\psi_{\mathrm{MP}}$ 之和（注意力按 head 分解），因此**逐层可加**，最大化问题退化为有界背包，由 NSC-DP 精确求解。

### 官方仓库目录（已克隆核实）

```
nsc/               psi_mp 参考实现
ranking/           五个架构族上的 NSC 评分与基线（#FLOPs、W-PCA、ZeroLM、SNIP、GradNorm、SynFlow）
search/txl/        Transformer-XL 上的 NSC-DP、EA 基线、WikiText-103 训练
search/autoformer/ AutoFormer 上的 NSC-DP
pruning_lonas/     LLaMA-7B 剪枝：NSC-DP、proxy+GA 基线、Avg-8 评测
appendix/          聚合方式、MP 有限尺寸收敛、初始化方差鲁棒性、psi_MP 凹凸性
```

### 官方已报告的关键数值

FlexiBERT 500 架构（GLUE）排序质量：

| 方法 | τ | 10%-配对 τ | 500 架构耗时 |
|---|:---:|:---:|:---:|
| NSC | 0.695 | 0.505 | 2 ms (CPU) |
| #Params | 0.485 | 0.082 | — |
| #FLOPs | 0.552 | 0.329 | 2 ms (CPU) |
| W-PCA | 0.635 | 0.417 | 88 s (A100) |
| ZeroLM | 0.527 | 0.355 | 63 s (A100) |
| SNIP | 0.289 | 0.237 | 73 s (A100) |
| GradNorm | 0.171 | 0.164 | 74 s (A100) |

搜索侧：Transformer-XL / WikiText-103 上 NSC-DP 得 23.087 PPL（2.0 s），优于人工设计的 TXL Base 23.279（38M）与各代理基线。

## 3. 官方消融覆盖范围（对本项目的约束）

`appendix/` 目录四个脚本的对比维度：

| 脚本 | 消融维度 | 与本项目关系 |
|---|---|---|
| `ablation_aggregation.py` | sum / mean / harmonic / geometric / min / product / sum-of-log / global-bottleneck | **最需警惕**：已覆盖层间非加性聚合 |
| `ablation_aggregation_specfaithful.py` | 同上，spec-faithful 协议（FlexiBERT + GPT-2） | 同上 |
| `ablation_init_variance.py` | Xavier / Kaiming / TruncNorm 0.02 | 间接相关（本项目 `s` 默认 0.02） |
| `ablation_mp_convergence.py` | MP 定律有限尺寸收敛 | 数值基线参考 |
| `ablation_min_generalization.py` | min 聚合的泛化性 | 备查 |
| `plot_psi_concavity.py` | $\psi_{\mathrm{MP}}$ 凹凸性 | 与 DP 正确性相关 |

**判定**：本项目 `residual_context_score` 的差异化点不在"非加性聚合"本身，而在于**聚合权重由前序残差更新的累积负载决定，即显式引入层序（layer index）依赖**。官方所有消融都是对同层集合做对称聚合函数，**与层序无关**。这一定位必须在报告的 related work / 差异化论证中明确写出。

## 4. 外部依赖仓库（NSC 论文的实验素材来源）

官方 README 明确列出外部资源，均为公开仓库，可作为本项目的数据与基线来源：

| 仓库 | Star | 用途 |
|---|:---:|---|
| `microsoft/archai` | 487 | LiteTransformerSearch；`gpt2_benchmark.json`（GPT-2 架构基准，**最贴近本项目 200 架构 decoder 面板的形式**） |
| `mit-han-lab/once-for-all` | 1955 | OFA MobileNetV3 精度表 |
| `D-X-Y/NATS-Bench` | 195 | 规模搜索空间（NATS-Bench-SSS） |
| `jha-lab/txf_design-space` | 9 | FlexiBERT 代码 + `BERT_benchmark.json`（500 架构） |
| `IntelLabs/Hardware-Aware-Automated-Machine-Learning` | 77 | LoNAS LLaMA-7B 八项常识推理适配器 |
| `kimiyoung/transformer-xl` | — | Transformer-XL 基线 + WikiText-103 |
| `microsoft/Cream/AutoFormer` | — | AutoFormer 超网与检查点 |

**特别提示**：`gpt2_benchmark.json`（GPT-2 架构基准）在本项目 200 架构 decoder 面板缺失的情况下，是**唯一可立即获得、且结构同构的替代数据**。官方 `ablation_aggregation.py` 中的 `compute_gpt2_layers()` 已展示了完整解析路径：读取 `d_model` / `n_layer` / `n_head` / `d_inner`（支持逐层列表），按 head 分解注意力、按 up/down 投影分解 FFN。可作为自建 adapter 的参考实现。

## 5. 邻近研究方向（无直接代码可复用，但需在 related work 引用）

| 工作 | 与本项目的关系 |
|---|---|
| **LiteTransformerSearch**（NeurIPS 2022） | 立场相反的强基线：主张 decoder 参数量与困惑度高度秩相关（Spearman 0.97–0.99），**与架构拓扑无关**。这正是 NSC 论文反驳的对象。NSC 论文的关键反驳是"参数相同但拓扑不同"的细粒度配对（10% 配对 τ 从 0.082 提升到 0.505）。本项目应把 LTS 列为**必须对比的基线**，且其"拓扑无关"论断正是残差上下文加权可能突破的方向。 |
| **Variable-Width Transformers**（arXiv:2606.18246, 2026-06） | 实证支持非均匀宽度分配：×-形（bowtie）Transformer 在参数匹配下持续优于均匀基线，FLOPs 降 22%，KV cache 降 15%；并分析残差流表示的定性差异。**为本项目"顺序/宽度分配影响性能"提供正面经验证据**，且其"残差流表示不同"的机制描述与本项目的残差上下文动机高度呼应。 |
| **Residual Stream Duality**（arXiv:2603.16039, 2026） | 综述残差路径的架构设计空间（ELC-BERT、DenseFormer、Vertical Attention、Hyper-Connections、Deep Delta Learning）。为"残差路径参与表示而非仅是优化管道"提供理论依据，可用于论证加权项的合理性。 |
| **LoNAS / LLM 剪枝** | 层/宽度分配的相邻问题，官方 NSC-DP 已在该场景验证。 |

## 6. 对项目的直接影响

### 6.1 立刻可做

1. **引入官方 `psi_mp` 作为基线**：仅需 numpy/scipy，不依赖 starter 归档。在 starter 到位前即可完成 NSC 基线复现与受控重排示例。
2. **自建 200 架构面板的临时替代**：拉取 `gpt2_benchmark.json`，用官方 `compute_gpt2_layers()` 的解析方式生成 per-layer $\psi_{\mathrm{MP}}$，作为方法调试数据。**注意**：此数据仅供开发调试，正式评估仍须使用课程 starter 的固定 100/100 划分。
3. **差异化论证写入报告**：明确本项目引入的是**层序依赖**，与官方的同层集合聚合函数正交。

### 6.2 需在报告中说明的风险

| 风险 | 应对 |
|---|---|
| 与官方 `ablation_aggregation.py` 的非加性聚合重叠 | 明确区分：官方为同层集合的对称聚合，本项为层序加权。补一组"同输入下对比 harmonic/min 聚合"的实验 |
| `q_l = d·f_l·s^4/2` 假设过强（各向同性输入、忽略归一化与可学习缩放） | 报告已明确其为待检验启发式。补 `s` 的敏感性分析（可复用官方 `ablation_init_variance.py` 的三约定框架，但需区分：官方是改变**全局** σ，本项目需改变**层间相对**权重） |
| 官方数据集为公开基准，本项目为课程归档面板 | 结论仅适用于该固定面板，泛化性不做外推（报告已声明） |
| 无 seed 级重复，单架构单结果 | 不估计训练随机性；用 tie 数与配对 τ 缓解 |

### 6.3 与 starter 接入的衔接

课程 starter 很可能就是官方仓库的教学化裁剪版（官方 `ranking/` 下的 `eval_flexibert.py` / `compute_gpt2.py` 与 README 中"200 个 decoder Transformer 配置 + 100/100 划分"的描述高度吻合）。因此：

- 拿到 starter 后，应先核对其 `psi_mp` 与官方 `nsc_utils.py` 是否一致，确认基线可对齐。
- 项目的 `src/student_score.py` 已设计为接受外部传入容量序列，接口上可直接对接，无需重写。
- `alpha` 的调参须严格限制在开发集；官方代码中 `_PSI_CACHE` 为全局字典，跨划分复用不引入数据泄漏（纯形状函数），但应在报告中说明。

## 7. 引用清单

```bibtex
@inproceedings{zhu2026nsc,
  title={Neural Spectral Capacity: Measuring and Designing Architectures from Network Specification Alone},
  author={Zhu, Chenyu and Zhao, Ruoyu and Lu, Zhichao},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026}
}
```

- NSC 论文：https://arxiv.org/abs/2609.23087
- 官方代码：https://github.com/Optima-CityU/neural-spectral-capacity
- LTS：https://github.com/microsoft/archai （NeurIPS 2022, arXiv:2203.02094）
- Variable-Width Transformers：https://arxiv.org/abs/2606.18246
- Residual Stream Duality：https://arxiv.org/abs/2603.16039
