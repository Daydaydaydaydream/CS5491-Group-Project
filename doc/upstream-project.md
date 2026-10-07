# 上游 NSC 项目与本地课程项目的关系

## 上游项目

- 仓库：<https://github.com/Optima-CityU/neural-spectral-capacity>
- 本地参考版本：`7f003cbfc650a69b4456d85602357f4d115c3d69`（2026-09-26）
- 论文：Chenyu Zhu、Ruoyu Zhao、Zhichao Lu，*Neural Spectral Capacity: Measuring and Designing Architectures from Network Specification Alone*，NeurIPS 2026 poster。
- 核心实现：`nsc/nsc_utils.py` 中的 `psi_mp` 与 `xavier_sigma`。
- 上游工作还包含五个架构族的排序评估、Transformer-XL 与 AutoFormer 搜索、LoNAS 剪枝，以及 MP 收敛和聚合方式等附录分析。

## 本地项目如何复用

本地 `nsc/nsc_utils.py` 与上游参考文件一致，MIT 许可证随文件保留；完整上游 checkout 在 `references/neural-spectral-capacity/`，当前检出提交为 `7f003cbfc650a69b4456d85602357f4d115c3d69`。课程代码显式管理初始化标准差，以匹配 course brief 固定 `s = 0.02` 的评估约定。上游论文常用 Xavier 标准差，调用 `xavier_sigma(m, n)` 可取得该约定。课程核心项目只需 NumPy 与 SciPy；上游搜索和剪枝模块所需的深度学习框架、模型、数据集及权重不属于本课程起步环境。

本项目的研究改动是基于前序 FFN 残差负载调节后续层贡献。上游附录已经比较 sum、mean、harmonic、geometric、min 等对层顺序不敏感的聚合函数，因此本项目的差异需要表述为“层序依赖的残差上下文加权”。评估中应增加 harmonic/min 聚合对照，避免把一般非加性聚合误报为新贡献。

## 可用数据与限制

上游 README 列出了 FlexiBERT、GPT-2/LiteTransformerSearch、AutoFormer、NATS-Bench、MobileNetV3、Transformer-XL 和 LoNAS 等外部资源。上游代码仓库并不随代码一同提供这些数据、检查点或训练产物。课程正式实验仍以课程 starter 的固定架构面板和开发/最终划分为准；外部 GPT-2 面板如用于接入调试，必须标为临时调试数据。

## 引用

```bibtex
@inproceedings{zhu2026nsc,
  title={Neural Spectral Capacity: Measuring and Designing Architectures from Network Specification Alone},
  author={Zhu, Chenyu and Zhao, Ruoyu and Lu, Zhichao},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026}
}
```
