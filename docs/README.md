# 文档索引

本目录集中存放课程项目的说明性材料。数据与代码约定另见仓库根目录的 `README.md`。

| 文件 | 内容 | 用途 |
|---|---|---|
| [`topic-neural-spectral-capacity.pdf`](topic-neural-spectral-capacity.pdf) | 课程主题说明（官方原件） | 核对研究问题、数据划分与提交要求 |
| [`project-plan.md`](project-plan.md) | 项目规划与时间表 | 里程碑安排、本周优先事项、风险应对 |
| [`project-proposal-framework-cn.md`](project-proposal-framework-cn.md) | 一页立项书框架（中文） | 10-15 提交；按 A4 / Times New Roman 12 pt / 2.5 cm 页边距 / 单倍行距排版 |
| [`project-proposal.docx`](project-proposal.docx) | 立项书草稿（可编辑） | 在框架基础上填写完整；提交前转为 PDF |
| [`github-survey.md`](github-survey.md) | 类似项目调研（2026-10-07） | 确认官方开源实现、研究空白与最大风险 |
| [`upstream-project.md`](upstream-project.md) | 上游项目与本项目的关系 | 引用来源、MIT 许可归属、可用数据与限制 |

## 阅读顺序建议

1. 先读课程主题说明，确认任务边界与评分口径。
2. 再读调研文档，明确本项目与官方 NSC 实现差异化之处（层序依赖，而非一般非加性聚合）。
3. 动手前读项目规划，确认当前阶段的交付物。

## 约定

- 文档中的结论区分「已验证」与「待检验」。评分对层序敏感不等同于预测性能提升，两者分别报告。
- 正式实验结果不得在运行前填写。开发集用于方法选择，最终集只在冻结方案后评估。
- 引用上游论文时使用 `README.md` 中的 BibTeX 条目。