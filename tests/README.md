# Tests

`cs5491_nsc` 包的自动化测试，运行：

```bash
make test# 或 python -m pytest
```

当前覆盖两块：

- `test_nsc_utils.py`：MP 密度归一化与支撑集、`psi_mp` 的形状对称性与单调性、缓存一致性。
- `test_student_score.py`：项目核心契约——原始 NSC 对层序不变、残差上下文评分在 `alpha > 0` 时对层序敏感且在 `alpha = 0` 时退化为加性基线；以及输入校验与边界情况。

`conftest.py` 提供课程约定的 `init_std = 0.02` 与受控换序层规格 fixture。新增行为时在此补测试，并同步更新 `results/` 中的实验记录。