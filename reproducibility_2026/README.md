# 补充实验：代码、结果与运行记录

本目录为 2026-11-20 使用本地保留的实验脚本重新运行生成的独立复现包。

在本目录安装 `requirements-lock.txt` 中的依赖，然后依次运行：

```powershell
python src/data_audit.py
python src/train_cancer.py
python src/robustness.py
python src/plot_results.py
python -m unittest discover -s tests -v
```

- `src/`：数据审计、主实验（500 棵树）、重复划分（200 棵树）、绘图。
- `outputs/`：本次真实运行产生的统计、指标和图表。
- `logs/`：脚本标准输出和警告、测试记录、与保留结果的比较记录。
- `environment.json` 与 `requirements-lock.txt`：本次实际 Python、平台与依赖版本。

数据是 scikit-learn 内置的 Wisconsin Diagnostic Breast Cancer；恶性为正类。校准只使用拟合集，验证集选择模型与阈值；验证集的 30% 预算约束不保证测试集满足。10 次划分的测试样本存在重叠，不代表独立外部验证。本项目仅用于教学。
