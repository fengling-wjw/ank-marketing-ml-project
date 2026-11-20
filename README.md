# 乳腺癌数据分析：概率校准与预算约束

使用 scikit-learn 内置 Wisconsin Diagnostic Breast Cancer 数据（569 条、30 个特征），比较逻辑回归与随机森林，开展概率校准、阈值选择、置换重要性和分组分析。仅供教学，不可用于临床。

参考 Nasir 等 (2026), *Marketing analytics in banking 4.0*, PLOS ONE, doi:10.1371/journal.pone.0348767，借鉴集成学习、概率校准和可解释分析的思路。原论文为银行营销；本项目迁移到肿瘤数据，采用置换重要性并加入预算阈值搜索，并非完整复现，不声称提出了新算法。

运行 `pip install -r requirements.txt` 后执行 `python src/train_cancer.py`。结果写入 outputs/corrected。种子为 2026，开发集与测试集比例 80:20；开发集再分为拟合集和验证集。校准仅使用拟合集；验证集用于模型和阈值选择，存在选择偏差风险。

测试集 ROC-AUC 0.9934，平均精确率 0.9897，召回率 0.8571，精确率 1.0000；阈值 0.843。验证集标记率约束为 30%，测试集实际 31.58%，无法保证新数据满足预算。效用为教学假设。

outputs/corrected 包括验证集模型比较、阈值、置换重要性、分组与结果摘要。旧版 src/train.py 是银行营销探索，不用于此实验。

## 扩展实验与图表

运行 `python src/data_audit.py` 保存数据质量、30 个特征的描述统计和分类别均值。运行 `python src/robustness.py` 进行 10 次分层随机划分，各次重新选择模型、校准与阈值；该实验随机森林使用 200 棵树，不同于主实验的 500 棵树。测试集可能相互重叠，因此不能当成独立外部验证。运行 `python src/plot_results.py` 使用已保存结果生成 PNG 图。

10 次验证的召回率范围为 0.4762–0.9286，其中 6 次测试标记率超过 30%。这反映单次结果的波动和阈值预算约束的局限。所有输出均为对应脚本运行结果，实验无深度学习 epoch 训练损失轨迹；模型比较和逐次划分指标即该项目的过程记录。

## 补充实验复现包

`reproducibility_2026/` 包含独立可运行的代码副本、数据审计、主实验、10 次重复划分结果、图表、测试、实际运行日志和本次运行的依赖版本。该目录为 2026-11-20 在本机重新运行生成，不冒充原会话中未取回的压缩包。

```powershell
cd reproducibility_2026
python -m pip install -r requirements-lock.txt
python src/data_audit.py
python src/train_cancer.py
python src/robustness.py
python src/plot_results.py
python -m unittest discover -s tests -v
```

主实验输出位于 `outputs/corrected/`，10 次实验位于 `outputs/robustness/`，日志与结果一致性核对记录位于 `logs/`。Python 版本参见 `environment.json`。本次提交不包含早期未修正结果与旧版 Word 报告。
