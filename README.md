# 可解释与成本敏感的银行营销响应预测

本项目以 UCI Bank Marketing 公开数据预测定期存款认购，复现集成学习、概率校准与可解释分析，并新增：删除通话后泄露变量 `duration`、在联系预算约束下选阈值、年龄分组稳健性检验。

参考论文：Nasir et al. (2026), *Marketing analytics in banking 4.0: A two-stage explainable AI framework for high-accuracy and well-calibrated predictions*, PLOS ONE. https://doi.org/10.1371/journal.pone.0348767

## 运行

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py --data-url https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
```

运行后所有真实指标、图和日志均保存至 `outputs/`。原始数据与模型文件不提交。
