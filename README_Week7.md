# Week 7 语言模型训练与对比分析平台

## 1. 项目说明

本作业使用 Streamlit 构建四模块交互式 Web 应用：

1. n-gram 统计语言模型与加一平滑（Laplace Smoothing）
2. 从零训练字符级 RNN/LSTM 语言模型
3. BERT（Masked LM）与 GPT-2（Causal LM）生成机制对比
4. 基于 GPT-2 的困惑度（Perplexity）计算

核心代码文件：`week7_lm_platform.py`

## 2. 环境安装

建议 Python 3.10+。

```bash
pip install -r requirements_week7.txt
```

## 3. 启动方式

```bash
streamlit run week7_lm_platform.py
```

如果 `streamlit` 不在 PATH 中，可用：

```bash
py -m streamlit run week7_lm_platform.py
```

或直接双击：`run_week7_app.bat`

## 4. 提交建议

建议提交以下文件：

- `week7_lm_platform.py`（核心代码）
- `Week7_实验报告.md`（实验报告）
- `Week7_实验报告_可视化.html`（可视化网页版报告）

另外可在运行 Streamlit 后使用浏览器“另存为”补充保存课堂演示页面快照。

