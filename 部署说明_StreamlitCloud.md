# Week7 固定公网链接部署（Streamlit Cloud）

## 目标
把本项目部署为长期固定网址（不依赖你本地电脑开机）。

## 步骤
1. 新建 GitHub 仓库（例如：`week7-lm-platform`）。
2. 把 `周四week7` 文件夹内文件上传到仓库根目录。
3. 打开 [https://share.streamlit.io/](https://share.streamlit.io/) 并登录 GitHub。
4. 点击 `New app`，选择你的仓库与分支。
5. `Main file path` 填：`app.py`，然后点击 `Deploy`。

## 部署完成后
- 你会得到一个固定链接（形如 `https://xxx.streamlit.app`）。
- 这个链接长期可用（除非你删除应用或仓库）。

## 说明
- 本项目依赖已经放在 `requirements.txt`，可直接被 Streamlit Cloud 识别安装。
- 若后续更新代码，只需 push 到 GitHub，线上应用会自动更新。
