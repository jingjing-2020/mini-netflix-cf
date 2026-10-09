# 🎬 Mini Netflix — Collaborative Filtering from Scratch

[中文说明](#中文快速开始) | [English](#english-quick-start)

An educational movie recommender built with **NumPy matrix factorization and gradient descent**, with an interactive **Streamlit** interface. It supports a 10-user × 20-movie **synthetic teaching demo**, optional real **MovieLens** ratings, personal 1–5 star ratings, Top 5 suggestions, a training-loss curve, and a small holdout RMSE comparison.

> **Important:** This is an educational prototype, not Netflix's code or an official Netflix product. Demo-user ratings are artificially generated, not real viewers. Results from a 20×10 selection are too small/dense to establish production recommendation quality. No recommendation accuracy is guaranteed.

## 中文快速开始

### 在自己电脑运行

安装 Python 3.10+。在此项目文件夹打开终端：

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run netflix_app.py
```

打开终端输出的本地地址（通常为 <http://localhost:8501>）。

网页中：
1. 选择 **教学模拟数据**，或者上传 MovieLens 的 `movies.csv`、`ratings.csv`。
2. 展开 **Y、R 矩阵** 检查数据。
3. 给至少 3 部已经看过的电影打分，其他保持“未评分”。
4. 点击 **训练模型并推荐电影**，查看 Cost 曲线与 Top 5。
5. 点击 **进行留出测试**，比较协同过滤 RMSE 与电影平均分基线。

纯命令行学习版本：

```bash
python3 netflix_mini.py
```

### 在线部署：Streamlit Community Cloud

1. 新建一个 **public** GitHub repository，上传项目中的文件，确保 `netflix_app.py` 和 `requirements.txt` 在**仓库根目录**。
2. 打开 <https://share.streamlit.io/>，使用 GitHub 登录。
3. 选择 **Create app** / **Deploy a public app from GitHub**；选择你的仓库、`main` 分支和入口 `netflix_app.py`。
4. 点击 **Deploy**。云端会从 `requirements.txt` 安装依赖。
5. 部署成功后得到可公开访问的 `*.streamlit.app` 链接，复制给朋友即可。

如果显示 **Unable to deploy / code is not connected to a remote GitHub repository**：这通常是在本地 Streamlit 的部署入口尝试部署，但本地代码尚未关联已发布的 GitHub 分支。按以上步骤先上传到 GitHub，再从 Community Cloud 创建应用即可。

**数据隐私：** 本项目不会自行建立用户账户或持久化评分数据库；但上传到公共托管服务的文件会在其服务器处理。请勿上传私密数据或受限数据。MovieLens 原始数据无需提交到 GitHub；访客可以在应用中自行上传 CSV。公开分享前请阅读 MovieLens 数据集使用条款。

## English quick start

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run netflix_app.py
# CLI demonstration:
python3 netflix_mini.py
```

For the real-data mode, download [MovieLens latest-small](https://grouplens.org/datasets/movielens/latest/), extract it and upload `movies.csv` and `ratings.csv` to the running app. **Do not commit the MovieLens files** unless permitted by their license.

Deploy to [Streamlit Community Cloud](https://share.streamlit.io/) after uploading the repository to GitHub. Use `netflix_app.py` as the main file and the branch containing this code.

## How the algorithm works / 数学原理

Let Y be the movies×users observed rating matrix, R the boolean observed mask. For movie `i` and user `j`, the centered estimate is:

\[
\hat y_{ij}^{centered}= x_i\cdot w_j + b_j
\]

The objective is:

\[
J=\tfrac12\sum_{(i,j):R_{ij}=1}(x_i\cdot w_j+b_j-(Y_{ij}-\bar Y_i))^2+\tfrac{\lambda}{2}(\|X\|_F^2+\|W\|_F^2).
\]

Gradient descent learns X, W and b. Add movie means back to obtain predicted ratings, clipped to the 1–5 display scale. This is **matrix factorization**, a form of collaborative filtering, not a content-aware recommendation engine.

## Files

| File | Purpose |
|---|---|
| `netflix_mini.py` | Step-by-step NumPy CLI teaching example (synthetic data) |
| `netflix_app.py` | Streamlit interactive app and MovieLens upload mode |
| `requirements.txt` | Deployment/runtime dependencies |
| `tests/test_project.py` | Lightweight numerical and data-integrity tests |
| `.streamlit/config.toml` | Cloud-friendly UI settings |
| `.gitignore` | Ignore caches, environments, user data |

## Limitations & next steps

- The MovieLens option intentionally restricts selection to 20 popular films and 10 high-coverage users; this makes the matrix extremely dense and is not a realistic benchmark.
- Holdout errors are useful for basic verification, but a larger, sparser user/movie set and repeated train/test splits are needed for an honest evaluation.
- The demo's film titles are real, but its history is synthetic and does **not** reflect actual content features.
- The app does not save your input ratings between visits or keep personal profiles.
- For real use: increase the movie/user count, measure ranking metrics (Recall@K / NDCG@K), compare popularity baselines, handle cold-start cases, and add safe persistence only if necessary.

## Sharing

**For friends:** send the deployed Streamlit URL — no Python or GitHub account required to use the page. **For developers:** share the GitHub repository. A ChatGPT Skill is **not** required to publish an interactive website; a skill is more suited to reusable instructions/tools for an AI assistant.

## License

Project code is available under the MIT License (see `LICENSE`). Movie names and any optional MovieLens dataset have their own respective rights and terms.