"""Mini Netflix collaborative filtering application.
Run: python -m streamlit run netflix_app.py
Uses synthetic ratings in demo; upload MovieLens latest-small CSV files to test real historical ratings.
"""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Mini Netflix 推荐系统', page_icon='🎬', layout='wide')
st.title('🎬 Mini Netflix：协同过滤推荐系统')
st.caption('从零实现矩阵分解、梯度下降与电影评分预测。教学演示数据与真实 MovieLens 数据严格区分。')

DEMO_MOVIES = [
    'Titanic', 'The Notebook', 'La La Land', 'Pride & Prejudice',
    'Inception', 'Interstellar', 'The Matrix', 'Avatar',
    'Toy Story', 'Finding Nemo', 'Inside Out', 'Coco',
    'The Dark Knight', 'Iron Man', 'Spider-Man', 'Avengers',
    'The Shawshank Redemption', 'Forrest Gump', 'The Godfather', 'Whiplash'
]


def demo_data():
    rng = np.random.default_rng(42)
    n_movies, n_users, n_features = 20, 10, 4
    true_movie_features = rng.normal(size=(n_movies, n_features))
    true_user_tastes = rng.normal(size=(n_users, n_features))
    raw = 3.2 + 0.7 * (true_movie_features @ true_user_tastes.T) + rng.normal(0, 0.2, (n_movies, n_users))
    all_ratings = np.clip(np.rint(raw), 1, 5)
    R = rng.random((n_movies, n_users)) < 0.65
    R[:, 0] = False
    indices = [0, 1, 4, 5, 8, 10, 12, 16]
    R[indices, 0] = True
    Y = np.where(R, all_ratings, 0.0)
    Y[indices, 0] = [5, 4, 2, 5, 4, 2, 5, 3]
    return DEMO_MOVIES, Y, R


def movie_lens_data(movie_csv, ratings_csv):
    movies_df = pd.read_csv(movie_csv)
    ratings_df = pd.read_csv(ratings_csv)
    if not {'movieId', 'title'}.issubset(movies_df.columns):
        raise ValueError('movies.csv 必须包含 movieId 和 title')
    if not {'userId', 'movieId', 'rating'}.issubset(ratings_df.columns):
        raise ValueError('ratings.csv 必须包含 userId、movieId、rating')
    data = ratings_df.merge(movies_df[['movieId', 'title']], on='movieId', how='inner')
    # Select widely rated movies, then users with best coverage of those movies.
    top_ids = data.groupby('movieId').size().nlargest(20).index
    data = data[data['movieId'].isin(top_ids)]
    top_users = data.groupby('userId').size().nlargest(10).index
    data = data[data['userId'].isin(top_users)]
    if data.empty:
        raise ValueError('没有足够的有效评分')
    titles = movies_df.set_index('movieId').loc[top_ids, 'title'].tolist()
    movie_ids = list(top_ids)
    user_ids = list(top_users)
    m_to_ix = {v: i for i, v in enumerate(movie_ids)}
    u_to_ix = {v: j for j, v in enumerate(user_ids)}
    Y = np.zeros((len(movie_ids), len(user_ids)), dtype=float)
    R = np.zeros_like(Y, dtype=bool)
    for row in data.itertuples(index=False):
        i, j = m_to_ix[row.movieId], u_to_ix[row.userId]
        Y[i, j] = float(row.rating)
        R[i, j] = True
    return titles, Y, R


def fit_cf(Y, R, k=4, lam=0.1, lr=0.01, epochs=1800, seed=7):
    n_movies, n_users = Y.shape
    counts = R.sum(axis=1, keepdims=True)
    global_mean = float(Y[R].mean()) if R.any() else 3.0
    means = np.divide(Y.sum(axis=1, keepdims=True), counts,
                      out=np.full((n_movies, 1), global_mean), where=counts != 0)
    Ynorm = np.where(R, Y - means, 0.0)
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.1, size=(n_movies, k))
    W = rng.normal(scale=0.1, size=(n_users, k))
    b = np.zeros((1, n_users))
    costs = []
    for epoch in range(epochs + 1):
        pred = X @ W.T + b
        err = (pred - Ynorm) * R
        cost = .5 * np.sum(err**2) + .5 * lam * (np.sum(X**2) + np.sum(W**2))
        if epoch % 50 == 0:
            costs.append({'epoch': epoch, 'cost': float(cost)})
        if epoch == epochs:
            break
        dX = err @ W + lam * X
        dW = err.T @ X + lam * W
        db = np.sum(err, axis=0, keepdims=True)
        X -= lr * dX
        W -= lr * dW
        b -= lr * db
    return np.clip(X @ W.T + b + means, 1, 5), pd.DataFrame(costs)


mode = st.radio('选择数据来源', ['教学模拟数据（10用户×20电影）', '真实 MovieLens 评分数据（上传 CSV）'], horizontal=True)
if mode.startswith('教学'):
    titles, Y, R = demo_data()
    st.info('这里的电影名称是真实的，但其他用户的评分由程序模拟生成。推荐结果只能用于学习算法。')
else:
    st.markdown('从 [MovieLens latest-small](https://grouplens.org/datasets/movielens/latest/) 下载并解压，然后上传 `movies.csv` 和 `ratings.csv`。程序选出20部评分最多的电影及其中评分覆盖最多的10名用户。')
    up1 = st.file_uploader('movies.csv', type='csv', key='movies')
    up2 = st.file_uploader('ratings.csv', type='csv', key='ratings')
    if up1 is None or up2 is None:
        st.stop()
    try:
        titles, Y, R = movie_lens_data(up1, up2)
    except Exception as exc:
        st.error(f'CSV 无法读取：{exc}')
        st.stop()
    st.success(f'已读取真实数据，选出 {len(titles)} 部电影、{Y.shape[1]} 个用户，已知评分 {R.sum()} 条。')

st.subheader('① 查看历史评分 Y 和评分记录 R')
users = [f'User {i+1}' for i in range(Y.shape[1])]
with st.expander('显示 Y、R 矩阵'):
    a, b_col = st.columns(2)
    a.markdown('**Y：已知评分，0 表示缺失**')
    a.dataframe(pd.DataFrame(Y, index=titles, columns=users))
    b_col.markdown('**R：1 有评分，0 无评分**')
    b_col.dataframe(pd.DataFrame(R.astype(int), index=titles, columns=users))

st.subheader('② 自己给电影评分')
st.caption('你是一个新的用户。可以只给你看过的电影打分，未评分保持“未看过/不评分”。其他10人的历史评分用于学习电影之间的关系。')
ratings = {}
cols = st.columns(2)
for i, title in enumerate(titles):
    with cols[i % 2]:
        choice = st.selectbox(title, ['未评分', '1', '2', '3', '4', '5'], key=f'{mode}_{i}')
        if choice != '未评分':
            ratings[i] = float(choice)

st.subheader('③ 训练并生成你的推荐')
lam = st.slider('正则化 λ', 0.0, 1.0, 0.1, 0.05)
epochs = st.slider('训练次数', 200, 3000, 1800, 200)
if st.button('训练模型并推荐电影', type='primary'):
    if len(ratings) < 3:
        st.warning('请至少给3部电影评分，才能尝试学习你的偏好。建议评分5部以上。')
    elif len(ratings) == len(titles):
        st.warning('所有电影都已评分，没有未评分电影可以推荐。')
    else:
        # Add the real visitor as an extra column and refit the same CF model.
        new_y = np.zeros((len(titles), 1))
        new_r = np.zeros((len(titles), 1), dtype=bool)
        for i, rating in ratings.items():
            new_y[i, 0], new_r[i, 0] = rating, True
        y_train = np.column_stack([Y, new_y])
        r_train = np.column_stack([R, new_r])
        with st.spinner('正在进行矩阵分解和梯度下降…'):
            scores, costs = fit_cf(y_train, r_train, lam=lam, epochs=epochs)
        st.line_chart(costs.set_index('epoch')['cost'])
        indices = np.flatnonzero(~new_r[:, 0])
        rank = sorted(indices, key=lambda i: scores[i, -1], reverse=True)
        results = pd.DataFrame([
            {'电影': titles[i], '预测评分（1-5）': round(float(scores[i, -1]), 2)} for i in rank[:5]
        ])
        st.markdown('**给你的 Top 5 推荐**')
        st.dataframe(results, hide_index=True, use_container_width=True)
        st.caption('这些是模型预测分，不是你实际给出的评分。小样本结果可能不可靠。')

st.divider()
st.subheader('④ 怎样真正检验预测是否准确？')
st.markdown('把部分**已知历史评分**先隐藏起来：用剩余评分训练，然后预测被隐藏的评分，比较预测值和真实评分。')
if st.button('进行留出测试（真实数据模式更有意义）'):
    observed = np.argwhere(R)
    if len(observed) < 12:
        st.warning('已知评分过少，无法测试。')
    else:
        rng = np.random.default_rng(2026)
        # Avoid removing the only known rating for any movie or user.
        inds = rng.permutation(len(observed))
        train_r = R.copy()
        held = []
        for ix in inds:
            i, j = observed[ix]
            if train_r[i].sum() > 1 and train_r[:, j].sum() > 1:
                train_r[i, j] = False
                held.append((i, j))
            if len(held) >= max(1, int(0.2 * len(observed))):
                break
        train_y = np.where(train_r, Y, 0)
        predictions, costs = fit_cf(train_y, train_r, lam=lam, epochs=epochs)
        actual = np.array([Y[i, j] for i, j in held])
        predicted = np.array([predictions[i, j] for i, j in held])
        train_counts = train_r.sum(axis=1)
        baseline_movie_means = np.divide(train_y.sum(axis=1), train_counts,
                                         out=np.full(len(titles), Y[R].mean()), where=train_counts != 0)
        baseline = np.array([baseline_movie_means[i] for i, _ in held])
        rmse = lambda a, b: float(np.sqrt(np.mean((a - b)**2)))
        c1, c2, c3 = st.columns(3)
        c1.metric('测试评分数', str(len(held)))
        c2.metric('协同过滤 RMSE', f'{rmse(predicted, actual):.3f}')
        c3.metric('只用电影平均分 RMSE', f'{rmse(baseline, actual):.3f}')
        st.caption('RMSE 越小越好。若协同过滤不优于简单基线，表示该小数据集或模型设置下没有体现更好的预测效果。')
        st.dataframe(pd.DataFrame({
            '电影': [titles[i] for i, j in held],
            '用户': [users[j] for i, j in held],
            '真实评分': actual,
            '预测评分': np.round(predicted, 2)
        }).head(20), hide_index=True)