"""10 users x 20 movies: collaborative filtering from scratch (NumPy only)."""
import numpy as np

rng = np.random.default_rng(42)
movies = [
    'Titanic', 'The Notebook', 'La La Land', 'Pride & Prejudice',
    'Inception', 'Interstellar', 'The Matrix', 'Avatar',
    'Toy Story', 'Finding Nemo', 'Inside Out', 'Coco',
    'The Dark Knight', 'Iron Man', 'Spider-Man', 'Avengers',
    'The Shawshank Redemption', 'Forrest Gump', 'The Godfather', 'Whiplash'
]
num_movies, num_users, num_features = 20, 10, 4

# Synthetic ratings for teaching: hidden tastes create coherent ratings.
true_movie_features = rng.normal(size=(num_movies, num_features))
true_user_tastes = rng.normal(size=(num_users, num_features))
raw = 3.2 + 0.7 * (true_movie_features @ true_user_tastes.T) + rng.normal(0, 0.2, (num_movies, num_users))
all_ratings = np.clip(np.rint(raw), 1, 5)
R = rng.random((num_movies, num_users)) < 0.65
R[:, 0] = False
R[[0, 1, 4, 5, 8, 10, 12, 16], 0] = True
Y = np.where(R, all_ratings, 0.0)
# Make User 1 ratings varied and easy to interpret.
Y[[0, 1, 4, 5, 8, 10, 12, 16], 0] = [5, 4, 2, 5, 4, 2, 5, 3]

# Center each movie's observed ratings for more stable optimization.
counts = R.sum(axis=1, keepdims=True)
movie_mean = np.divide(Y.sum(axis=1, keepdims=True), counts, out=np.full((num_movies, 1), 3.0), where=counts != 0)
Ynorm = np.where(R, Y - movie_mean, 0.0)

# Train x (movie features), w (user tastes), b (user biases) together.
X = rng.normal(scale=0.1, size=(num_movies, num_features))
W = rng.normal(scale=0.1, size=(num_users, num_features))
b = np.zeros((1, num_users))
lam, learning_rate, epochs = 0.1, 0.01, 3000

for epoch in range(epochs + 1):
    predictions = X @ W.T + b
    errors = (predictions - Ynorm) * R
    cost = 0.5 * np.sum(errors**2) + 0.5 * lam * (np.sum(X**2) + np.sum(W**2))
    if epoch % 500 == 0:
        print(f'epoch={epoch:4d}, cost={cost:.3f}')
    if epoch == epochs:
        break
    dX = errors @ W + lam * X
    dW = errors.T @ X + lam * W
    db = np.sum(errors, axis=0, keepdims=True)
    X -= learning_rate * dX
    W -= learning_rate * dW
    b -= learning_rate * db

scores = np.clip(X @ W.T + b + movie_mean, 1, 5)
user_index = 0
unwatched = np.where(~R[:, user_index])[0]
ranked = sorted(unwatched, key=lambda i: scores[i, user_index], reverse=True)
print('\nUser 1 known ratings:')
for i in np.where(R[:, user_index])[0]:
    print(f'  {movies[i]:28s} {Y[i, user_index]:.0f}/5')
print('\nTop 5 recommendations for User 1 (unrated movies only):')
for i in ranked[:5]:
    print(f'  {movies[i]:28s} predicted {scores[i, user_index]:.2f}/5')