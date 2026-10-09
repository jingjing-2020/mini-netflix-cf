"""Tests for teaching algorithm, using AST to avoid starting a Streamlit server."""
import ast
import pathlib
import unittest
import numpy as np
import pandas as pd

APP_PATH = pathlib.Path(__file__).resolve().parents[1] / 'netflix_app.py'
tree = ast.parse(APP_PATH.read_text(encoding='utf-8'))
selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'demo_data', 'movie_lens_data', 'fit_cf'}]
module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
namespace = {'np': np, 'pd': pd, 'DEMO_MOVIES': [
    'Titanic', 'The Notebook', 'La La Land', 'Pride & Prejudice', 'Inception',
    'Interstellar', 'The Matrix', 'Avatar', 'Toy Story', 'Finding Nemo',
    'Inside Out', 'Coco', 'The Dark Knight', 'Iron Man', 'Spider-Man',
    'Avengers', 'The Shawshank Redemption', 'Forrest Gump', 'The Godfather', 'Whiplash'
]}
exec(compile(module, str(APP_PATH), 'exec'), namespace)

class RecommenderTests(unittest.TestCase):
    def test_demo_data_shapes_and_mask(self):
        titles, Y, R = namespace['demo_data']()
        self.assertEqual((len(titles), Y.shape, R.shape), (20, (20, 10), (20, 10)))
        self.assertTrue(np.all(Y[~R] == 0))
        self.assertEqual(int(R[:, 0].sum()), 8)

    def test_cost_decreases_and_prediction_range(self):
        _, Y, R = namespace['demo_data']()
        result, costs = namespace['fit_cf'](Y, R, epochs=250)
        self.assertEqual(result.shape, Y.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all((result >= 1) & (result <= 5)))
        self.assertLess(costs.iloc[-1]['cost'], costs.iloc[0]['cost'])

    def test_uploaded_movie_data(self):
        from io import StringIO
        movies = StringIO('movieId,title\n1,Film A\n2,Film B\n')
        ratings = StringIO('userId,movieId,rating\n11,1,4.5\n12,2,3.0\n11,2,5.0\n')
        titles, Y, R = namespace['movie_lens_data'](movies, ratings)
        self.assertEqual(Y.shape, (2, 2))
        self.assertEqual(int(R.sum()), 3)
        self.assertTrue(np.all(Y[~R] == 0))

if __name__ == '__main__':
    unittest.main()
