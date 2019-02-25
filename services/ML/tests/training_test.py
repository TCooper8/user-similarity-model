import unittest

from src.analyzer.training import train_user_similarities

class TestTrainUserSimilarities(unittest.TestCase):
  def test(self):
    model = train_user_similarities()
    user_handle = 1
    similar = model.similar_to(user_handle)

    # The list should always at least contain themself.
    self.assertTrue(
      str(user_handle) in similar,
    )