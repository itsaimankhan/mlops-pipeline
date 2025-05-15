
import unittest
from data_processing import load_and_preprocess
from train_model import train

class TestPipeline(unittest.TestCase):
    def test_data_loading(self):
        X_train, X_test, y_train, y_test = load_and_preprocess()
        self.assertEqual(len(X_train) > 0, True)

    def test_model_accuracy(self):
        accuracy = train()
        self.assertGreaterEqual(accuracy, 0.8)

if __name__ == "__main__":
    unittest.main()
