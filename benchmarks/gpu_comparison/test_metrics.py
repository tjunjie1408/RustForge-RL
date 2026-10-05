"""Guard the benchmark's metric validation against false passes and false failures."""

import tempfile
import unittest
from pathlib import Path

from run import read_metrics


class MetricsValidation(unittest.TestCase):
    def check_csv(self, rows, episodes=2):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text("episode,reward,avg_loss,epsilon,global_step\n" + rows, encoding="utf-8")
            return read_metrics(path, "dqn", episodes)

    def test_documented_warmup_loss_is_accepted(self):
        self.assertEqual(self.check_csv("0,10,NaN,1,11\n1,20,0.5,0.9,32\n"), (32, 15.0))

    def test_run_without_training_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check_csv("0,10,NaN,1,11\n1,20,NaN,0.9,32\n")

    def test_nan_after_training_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check_csv("0,10,0.5,1,11\n1,20,NaN,0.9,32\n")

    def test_infinite_loss_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check_csv("0,10,inf,1,11\n1,20,0.5,0.9,32\n")

    def test_nonfinite_reward_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check_csv("0,NaN,NaN,1,11\n1,20,0.5,0.9,32\n")

    def test_missing_episodes_are_rejected(self):
        with self.assertRaises(ValueError):
            self.check_csv("0,10,0.5,1,11\n")

    def test_nonincreasing_steps_are_rejected(self):
        with self.assertRaises(ValueError):
            self.check_csv("0,10,0.5,1,11\n1,20,0.5,0.9,11\n")


if __name__ == "__main__":
    unittest.main()
