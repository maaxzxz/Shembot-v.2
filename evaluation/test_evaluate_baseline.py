import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector import Detection  # noqa: E402
from evaluate_baseline import iou, match_detections  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_iou_identity_and_disjoint(self) -> None:
        self.assertEqual(iou((10, 10, 20, 20), (10, 10, 20, 20)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 10, 10)), 0.0)

    def test_one_to_one_matching(self) -> None:
        detections = [
            Detection((10, 10, 20, 20), 0.9),
            Detection((11, 11, 20, 20), 0.8),
        ]
        self.assertEqual(match_detections(detections, [(10, 10, 20, 20)], 0.5), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()

