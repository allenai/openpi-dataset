import unittest

import pytest

from eval.generation_metric import ExactMetric, BLEUMetric, ROUGEMetric
from eval.simple_eval import f1_emnlp2020


class SimpleEvalTest(unittest.TestCase):

    def setUp(self) -> None:
        self.exact_metric = ExactMetric()
        self.bleu_metric = BLEUMetric()
        self.rouge_metric = ROUGEMetric()

    def test_simple_eval_bothEmpty(self):
        for metric in [self.exact_metric, self.bleu_metric, self.rouge_metric]:
            f1_both_empty = f1_emnlp2020(predictions=[], gold=[], generation_metric=metric)
            assert f1_both_empty == (1.0, 1.0, 1.0)

    def test_simple_eval_oneEmpty(self):
        empty_vals = []
        non_empty_vals = ["roasting pan was light before, and heavier afterwards.", "oil was in bottle before, and on pan afterwards."]

        for metric in [self.exact_metric, self.bleu_metric, self.rouge_metric]:
            f1_one_empty = f1_emnlp2020(predictions=empty_vals, gold=non_empty_vals, generation_metric=metric)
            assert f1_one_empty == (1.0, 0.0, 0.0)

        for metric in [self.exact_metric, self.bleu_metric, self.rouge_metric]:
            f1_one_empty = f1_emnlp2020(predictions=non_empty_vals, gold=empty_vals, generation_metric=metric)
            assert f1_one_empty == (0.0, 1.0, 0.0)

    def test_simple_eval_oneEmptyString(self):
        empty_vals = [""]
        non_empty_vals = ["roasting pan was light before, and heavier afterwards.", "oil was in bottle before, and on pan afterwards."]

        for metric in [self.exact_metric, self.bleu_metric, self.rouge_metric]:
            f1_one_empty = f1_emnlp2020(predictions=empty_vals, gold=non_empty_vals, generation_metric=metric)
            assert f1_one_empty == (0.0, 0.0, 0.0)

        for metric in [self.exact_metric, self.bleu_metric, self.rouge_metric]:
            f1_one_empty = f1_emnlp2020(predictions=non_empty_vals, gold=empty_vals, generation_metric=metric)
            assert f1_one_empty == (0.0, 0.0, 0.0)

    def test_simple_eval_nonEmpty(self):
        gold_vals = ["oil was cool before, and hot afterwards.",
                     "asparagus was dry before, and wet afterwards.",
                     "pan was cool before, and hot afterwards.",
                     "asparagus was crispy before, and softened afterwards."]
        prediction_vals = ["asparagus was dry before, and wet afterwards.",
                           "pan was cold before, and warm afterwards.",
                           "ham was cold before, and hot afterwards."]

        expected_metric_vals = [(0.3333333333333333, 0.25, 0.28571428571428575),
                                (0.3081907889704669, 0.25197642589753966, 0.2772629723345005),
                                (0.4382716049382716, 0.3703703703703704, 0.4014701724625389)]
        for metric, expected_metric_val in zip([self.exact_metric,
                                                self.bleu_metric,
                                                self.rouge_metric],
                                               expected_metric_vals):
            f1_nonempty = f1_emnlp2020(predictions=prediction_vals, gold=gold_vals, generation_metric=metric)
            print(f"metric:{metric.name()} val={f1_nonempty}")
            assert f1_nonempty == expected_metric_val

    def test_simple_eval_equal(self):
        gold_vals = ["oil was cool before, and hot afterwards.", "asparagus was dry before, and wet afterwards.", "pan was cool before, and hot afterwards.", "asparagus was crispy before, and softened afterwards."]
        prediction_vals = gold_vals

        expected_metric_vals = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]
        for metric, expected_metric_val in zip([self.exact_metric, self.bleu_metric], expected_metric_vals):
            print(metric.name())
            f1_nonempty = f1_emnlp2020(predictions=prediction_vals, gold=gold_vals, generation_metric=metric)

            for i in range(3):
                assert f1_nonempty[i] == pytest.approx(expected_metric_val[i], 0.1)

    def test_simple_eval_equal_diff_order(self):
        gold_vals = ["oil was cool before, and hot afterwards.",
                     "asparagus was dry before, and wet afterwards.",
                     "pan was cool before, and hot afterwards.",
                     "asparagus was crispy before, and softened afterwards."]
        prediction_vals = [
            gold_vals[3],
            gold_vals[0],
            gold_vals[2],
            gold_vals[1]
            ]

        expected_metric_vals = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]
        for metric, expected_metric_val in zip(
                [self.exact_metric, self.bleu_metric, self.rouge_metric], expected_metric_vals):
            print(metric.name())
            f1_nonempty = f1_emnlp2020(predictions=prediction_vals, gold=gold_vals, generation_metric=metric)
            for i in range(3):
                assert f1_nonempty[i] == pytest.approx(expected_metric_val[i], 0.1)

    def test_simple_eval_bothEmptyStrings(self):
        for metric in [self.exact_metric, self.bleu_metric, self.rouge_metric]:
            f1_both_empty = f1_emnlp2020(predictions=[""], gold=[""], generation_metric=metric)
            print(f"{metric.name()}\t{f1_both_empty}")
            assert f1_both_empty == (1.0, 1.0, 1.0)

    def test_simple_eval_for_normalization(self):
        gold_vals = ["oven racks"]
        prediction_vals = ["oven rack"]

        expected_metric_vals = [(1.0, 1.0, 1.0),
                                (1.0, 1.0, 1.0)]
        for metric, expected_metric_val in zip(
                [self.exact_metric, self.bleu_metric], expected_metric_vals):
            f1 = f1_emnlp2020(predictions=prediction_vals, gold=gold_vals, generation_metric=metric)
            print(f"{metric.name()}\t{f1}")
            for i in range(3):
                assert f1[i] == pytest.approx(expected_metric_val[i], 0.1)
