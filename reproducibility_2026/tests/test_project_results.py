"""Check the recorded experiments agree with their published summaries."""
from pathlib import Path
import json
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

class RecordedResultsTest(unittest.TestCase):
    def test_quality_and_dataset_counts(self):
        q=json.loads((ROOT/'outputs/data_audit/data_quality.json').read_text(encoding='utf-8'))
        self.assertEqual(q['rows'],sum(q['diagnosis_counts'].values()))
        self.assertEqual(q['features'],30)

    def test_repeated_split_summary_matches_rows(self):
        root=ROOT/'outputs/robustness'
        rows=pd.read_csv(root/'repeated_splits.csv')
        summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows),summary['n_splits'])
        self.assertAlmostEqual(rows.recall.mean(),summary['metrics']['recall']['mean'])
        self.assertEqual(int((rows.test_flag_rate>.3).sum()),summary['test_budget_exceeded_count'])

    def test_main_threshold_satisfies_validation_budget(self):
        root=ROOT/'outputs/corrected'
        summary=json.loads((root/'run_summary.json').read_text(encoding='utf-8'))
        curve=pd.read_csv(root/'threshold_curve.csv')
        chosen=curve[(curve.threshold-summary['threshold']).abs()<1e-9]
        self.assertEqual(len(chosen),1)
        self.assertLessEqual(float(chosen.iloc[0].flag_rate),summary['budget_constraint'])
        self.assertGreater(summary['test_metrics']['flag_rate'],summary['budget_constraint'])

if __name__=='__main__':unittest.main()
