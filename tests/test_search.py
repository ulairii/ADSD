"""Check that invalid deployment tokens cannot enter a stronger search beam."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from sd_robustness.seed_search import run_gradient_guided_seed_search, select_calibration_indices

class SearchTests(unittest.TestCase):
    def test_invalid_high_scoring_candidate_cannot_replace_incumbent(self):
        scored=[];progress=[]
        def score(ids, position):
            scored.append(tuple(ids));return sum(ids)
        result=run_gradient_guided_seed_search(
            initial_candidate=SimpleNamespace(suffix_ids=[1,1],metrics=2),
            suffix_length=2,sweeps=2,beam_size=2,
            rank_token_ids=lambda c,p:[1,2,99],score_suffix=score,
            build_candidate=lambda ids,m:SimpleNamespace(suffix_ids=ids,metrics=m),
            objective_from_metrics=float,
            log_progress=lambda a,b,c,d,e,f:progress.append(f),
            candidate_allowed=lambda ids:99 not in ids)
        self.assertEqual(result[0].suffix_ids,[2,2])
        self.assertTrue(all(99 not in ids for ids in scored))
        self.assertEqual(progress, sorted(progress))

    def test_explicit_calibration_preserves_original_ids_and_rejects_duplicates(self):
        self.assertEqual(select_calibration_indices(100,3,0,1,[17,4,80]),[17,4,80])
        with self.assertRaises(ValueError):select_calibration_indices(100,3,0,1,[17,17,80])
        with self.assertRaises(ValueError):select_calibration_indices(100,3,0,1,[17,4,100])
