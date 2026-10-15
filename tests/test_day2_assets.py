"""Public starter/setup checks: no completed student implementations or answers."""
import ast,json,unittest
from pathlib import Path
import numpy as np
from cv_exercises import day2
from start import DAY2_NOTEBOOKS
class Day2Assets(unittest.TestCase):
    def test_starter_paths_and_todos(self):
        for path in DAY2_NOTEBOOKS.values():
            source=(day2.ROOT/'exercises'/path).read_text();ast.parse(source)
            self.assertIn('# TODO',source);self.assertIn('# END TODO',source)
            self.assertTrue((day2.ROOT/'exercises'/path).with_name('README.md').is_file())
    def test_pretraining_provenance_and_cached_identity(self):
        provenance=json.loads((day2.ASSETS/'training-provenance.json').read_text())
        self.assertTrue(provenance['source_target_disjoint'])
        cache=np.load(day2.ASSETS/'cached-features.npz')
        self.assertEqual(cache['features'].shape,(300,48))
        self.assertTrue(np.isfinite(cache['features']).all())
        self.assertTrue(np.array_equal(cache['image_ids'],day2.target_data()[2]))
    def test_checkpoints_load_and_remain_finite(self):
        x,*_=day2.target_data();model=day2.tiny_model()
        import torch
        with torch.no_grad():output=model(x[:2])
        self.assertEqual(tuple(output.shape),(2,10));self.assertTrue(torch.isfinite(output).all())
        for k in [4,16]:
            clean,noisy,models,_=day2.ae_inputs(0,k,.25)
            self.assertEqual(tuple(clean.shape),tuple(noisy.shape))
            for model in models.values():
                with torch.no_grad():output=model(noisy)
                self.assertEqual(tuple(output.shape),(1,64));self.assertTrue(torch.isfinite(output).all())
    def test_detector_snapshot_identity(self):
        asset=day2.detector_snapshot();self.assertEqual(len(asset['images']),2)
        for entry in asset['images']:
            self.assertTrue((day2.ASSETS/entry['image']).is_file());self.assertTrue(entry['references'])
            self.assertEqual(set(entry['outputs']),{'YOLO11n','SSDLite320'})
            for candidates in entry['outputs'].values():
                self.assertTrue(candidates)
                for d in candidates:
                    self.assertTrue(np.isfinite(d['box']).all());self.assertGreater(d['box'][2],d['box'][0]);self.assertGreater(d['box'][3],d['box'][1]);self.assertGreaterEqual(d['score'],0);self.assertLessEqual(d['score'],1)
    def test_no_split_overlap(self):
        *_,tr,va,te=day2.target_data()
        self.assertFalse(set(tr)&set(va));self.assertFalse(set(tr)&set(te));self.assertFalse(set(va)&set(te))
if __name__=='__main__':unittest.main()
