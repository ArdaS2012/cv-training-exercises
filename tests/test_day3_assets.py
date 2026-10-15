"""Public asset/interface checks; no concept TODO answers."""
import unittest,json
import numpy as np
import torch
from cv_exercises import day3
class Day3Assets(unittest.TestCase):
 def test_segmentation_assets(self):
  for i in range(4):
   s=day3.segmentation(i)
   self.assertEqual(s['probabilities'].shape,(3,32,48))
   self.assertTrue(np.isfinite(s['probabilities']).all())
   self.assertTrue(np.allclose(s['probabilities'].sum(0),1,atol=1e-6))
   self.assertEqual(s['instance'].shape,s['reference_instance'].shape)
 def test_pair_assets(self):
  data,a,b=day3.pairs()
  self.assertEqual(a.shape,(24,8));self.assertEqual(b.shape,a.shape)
  self.assertTrue(np.isfinite(a).all())
  provenance=json.loads((day3.ASSETS/'teaching-provenance.json').read_text(encoding='utf-8'))
  self.assertFalse(set(provenance['siamese_source_ids'])&set(__import__('cv_exercises.data',fromlist=['images']).images()[2]))
 def test_foundation_manifest(self):
  specs=day3.model_specs()['foundation'];cached=torch.load(day3.ASSETS/'foundation-encodings.pt',weights_only=True)
  for spec in specs.values():
   self.assertEqual(len(spec['revision']),40);self.assertTrue(spec['license']);self.assertTrue(spec['source'].startswith('https://'))
  for value in cached.values():self.assertTrue(torch.isfinite(value).all())
 def test_starters_unsupplied(self):
  from start import DAY3_NOTEBOOKS
  for path in DAY3_NOTEBOOKS.values():
   text=(day3.ROOT/'exercises'/path).read_text(encoding='utf-8');self.assertIn('TODO',text);self.assertIn('=None',text.replace(' ',''))
if __name__=='__main__':unittest.main()
