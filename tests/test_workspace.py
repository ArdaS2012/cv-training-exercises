import ast
import unittest
from pathlib import Path
import numpy as np
from cv_exercises.data import images,scene,pixel_inputs,classification_split,geometry
from start import NOTEBOOKS
ROOT=Path(__file__).resolve().parents[1]
class WorkspaceTests(unittest.TestCase):
    def test_assets_and_sample_identity(self):
        x,y,ids=images()
        self.assertEqual(x.shape,(300,64));self.assertEqual(len(set(ids)),300)
        self.assertTrue(np.isfinite(x).all());self.assertTrue(((x>=0)&(x<=1)).all())
        self.assertEqual(set(y),set(range(10)))
        self.assertEqual(geometry('curved').shape,(150,2))
    def test_pixel_row_order(self):
        rgb=scene();color,position=pixel_inputs(rgb)
        self.assertEqual(color.shape,(1536,3));self.assertEqual(position.shape,(1536,2))
        self.assertTrue(np.array_equal(color.reshape(rgb.shape),rgb))
    def test_no_split_overlap(self):
        _,_,_,tr,va,te=classification_split()
        self.assertEqual([len(tr),len(va),len(te)],[180,60,60])
        self.assertFalse(set(tr)&set(va));self.assertFalse(set(tr)&set(te));self.assertFalse(set(va)&set(te))
    def test_starters_and_selector_paths(self):
        for path in NOTEBOOKS.values():
            p=ROOT/'exercises'/path
            ast.parse(p.read_text(encoding='utf-8'))
            self.assertIn('# TODO',p.read_text(encoding='utf-8'));self.assertIn('# END TODO',p.read_text(encoding='utf-8'))
            self.assertTrue((p.parent/'README.md').is_file())
if __name__=='__main__':unittest.main()
