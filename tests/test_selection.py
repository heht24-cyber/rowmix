from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.stats import binomtest
from rowmix.select import lower_bound
from test_compiler import toy

ROOT=Path(__file__).resolve().parents[1]


class SelectionTests(unittest.TestCase):
    def test_exact_limits_and_multiple_candidates(self):
        for k in (0,1,15,36):
            per=binomtest(k,36,alternative="greater").proportion_ci(confidence_level=.99,method="exact").low
            self.assertAlmostEqual(lower_bound(k,36,.95,5),per,places=10)
        self.assertLess(lower_bound(36,36,.95,5),.95)
        self.assertGreater(lower_bound(360,360,.95,5),.95)

    def test_failed_gate_preserves_model_and_writes_report(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); g,x=toy(); g.save(p/"graph");np.save(p/"cal.npy",x[:6]);np.save(p/"val.npy",x[6:])
            (p/"deployment").write_bytes(b"previous")
            command=[sys.executable,"-m","rowmix.select",str(p/"graph"),str(p/"cal.npy"),str(p/"val.npy"),
                     "--runner",str(ROOT/"build/rowmix-run"),"--output",str(p/"deployment"),"--report",str(p/"r"),
                     "--profiles","3","--passes","1"]
            r=subprocess.run(command+["--min-agreement","1"],capture_output=True,cwd=ROOT)
            self.assertEqual(r.returncode,3,r.stderr)
            self.assertEqual((p/"deployment").read_bytes(),b"previous")
            self.assertIsNone(json.loads((p/"r").read_text())["selected_sha256"])
            subprocess.run(command+["--min-agreement","0"],capture_output=True,check=True,cwd=ROOT)
            result=json.loads((p/"r").read_text()); self.assertIsNotNone(result["selected_sha256"])
            self.assertEqual((p/"deployment").stat().st_size,min(a["model_bytes"] for a in result["candidates"]))
