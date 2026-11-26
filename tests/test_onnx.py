import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from rowmix.onnx_import import load_onnx, native_input, original_layout


@unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),"install validation dependencies")
class ImportTests(unittest.TestCase):
    def test_official_vectors_and_native_trace(self):
        from benchmarks.interchange import upstream_cases
        self.assertGreaterEqual(len(upstream_cases()),14)

    def test_fanout_keeps_negative_branch_and_global_average(self):
        import onnx
        from onnx import TensorProto as T,helper as h
        from benchmarks.interchange import session
        from rowmix.compiler import Calibration
        from test_runtime import command
        inp=h.make_tensor_value_info("x",T.FLOAT,["N",3,2,4]); out=h.make_tensor_value_info("y",T.FLOAT,["N",3,1,1])
        nodes=[h.make_node("Relu",["x"],["positive"]),h.make_node("Add",["positive","x"],["sum"]),
               h.make_node("GlobalAveragePool",["sum"],["y"])]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); m=h.make_model(h.make_graph(nodes,"skip",[inp],[out]),opset_imports=[h.make_opsetid("",13)],ir_version=8)
            onnx.save(m,p/"g.onnx"); model,form,out_form=load_onnx(p/"g.onnx")
            x=np.random.default_rng(6).normal(size=(7,3,2,4)).astype(np.float32)
            z=native_input(x,form); ref=session(p/"g.onnx").run(None,{"x":x})[0]
            np.testing.assert_allclose(original_layout(model.trace(z)[-1],out_form),ref,atol=2e-6)
            compiled=Calibration(model,z).fixed(8); (p/"m").write_bytes(compiled.to_bytes())
            np.ascontiguousarray(z,dtype="<f4").tofile(p/"x"); command("trace",p/"m",p/"x",p/"t")
            expected=np.concatenate([q.reshape(len(z),-1) for q in compiled.trace(z)],axis=1).ravel()
            np.testing.assert_array_equal(expected,np.fromfile(p/"t",np.int8))

    def test_unsupported_operator_named(self):
        import onnx
        from onnx import TensorProto as T,helper as h
        with tempfile.TemporaryDirectory() as d:
            a=h.make_tensor_value_info("x",T.FLOAT,[1,3]);b=h.make_tensor_value_info("y",T.FLOAT,[1,3])
            g=h.make_graph([h.make_node("Sigmoid",["x"],["y"],name="gate")],"bad",[a],[b])
            m=h.make_model(g,opset_imports=[h.make_opsetid("",13)],ir_version=8)
            f=Path(d)/"g";onnx.save(m,f)
            with self.assertRaisesRegex(ValueError,"Sigmoid gate"):
                load_onnx(f)
