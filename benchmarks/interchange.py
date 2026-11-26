"""Compare RowMix imports with ONNX Runtime and upstream ONNX test vectors."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper as H, numpy_helper as N
import onnxruntime as ort
from sklearn.datasets import load_digits, load_wine
from rowmix.compiler import Calibration
from rowmix.graph import Graph
from rowmix.onnx_import import load_onnx, native_input, original_layout
from .run import ROOT, invoke
from .train import split


def export(graph, path):
    """Build standard NCHW test models, including explicit input preprocessing."""
    initial=[]; ops=[]
    def const(name, a):
        initial.append(N.from_array(np.asarray(a,dtype=np.float32),name)); return name
    def op(kind, ins, out, **kw):
        ops.append(H.make_node(kind,ins,[out],**kw)); return out
    sh=graph.nodes[0].shape
    image=sh[:2]!=(1,1)
    shape=["N",sh[2],sh[0],sh[1]] if image else ["N",sh[2]]
    names=["x"]; forms=[image]
    if graph.preprocess is not None:
        pre=graph.preprocess.T
        broadcast=(1,-1,1,1) if image else (1,-1)
        a=op("Sub",["x",const("mean",pre[0].reshape(broadcast))],"centered")
        names[0]=op("Mul",[a,const("inv_std",pre[1].reshape(broadcast))],"scaled")
    for i,node in enumerate(graph.nodes[1:],1):
        tag=f"n{i}"; par=node.parents[0]; x=names[par]; is_image=forms[par]
        h,w,c=graph.nodes[par].shape
        if node.op=="conv":
            weights=node.weights.reshape(-1,node.kernel,node.kernel_w,c//node.groups).transpose(0,3,1,2)
            z=op("Conv",[x,const(tag+"W",weights),const(tag+"B",node.bias)],tag,
                 strides=[node.stride]*2,pads=[node.pad]*4,group=node.groups)
        elif node.op=="dense":
            weights=node.weights
            if is_image:
                x=op("Flatten",[x],tag+"flat",axis=1)
                weights=weights.reshape(len(weights),h,w,c).transpose(0,3,1,2).reshape(len(weights),-1)
            z=op("Gemm",[x,const(tag+"W",weights),const(tag+"B",node.bias)],tag,transB=1)
            is_image=False
        elif node.op=="add":
            z=op("Add",[x,names[node.parents[1]]],tag)
        elif node.op=="maxpool":
            z=op("MaxPool",[x],tag,kernel_shape=[node.kernel,node.kernel_w],strides=[node.stride]*2,pads=[node.pad]*4)
        elif node.op=="globalavg": z=op("GlobalAveragePool",[x],tag)
        else: z=op("Relu",[x],tag)
        if node.relu: z=op("Relu",[z],tag+"relu")
        names.append(z); forms.append(is_image)
    h,w,c=graph.nodes[-1].shape; outshape=["N",c,h,w] if forms[-1] else ["N",c]
    g=H.make_graph(ops,"rowmix-validation",[H.make_tensor_value_info("x",TensorProto.FLOAT,shape)],
                   [H.make_tensor_value_info(names[-1],TensorProto.FLOAT,outshape)],initial)
    model=H.make_model(g,opset_imports=[H.make_opsetid("",13)],ir_version=8)
    onnx.checker.check_model(model); onnx.save(model,str(path))


def session(path):
    options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path),options,providers=["CPUExecutionProvider"])


def upstream_cases():
    package=Path(onnx.__file__).parent/"backend/test/data/pytorch-converted"
    names=["test_Conv2d","test_Conv2d_strided","test_Conv2d_padding","test_Conv2d_no_bias",
           "test_Conv2d_depthwise","test_Conv2d_depthwise_padded","test_Conv2d_depthwise_strided",
           "test_Conv2d_depthwise_with_multiplier","test_Linear","test_Linear_no_bias",
           "test_MaxPool2d","test_BatchNorm2d_eval","test_BatchNorm2d_momentum_eval","test_ReLU"]
    results=[]
    for name in names:
        path=package/name
        def read_tensor(filename):
            proto=TensorProto();proto.ParseFromString((path/"test_data_set_0"/filename).read_bytes())
            return N.to_array(proto)
        x=read_tensor("input_0.pb"); expected=read_tensor("output_0.pb")
        g,form,output=load_onnx(path/"model.onnx")
        z=native_input(x,form); got=original_layout(g.trace(z)[-1],output)
        np.testing.assert_allclose(got,expected,rtol=2e-5,atol=2e-5)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); cal=Calibration(g,z); m=cal.fit((cal.minimum+cal.maximum)//2)
            (d/"m").write_bytes(m.to_bytes()); np.ascontiguousarray(z,dtype="<f4").tofile(d/"x")
            invoke("trace",d/"m",d/"x",d/"trace")
            target=np.concatenate([t.reshape(len(z),-1) for t in m.trace(z)],axis=1).ravel()
            np.testing.assert_array_equal(np.fromfile(d/"trace",np.int8),target)
        results.append(dict(case=name,float_values=int(got.size),max_abs=float(np.max(np.abs(got-expected))),
                            matched_native_trace_values=target.size))
    return results


def main():
    result=dict(onnx=onnx.__version__,onnxruntime=ort.__version__,models=[],upstream=upstream_cases())
    for name,data in (("digits",load_digits),("wine",load_wine)):
        dest=ROOT/"examples"/name
        g=Graph.load(dest/"float.npz"); export(g,dest/"model.onnx")
        sets,ys,_=split(*data(return_X_y=True)); x=sets[2]
        onnx_x=x.reshape(-1,1,8,8) if name=="digits" else x
        og,form,layout=load_onnx(dest/"model.onnx")
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            s=session(dest/"model.onnx"); y=s.run(None,{s.get_inputs()[0].name:onnx_x})[0]
            imported=original_layout(og.trace(native_input(onnx_x,form))[-1],layout)
            np.testing.assert_allclose(imported,y,atol=3e-5,rtol=2e-5)
            calib=Calibration(og,sets[1]); model=calib.fit((calib.minimum+calib.maximum)//2)
            (d/"m").write_bytes(model.to_bytes()); x.astype("<f4").tofile(d/"x")
            invoke("run",d/"m",d/"x",d/"q"); q=np.fromfile(d/"q",dtype="<f4").reshape(y.shape)
            ref=model.trace(x)[-1].reshape(y.shape)*model.scales[-1]
            np.testing.assert_array_equal(q,ref.astype(np.float32))
            result["models"].append(dict(name=name,float_max_abs=float(np.max(np.abs(y-imported))),
                samples=len(x),values=int(y.size),onnx_float_correct=int(np.sum(y.argmax(1)==ys[2])),
                native_quantized_correct=int(np.sum(q.argmax(1)==ys[2])),
                onnx_bytes=(dest/"model.onnx").stat().st_size,rmx_bytes=len(model.to_bytes()),
                onnx_sha256=hashlib.sha256((dest/"model.onnx").read_bytes()).hexdigest()))
    (ROOT/"benchmarks/interchange.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["models"],indent=2)); print(len(result["upstream"]),"upstream fixtures passed")


if __name__=="__main__": main()
