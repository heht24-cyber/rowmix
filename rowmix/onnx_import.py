"""Strict inference-only ONNX lowering. Public inputs are NCHW/NC; native files use NHWC."""
import argparse
from collections import Counter
from dataclasses import dataclass
import json

import numpy as np
from .graph import Graph
from .compiler import Calibration
from .cli import atomic_write


@dataclass(frozen=True)
class Value:
    node: int
    form: str  # image (NCHW), flat (flattened NCHW), or matrix (NC)
    direct: bool = False


def uniform(items, label):
    if len(set(items)) != 1:
        raise ValueError(f"{label} must be uniform, got {items}")
    return int(items[0])


def channel_value(a, channels, image):
    a=np.asarray(a,dtype=np.float64)
    if a.size==1:
        return np.full(channels,float(a.ravel()[0]))
    shapes=((channels,1,1),(1,channels,1,1)) if image else ((channels,),(1,channels))
    if a.shape not in shapes:
        raise ValueError("constant must broadcast per channel")
    return a.reshape(-1)


def load_onnx(path):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    model=onnx.load(str(path),load_external_data=False)
    if any(t.data_location==TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError("requires self-contained ONNX weights")
    if model.graph.sparse_initializer or model.functions:
        raise ValueError("sparse weights and custom functions are unsupported")
    onnx.checker.check_model(model)
    graph=model.graph
    constants={t.name:numpy_helper.to_array(t) for t in graph.initializer}
    inputs=[x for x in graph.input if x.name not in constants]
    if len(inputs)!=1 or len(graph.output)!=1:
        raise ValueError("requires one data input and one output")
    t=inputs[0].type.tensor_type
    dims=[int(d.dim_value) for d in t.shape.dim]
    if t.elem_type!=TensorProto.FLOAT or len(dims) not in (2,4) or min(dims[1:])<1:
        raise ValueError("requires float32 NC or NCHW with static nonbatch dimensions")
    if len(dims)==4:
        shape=(dims[2],dims[3],dims[1]); input_form="image"
    else:
        shape=(1,1,dims[1]); input_form="matrix"
    g=Graph(shape)
    values={inputs[0].name: Value(0,input_form,True)}
    uses=Counter(x for n in graph.node for x in n.input)
    uses.update([graph.output[0].name])

    def affine(source, name, multiplier, offset):
        n=g.nodes[source.node]
        if source.form=="flat":
            raise ValueError("per-channel affine on flattened image is unsupported")
        if source.direct and uses[name]==1 and source.node==0 and np.all(multiplier>0):
            mean,scale=(np.zeros(n.shape[2]),np.ones(n.shape[2])) if g.preprocess is None else g.preprocess.T
            scale=scale*multiplier
            g.set_preprocess(mean-offset/scale,scale)
            return Value(0,source.form,True)
        if source.direct and uses[name]==1 and n.weights is not None and not n.relu:
            n.weights=n.weights*multiplier[:,None]; n.bias=n.bias*multiplier+offset
            return Value(source.node,source.form,True)
        idx=g.add("conv",[source.node],kernel=1,groups=n.shape[2],weights=multiplier[:,None],bias=offset)
        return Value(idx,source.form,True)

    for node in graph.node:
        names=list(node.input); op=node.op_type
        try:
            if node.domain not in ("","ai.onnx") or len([s for s in node.output if s])!=1:
                raise ValueError("unsupported domain or number of outputs")
            attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
            def get(k,default):
                return attrs.pop(k,default)
            if op=="Constant":
                tensor=get("value",None)
                if tensor is None or attrs or tensor.data_location==TensorProto.EXTERNAL:
                    raise ValueError("requires one inline tensor")
                constants[node.output[0]]=numpy_helper.to_array(tensor)
                continue
            if op=="Transpose" and names[0] in constants:
                arg=constants[names[0]]
                perm=get("perm",list(reversed(range(arg.ndim))))
                if attrs: raise ValueError("unexpected transpose option")
                constants[node.output[0]]=arg.transpose(perm)
                continue
            if op in ("Add","Mul") and names[0] in constants and names[1] in values:
                names=names[::-1]
            source=values[names[0]]; n=g.nodes[source.node]
            h,w,c=n.shape
            if op in ("Conv","MaxPool"):
                if source.form!="image": raise ValueError("requires NCHW input")
                if get("auto_pad",b"NOTSET") not in (b"NOTSET",b"VALID"): raise ValueError("auto_pad mode unsupported")
                pad=uniform(get("pads",[0]*4),"padding")
                stride=uniform(get("strides",[1,1]),"stride")
                if get("dilations",[1,1])!=[1,1]: raise ValueError("dilation unsupported")
                if op=="Conv":
                    weight=constants[names[1]]
                    if weight.ndim!=4: raise ValueError("requires OIHW weights")
                    kh,kw=get("kernel_shape",list(weight.shape[2:]))
                    if weight.shape[2:]!=(kh,kw): raise ValueError("invalid weight dimensions")
                    bias=constants[names[2]] if len(names)>2 and names[2] else np.zeros(len(weight))
                    idx=g.add("conv",[source.node],kernel=kh,kernel_w=kw,stride=stride,pad=pad,groups=int(get("group",1)),
                              weights=weight.transpose(0,2,3,1).reshape(len(weight),-1),bias=bias)
                else:
                    if get("ceil_mode",0) or get("storage_order",0): raise ValueError("unsupported pooling mode")
                    kh,kw=get("kernel_shape",[])
                    idx=g.add("maxpool",[source.node],kernel=kh,kernel_w=kw,stride=stride,pad=pad)
                result=Value(idx,"image",True)
            elif op in ("Gemm","MatMul"):
                if source.form=="image": raise ValueError("flatten image before linear layer")
                weight=constants[names[1]]
                if weight.ndim!=2: raise ValueError("weights must be a matrix")
                if op=="Gemm":
                    if get("transA",0): raise ValueError("transA unsupported")
                    get("broadcast",1)  # only scalar or channel biases pass channel_value
                    W=(weight if get("transB",0) else weight.T)*float(get("alpha",1))
                    beta=float(get("beta",1))
                    bias=channel_value(constants[names[2]],len(W),False)*beta if len(names)>2 and names[2] else np.zeros(len(W))
                else:
                    W=weight.T; bias=np.zeros(len(W))
                if source.form=="flat":
                    W=W.reshape(len(W),c,h,w).transpose(0,2,3,1).reshape(len(W),-1)
                result=Value(g.add("dense",[source.node],weights=W,bias=bias),"matrix",True)
            elif op in ("Identity","Flatten","Reshape"):
                if op=="Flatten" and get("axis",1)!=1: raise ValueError("only axis=1 Flatten supported")
                if op=="Reshape":
                    target=constants[names[1]].tolist()
                    if target not in ([0,-1],[-1,n.count],[0,n.count]) or get("allowzero",0):
                        raise ValueError("only batch-preserving flatten reshapes supported")
                form="flat" if op!="Identity" and source.form in ("image","flat") else source.form
                result=Value(source.node,form,False)
            elif op=="Relu":
                if source.direct and uses[names[0]]==1 and n.op in ("dense","conv","add","globalavg"):
                    n.relu=True; idx=source.node
                else:
                    idx=g.add("relu",[source.node])
                result=Value(idx,source.form,True)
            elif op=="GlobalAveragePool":
                if source.form!="image": raise ValueError("requires an image")
                result=Value(g.add("globalavg",[source.node]),"image",True)
            elif op=="Add" and names[1] in values:
                other=values[names[1]]
                if source.form!=other.form: raise ValueError("broadcast residual unsupported")
                result=Value(g.add("add",[source.node,other.node]),source.form,True)
            elif op in ("Add","Sub","Mul","Div"):
                value=channel_value(constants[names[1]],c,source.form=="image")
                mul=np.ones(c); bias=np.zeros(c)
                if op=="Add": bias=value
                elif op=="Sub": bias=-value
                elif op=="Mul": mul=value
                else:
                    if np.any(value==0): raise ValueError("division by zero")
                    mul=1/value
                result=affine(source,names[0],mul,bias)
            elif op=="BatchNormalization":
                if get("training_mode",0) or not get("is_test",1): raise ValueError("inference BatchNorm only")
                get("momentum",.9)
                gamma,beta,mean,var=[constants[val].reshape(-1) for val in names[1:5]]
                eps=float(get("epsilon",1e-5))
                if any(len(z)!=c for z in (gamma,beta,mean,var)) or np.any(var+eps<=0):
                    raise ValueError("invalid batch normalization parameters")
                scale=gamma/np.sqrt(var.astype(np.float64)+eps)
                result=affine(source,names[0],scale,beta-mean*scale)
            else:
                raise ValueError("operator unsupported")
            if attrs: raise ValueError(f"unsupported attributes {sorted(attrs)}")
            values[node.output[0]]=result
        except (ValueError,IndexError,KeyError,TypeError) as e:
            raise ValueError(f"ONNX {op} {node.name or node.output[0]}: {e}") from e
    output=values[graph.output[0].name]
    if output.node==0 and len(g.nodes)==1:
        i=g.add("conv",[0],groups=shape[2],weights=np.ones((shape[2],1)),bias=np.zeros(shape[2]))
        output=Value(i,output.form)
    if output.node!=len(g.nodes)-1: raise ValueError("output must be the final reachable node")
    g.validate()
    return g,input_form,output.form


def native_input(x, form):
    x = np.asanyarray(x)
    expected_rank = 4 if form == "image" else 2
    if x.ndim != expected_rank:
        raise ValueError(f"expected a rank-{expected_rank} calibration batch for {form} input")
    return x.transpose(0, 2, 3, 1) if form == "image" else x


def original_layout(x, form):
    return x.transpose(0,3,1,2) if form=="image" else (
        x.transpose(0,3,1,2).reshape(len(x),-1) if form=="flat" else x.reshape(len(x),-1))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("onnx"); p.add_argument("calibration",help="ONNX-layout NCHW or NC .npy")
    p.add_argument("output",help="native .rmx file; inputs use flat NHWC")
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bytes",type=int);mode.add_argument("--fixed",type=int,choices=(4,8))
    p.add_argument("--save-graph",help="also write the converted float .npz")
    p.add_argument("--calibration-batch", type=int, help="maximum samples per float forward pass")
    args=p.parse_args()
    try:
        g,source,output=load_onnx(args.onnx)
        samples = np.load(args.calibration, mmap_mode="r", allow_pickle=False)
        cal = Calibration(g, native_input(samples, source), batch_size=args.calibration_batch)
        model=cal.fit(args.bytes) if args.bytes is not None else cal.fixed(args.fixed)
        blob=model.to_bytes(); atomic_write(args.output,blob)
        if args.save_graph: g.save(args.save_graph)
        print(json.dumps(dict(file_bytes=len(blob),arena_bytes=model.arena,onnx_input=source,onnx_output=output,
                              native_input_shape=g.nodes[0].shape,native_output_shape=g.nodes[-1].shape)))
    except (ValueError,OSError,KeyError) as exc:
        p.error(str(exc))


if __name__=="__main__": main()
