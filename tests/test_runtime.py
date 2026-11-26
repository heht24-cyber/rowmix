import concurrent.futures
import ctypes as ct
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

import numpy as np
from test_compiler import toy
from rowmix.compiler import Calibration, NODE, PREFIX
from rowmix.graph import Graph

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"build/rowmix-run"


def command(*args, kernel="auto", check=True):
    return subprocess.run([str(RUN),*map(str,args)],check=check,capture_output=True,env=dict(os.environ,ROWMIX_KERNEL=kernel))


class RuntimeTests(unittest.TestCase):
    def test_seeded_dags_all_native_values(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            for seed,channels in enumerate([1,3,5,8,13,17]):
                graph,x=toy(seed,channels)
                cal=Calibration(graph,x[:7])
                for model in (cal.fixed(4),cal.fixed(8),cal.fit((cal.minimum+cal.maximum)//2)):
                    path=d/"model"; path.write_bytes(model.to_bytes())
                    x[7:].astype("<f4").tofile(d/"x")
                    oracle=model.trace(x[7:])
                    ref=np.concatenate([a.reshape(len(a),-1) for a in oracle],axis=1).ravel()
                    for backend in ("scalar","auto"):
                        command("trace",path,d/"x",d/"q",kernel=backend)
                        np.testing.assert_array_equal(np.fromfile(d/"q",dtype=np.int8),ref)

    def test_output_preserved_on_last_bad_sample(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); g,x=toy(); m=Calibration(g,x[:4]).fixed(8)
            (d/"m").write_bytes(m.to_bytes()); x[-1,0,0,0]=np.nan; x.astype("<f4").tofile(d/"x")
            (d/"out").write_bytes(b"previous result")
            proc=command("run",d/"m",d/"x",d/"out",check=False)
            self.assertNotEqual(proc.returncode,0)
            self.assertEqual((d/"out").read_bytes(),b"previous result")
            self.assertFalse(list(d.glob(".rowmix-*")))

    def test_model_validation(self):
        g,x=toy(); orig=Calibration(g,x).fixed(4).to_bytes()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"m"
            examples=[orig[:-1],orig+b"suffix",b"invalid model"]
            for offset,value in [(12+NODE.size+7*4,0),(12+NODE.size+2*4,20)]:
                # Body contains the 12-byte graph header then input and convolution nodes.
                body=bytearray(orig[PREFIX.size:])
                struct.pack_into("<I",body,offset,value)
                examples.append(PREFIX.pack(b"ROWMIX1\0",len(body),zlib.crc32(body))+body)
            for content in examples:
                p.write_bytes(content)
                self.assertNotEqual(command("inspect",p,check=False).returncode,0)

    def test_cli_uses_atomic_budgeted_output(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); g,x=toy(); g.save(d/"g.npz"); np.save(d/"x.npy",x)
            cmd=[sys.executable,"-m","rowmix.cli",str(d/"g.npz"),str(d/"x.npy"),str(d/"m")]
            subprocess.run(cmd+["--fixed","8"],check=True,capture_output=True,cwd=ROOT)
            original=(d/"m").read_bytes()
            failed=subprocess.run(cmd+["--bytes","1"],capture_output=True,cwd=ROOT)
            self.assertNotEqual(failed.returncode,0); self.assertEqual((d/"m").read_bytes(),original)
            cal=Calibration(g,x); cap=(cal.minimum+cal.maximum)//2
            done=subprocess.run(cmd+["--bytes",str(cap),"--audit",str(d/"x.npy")],check=True,capture_output=True,cwd=ROOT)
            report=json.loads(done.stdout)
            self.assertEqual(report["file_bytes"],len((d/"m").read_bytes()))
            self.assertEqual(len(report["layers"]),len(g.nodes))

    def test_c_abi_concurrent_inference(self):
        lib=ct.CDLL(str(ROOT/"build/librowmix.so"))
        ptr=ct.c_void_p; size=ct.c_size_t
        lib.rm_open_file.argtypes=[ct.c_char_p,ptr,size]; lib.rm_open_file.restype=ptr
        lib.rm_predict.argtypes=[ptr,ptr,size,ptr,size,ptr,size,ptr,ptr,ptr,size]
        lib.rm_predict.restype=ct.c_int; lib.rm_close.argtypes=[ptr]
        with tempfile.TemporaryDirectory() as d:
            g,x=toy(); m=Calibration(g,x).fixed(8); path=Path(d)/"m"; path.write_bytes(m.to_bytes())
            err=ct.create_string_buffer(256); handle=lib.rm_open_file(os.fsencode(path),err,len(err))
            self.assertTrue(handle,err.value)
            ref=(m.trace(x)[-1].reshape(len(x),-1)*m.scales[-1]).astype(np.float32)
            def one(i):
                row=np.ascontiguousarray(x[i]); out=np.empty(ref.shape[1],np.float32)
                arena=np.empty(m.arena,np.int8); msg=ct.create_string_buffer(256)
                args=(handle,row.ctypes.data,row.size,out.ctypes.data,out.size,arena.ctypes.data)
                self.assertNotEqual(lib.rm_predict(*args,m.arena-1,None,None,msg,len(msg)),0)
                self.assertEqual(lib.rm_predict(*args,m.arena,None,None,msg,len(msg)),0)
                np.testing.assert_array_equal(out,ref[i])
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(one,range(len(x))))
            finally: lib.rm_close(handle)

    def test_examples(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("digits","wine"):
                p=ROOT/"examples"/name
                command("run",p/"mixed.rmx",p/"sample.f32",Path(d)/"out")
                self.assertEqual((Path(d)/"out").read_bytes(),(p/"expected.f32").read_bytes())
                info=json.loads(command("inspect",p/"mixed.rmx").stdout)
                self.assertGreater(info["arena_bytes"],0)
