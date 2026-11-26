#include "rowmix.h"
#include "kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t limit = 256u << 20;
constexpr uint32_t absent = UINT32_MAX;
static_assert(sizeof(float)==4 && sizeof(double)==8 && std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<float>::is_iec559, "requires IEEE floating point");

void require(bool ok, const char *msg) { if (!ok) throw std::runtime_error(msg); }
void error_text(char *buf, size_t size, const char *text) {
    if (buf && size) std::snprintf(buf, size, "%s", text);
}
struct Cursor {
    const uint8_t *p;
    size_t remaining;
    const uint8_t *take(size_t n) {
        require(n<=remaining, "truncated model");
        const uint8_t *start=p; p+=n; remaining-=n; return start;
    }
    uint32_t u32() {
        const uint8_t *b=take(4); uint32_t v=0;
        for (unsigned i=0;i<4;++i) v |= uint32_t(b[i]) << (i*8);
        return v;
    }
    double f64() {
        const uint8_t *b=take(8); uint64_t v=0;
        for (unsigned i=0;i<8;++i) v |= uint64_t(b[i]) << (i*8);
        double result; std::memcpy(&result,&v,8);
        require(std::isfinite(result),"nonfinite model parameter"); return result;
    }
};
uint32_t crc32(const uint8_t *data, size_t n) {
    uint32_t v=~uint32_t(0);
    for (size_t i=0;i<n;++i) {
        v^=data[i];
        for (int bit=0;bit<8;++bit) v=(v>>1) ^ ((v&1) ? 0xedb88320u : 0u);
    }
    return ~v;
}
uint32_t count3(uint32_t a,uint32_t b,uint32_t c) {
    require(a && b && c && a<=limit/b && a*b<=limit/c,"invalid tensor size");
    return a*b*c;
}
struct Row { uint8_t bits; double gain,bias; const uint8_t *data; };
struct Node {
    uint32_t op, relu, a,b,h,w,c,offset,k,s,pad,groups,kw,count;
    double scale;
    std::vector<Row> rows;
};
uint32_t spatial(uint32_t x,uint32_t pad,uint32_t k,uint32_t s) {
    require(x+2*pad>=k,"invalid kernel extent"); return (x+2*pad-k)/s+1;
}
int8_t quant(double x,double scale) {
    double u=std::clamp(x/scale,-127.,127.);
    return static_cast<int8_t>(std::copysign(std::floor(std::abs(u)+0.5),u));
}
} // namespace

struct rm_model {
    std::vector<uint8_t> storage;
    std::vector<Node> nodes;
    std::vector<double> mean, inverse_std;
    rowmix::Kernel kernel;
    uint32_t arena;
};

namespace {
void parse(rm_model &m) {
    Cursor r{m.storage.data(),m.storage.size()};
    require(std::memcmp(r.take(8),"ROWMIX1\0",8)==0,"unknown model format");
    auto length=r.u32(), crc=r.u32();
    require(length==r.remaining && crc==crc32(r.p,length),"length or checksum mismatch");
    m.arena=r.u32(); auto n=r.u32();
    require(n>=2 && n<=1024 && m.arena>0 && m.arena<=limit,"invalid graph size");
    auto pre=r.u32();
    require(pre<=r.remaining/16 && pre<=m.arena,"invalid preprocessing length");
    for (uint32_t i=0;i<pre;++i) {
        m.mean.push_back(r.f64()); m.inverse_std.push_back(r.f64());
        require(m.inverse_std.back()>0,"invalid preprocessing scale");
    }
    std::vector<uint32_t> last(n);
    for (uint32_t i=0;i<n;++i) {
        Node v{};
        v.op=r.u32(); v.relu=r.u32(); v.a=r.u32(); v.b=r.u32();
        v.h=r.u32(); v.w=r.u32(); v.c=r.u32(); v.offset=r.u32();
        v.k=r.u32(); v.s=r.u32(); v.pad=r.u32(); v.groups=r.u32(); v.kw=r.u32(); v.scale=r.f64();
        v.count=count3(v.h,v.w,v.c);
        require(v.scale>0 && v.scale<=std::numeric_limits<float>::max()/127., "invalid activation scale");
        require(v.offset<=m.arena && v.count<=m.arena-v.offset,"arena range out of bounds");
        require(v.relu<=1 && v.k && v.k<=4096 && v.kw && v.kw<=4096 && v.s && v.s<=4096 && v.pad<=4096,"bad operator attributes");
        if (v.op!=2 && v.op!=4) require(v.k==1 && v.kw==1 && v.s==1 && v.pad==0,"unexpected spatial attributes");
        require(v.groups>0 && (v.op==2 || v.groups==1),"invalid groups");
        last[i]=i;
        if (i==0) require(v.op==0 && !v.relu && v.a==absent && v.b==absent && (!pre || pre==v.c),"invalid input node");
        else {
            require(v.op>=1 && v.op<=6 && v.a<i,"invalid operator or parent");
            require(v.op==3 ? v.b<i : v.b==absent,"invalid second parent");
            last[v.a]=i;
            const auto &a=m.nodes[v.a];
            if (v.op==3) {
                last[v.b]=i; const auto &b=m.nodes[v.b];
                require(a.h==b.h && a.w==b.w && a.c==b.c && v.h==a.h && v.w==a.w && v.c==a.c,"incompatible add");
            } else if (v.op==1) require(v.h==1 && v.w==1,"invalid dense output");
            else if (v.op==5) require(v.h==a.h && v.w==a.w && v.c==a.c,"invalid relu");
            else if (v.op==6) require(v.h==1 && v.w==1 && v.c==a.c,"invalid global average");
            else {
                require(v.h==spatial(a.h,v.pad,v.k,v.s) && v.w==spatial(a.w,v.pad,v.kw,v.s),"invalid spatial output");
                if (v.op==4) require(v.c==a.c && !v.relu && v.pad<std::min(v.k,v.kw),"invalid maxpool");
            }
            if (v.op==1 || v.op==2) {
                require(a.c%v.groups==0 && v.c%v.groups==0,"incompatible group sizes");
                uint32_t width=v.op==1 ? a.count : count3(v.k,v.kw,a.c/v.groups);
                require(width<=100000 && v.c<=r.remaining/(17+(width+1)/2),"invalid weight row count");
                v.rows.reserve(v.c);
                for (uint32_t out=0;out<v.c;++out) {
                    auto bits=*r.take(1); require(bits==4 || bits==8,"invalid precision");
                    double scale=r.f64(), bias=r.f64();
                    double gain=scale*a.scale;
                    require(scale>0 && std::isfinite(gain) && gain>0,"invalid weight scale");
                    auto data=r.take(bits==4 ? (width+1)/2 : width);
                    v.rows.push_back(Row{bits,gain,bias,data});
                }
            }
        }
        m.nodes.push_back(std::move(v));
    }
    require(!r.remaining,"trailing model data");
    for (uint32_t i=0;i<n;++i) {
        require(i==n-1 || last[i]>i,"unconsumed node");
        for (uint32_t j=0;j<i;++j) if (last[j]>=i) {
            const auto &u=m.nodes[j], &v=m.nodes[i];
            require(u.offset+u.count<=v.offset || v.offset+v.count<=u.offset,"overlapping live buffers");
        }
    }
}
void affine(const Node &v,const Node &a,const int8_t *src,int8_t *dst,rowmix::Dot dot) {
    for (uint32_t y=0;y<v.h;++y) for (uint32_t x=0;x<v.w;++x) for (uint32_t o=0;o<v.c;++o) {
        const Row &r=v.rows[o]; int32_t acc=0;
        if (v.op==1) {
            acc=dot(src,r.data,0,a.count,r.bits);
        } else {
            const int64_t xbase=int64_t(x)*v.s-v.pad;
            const int64_t left=std::max<int64_t>(0,-xbase), right=std::min<int64_t>(v.kw,a.w-xbase);
            for (uint32_t ky=0;ky<v.k && left<right;++ky) {
                const int64_t iy=int64_t(y)*v.s+ky-v.pad;
                if (iy<0 || iy>=a.h) continue;
                const int8_t *pixel=src+(iy*a.w+xbase+left)*a.c;
                if (v.groups==1) {
                    const uint32_t start=static_cast<uint32_t>((ky*v.kw+left)*a.c);
                    acc+=dot(pixel,r.data,start,static_cast<uint32_t>((right-left)*a.c),r.bits);
                } else {
                    const auto width=a.c/v.groups;
                    pixel+=(o/(v.c/v.groups))*width;
                    for (int64_t kx=left;kx<right;++kx) {
                        auto start=static_cast<uint32_t>((ky*v.kw+kx)*width);
                        acc+=dot(pixel+(kx-left)*a.c,r.data,start,width,r.bits);
                    }
                }
            }
        }
        double z=acc*r.gain+r.bias;
        dst[(y*v.w+x)*v.c+o]=quant(v.relu?std::max(0.,z):z,v.scale);
    }
}
void pool(const Node &v,const Node &a,const int8_t *src,int8_t *dst) {
    for (uint32_t y=0;y<v.h;++y) for (uint32_t x=0;x<v.w;++x) for (uint32_t c=0;c<v.c;++c) {
        int m=-127;
        for (uint32_t ky=0;ky<v.k;++ky) for (uint32_t kx=0;kx<v.kw;++kx) {
            int64_t iy=int64_t(y)*v.s+ky-v.pad, ix=int64_t(x)*v.s+kx-v.pad;
            if (iy>=0 && iy<a.h && ix>=0 && ix<a.w) m=std::max(m,int(src[(iy*a.w+ix)*a.c+c]));
        }
        dst[(y*v.w+x)*v.c+c]=quant(m*a.scale,v.scale);
    }
}
} // namespace

extern "C" rm_model *rm_open_file(const char *path,char *error,size_t cap) {
    return rm_open_file_ex(path,RM_AUTO,error,cap);
}
extern "C" const char *rm_backend(const rm_model *m) { return m?m->kernel.name:""; }
extern "C" rm_model *rm_open_file_ex(const char *path,rm_kernel requested,char *error,size_t cap) {
    try {
        require(path,"missing path");
        std::ifstream f(path,std::ios::binary|std::ios::ate);
        require(bool(f),"cannot open model");
        auto sz=f.tellg(); require(sz>=16 && sz<=limit,"unsupported file size");
        auto m=std::make_unique<rm_model>(); m->storage.resize(static_cast<size_t>(sz));
        f.seekg(0); f.read(reinterpret_cast<char*>(m->storage.data()),sz);
        require(bool(f),"failed reading model"); parse(*m); m->kernel=rowmix::choose_kernel(requested); return m.release();
    } catch (const std::exception &e) { error_text(error,cap,e.what()); return nullptr; }
}
extern "C" void rm_close(rm_model *m) { delete m; }
extern "C" size_t rm_inputs(const rm_model *m) { return m?m->nodes.front().count:0; }
extern "C" size_t rm_outputs(const rm_model *m) { return m?m->nodes.back().count:0; }
extern "C" size_t rm_arena_bytes(const rm_model *m) { return m?m->arena:0; }
extern "C" size_t rm_node_count(const rm_model *m) { return m?m->nodes.size():0; }
extern "C" size_t rm_node_elements(const rm_model *m,size_t i) { return m && i<m->nodes.size()?m->nodes[i].count:0; }
extern "C" int rm_predict(const rm_model *m,const float *input,size_t ni,float *output,size_t no,
                          int8_t *arena,size_t na,rm_trace_fn trace,void *user,char *error,size_t cap) {
    if (!m || !input || !output || !arena || ni!=rm_inputs(m) || no!=rm_outputs(m) || na<m->arena) {
        error_text(error,cap,"invalid input, output or arena"); return -1;
    }
    for (size_t i=0;i<m->nodes.size();++i) {
        const Node &v=m->nodes[i]; auto *dst=arena+v.offset;
        if (i==0) {
            for (size_t j=0;j<ni;++j) {
                if (!std::isfinite(input[j])) { error_text(error,cap,"nonfinite input"); return -1; }
                double x=input[j];
                if (!m->mean.empty()) x=(x-m->mean[j%v.c])*m->inverse_std[j%v.c];
                dst[j]=quant(x,v.scale);
            }
        } else {
            const auto &a=m->nodes[v.a]; auto src=arena+a.offset;
            if (v.op==1 || v.op==2) affine(v,a,src,dst,m->kernel.dot);
            else if (v.op==3) {
                const auto &b=m->nodes[v.b]; auto rhs=arena+b.offset;
                for (size_t j=0;j<v.count;++j) {
                    double z=src[j]*a.scale+rhs[j]*b.scale;
                    dst[j]=quant(v.relu?std::max(0.,z):z,v.scale);
                }
            } else if (v.op==4) pool(v,a,src,dst);
            else if (v.op==5) {
                for (size_t j=0;j<v.count;++j) dst[j]=quant(std::max(0,int(src[j]))*a.scale,v.scale);
            } else {
                for (uint32_t c=0;c<v.c;++c) {
                    int64_t sum=0;
                    for (size_t j=c;j<a.count;j+=a.c) sum+=src[j];
                    double avg=sum*a.scale/(a.h*a.w);
                    dst[c]=quant(v.relu?std::max(0.,avg):avg,v.scale);
                }
            }
        }
        if (trace) trace(static_cast<uint32_t>(i),dst,v.count,user);
    }
    auto &v=m->nodes.back();
    for (size_t j=0;j<no;++j) output[j]=static_cast<float>(arena[v.offset+j]*v.scale);
    return 0;
}
