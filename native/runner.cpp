#include "rowmix.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace fs=std::filesystem;
using clock_type=std::chrono::steady_clock;
static void check(bool value,const char *message) { if (!value) throw std::runtime_error(message); }

struct Output {
    fs::path dest,dir,file;
    std::ofstream stream;
    explicit Output(const fs::path &target):dest(target) {
        std::random_device rng;
        for (int i=0;i<16;++i) {
            auto candidate=dest.parent_path()/(".rowmix-"+std::to_string(rng()));
            if (fs::create_directory(candidate)) { dir=candidate; break; }
        }
        check(!dir.empty(),"could not stage output");
        file=dir/"output";
        stream.open(file,std::ios::binary);
        if (!stream) { fs::remove_all(dir); throw std::runtime_error("cannot open output"); }
    }
    void finish() {
        stream.close(); check(bool(stream),"failed writing output");
#ifdef _WIN32
        check(MoveFileExW(file.c_str(),dest.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)!=0,
              "failed publishing output");
#else
        fs::rename(file,dest);
#endif
    }
    ~Output() { std::error_code ec; stream.close(); fs::remove_all(dir,ec); }
};
static void write_trace(uint32_t,const int8_t *p,size_t n,void *handle) {
    static_cast<std::ostream*>(handle)->write(reinterpret_cast<const char*>(p),n);
}
static int run(int argc,char **argv) {
    check(argc>=3,"usage: rowmix-run inspect MODEL | run/trace MODEL INPUT OUTPUT | bench MODEL INPUT REPEATS");
    const std::string cmd=argv[1];
    uint32_t endian=1;
    check(*reinterpret_cast<uint8_t*>(&endian)==1,"raw float CLI needs a little-endian host");
    char error[256]{};
    const char *setting=std::getenv("ROWMIX_KERNEL");
    std::string choice=setting?setting:"auto";
    check(choice=="scalar" || choice=="avx2" || choice=="auto","unknown ROWMIX_KERNEL");
    rm_kernel requested=choice=="scalar"?RM_SCALAR:choice=="avx2"?RM_AVX2:RM_AUTO;
    std::unique_ptr<rm_model,decltype(&rm_close)> m(rm_open_file_ex(argv[2],requested,error,sizeof(error)),rm_close);
    if (!m) throw std::runtime_error(error);
    auto ni=rm_inputs(m.get()),no=rm_outputs(m.get()),na=rm_arena_bytes(m.get());
    if (cmd=="inspect" && argc==3) {
        std::cout<<"{\"kernel\":\""<<rm_backend(m.get())<<"\",\"inputs\":"<<ni<<",\"outputs\":"<<no<<",\"arena_bytes\":"<<na<<",\"node_elements\":[";
        for (size_t i=0;i<rm_node_count(m.get());++i) std::cout<<(i?",":"")<<rm_node_elements(m.get(),i);
        std::cout<<"]}\n"; return 0;
    }
    check(argc==5 && (cmd=="run" || cmd=="trace" || cmd=="bench"),"invalid command or argument count");
    std::ifstream in(argv[3],std::ios::binary|std::ios::ate);
    check(bool(in),"cannot open inputs");
    auto end=in.tellg(); check(end>0 && uint64_t(end)%(ni*sizeof(float))==0,"incomplete or empty input sample");
    size_t samples=uint64_t(end)/(ni*sizeof(float)); in.seekg(0);
    std::vector<int8_t> arena(na);
    std::vector<float> y(no);
    auto predict=[&](const float *x,rm_trace_fn callback,void *user) {
        if (rm_predict(m.get(),x,ni,y.data(),no,arena.data(),na,callback,user,error,sizeof(error))) throw std::runtime_error(error);
    };
    if (cmd=="bench") {
        size_t consumed=0; long repeats=std::stol(argv[4],&consumed);
        check(consumed==std::strlen(argv[4]) && repeats>=1 && repeats<=10000,"invalid repeat count");
        check(uint64_t(end)<=256u*1024*1024,"benchmark input exceeds 256 MiB");
        std::vector<float> x(samples*ni);
        in.read(reinterpret_cast<char*>(x.data()),end); check(bool(in),"input read failed");
        for (size_t i=0;i<samples;++i) predict(x.data()+ni*i,nullptr,nullptr);
        std::vector<double> times;
        for (long pass=0;pass<repeats;++pass) {
            auto begin=clock_type::now();
            for (size_t i=0;i<samples;++i) predict(x.data()+ni*i,nullptr,nullptr);
            times.push_back(std::chrono::duration<double,std::micro>(clock_type::now()-begin).count()/samples);
        }
        std::sort(times.begin(),times.end());
        auto percentile=[&](double f){return times[static_cast<size_t>(std::ceil(f*times.size()))-1];};
        std::cout<<"{\"kernel\":\""<<rm_backend(m.get())<<"\",\"samples\":"<<samples<<",\"passes\":"<<repeats<<",\"median_us_per_sample\":"<<percentile(.5)
                 <<",\"p95_pass_mean_us\":"<<percentile(.95)<<"}\n";
    } else {
        std::vector<float> row(ni);
        Output output(argv[4]);
        for (size_t i=0;i<samples;++i) {
            in.read(reinterpret_cast<char*>(row.data()),ni*sizeof(float)); check(bool(in),"input read failed");
            predict(row.data(),cmd=="trace"?write_trace:nullptr,&output.stream);
            if (cmd=="run") output.stream.write(reinterpret_cast<const char*>(y.data()),no*sizeof(float));
            check(bool(output.stream),"output write failed");
        }
        output.finish();
    }
    return 0;
}
int main(int argc,char **argv) {
    try { return run(argc,argv); }
    catch (const std::exception &e) { std::cerr<<"rowmix: "<<e.what()<<"\n"; return 2; }
}
