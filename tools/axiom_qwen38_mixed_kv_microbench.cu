// Synthetic attention-only gate; no model weights or production KV access.
#include "../src/axiom_qwen38_attention.cu"
#include <cmath>
#include <stdexcept>
#include <chrono>

namespace mixed_gate {
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
struct Memory {
 std::vector<void*> pointers;
 template<class T>T* get(size_t n){void*p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));pointers.push_back(p);return static_cast<T*>(p);}
 ~Memory(){for(void*p:pointers)cudaFree(p);}
};
__device__ uint32_t hash(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__global__ void init(uint8_t* pages,float*q,size_t n,unsigned seed,bool zeros){
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
  // Addition of a hashed seed avoids merely permuting the same 256-element
  // blocks when small XOR seeds differ (the former fixture weakness).
  const unsigned x=hash(unsigned(i)+hash(seed));pages[i]=zeros?0:uint8_t((x&128u)|((x>>8)%65u));
 }
 for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<kQDim;i+=gridDim.x*blockDim.x)
  q[i]=float(int(hash(i+hash(seed^0x12345678))%2049)-1024)/1024.f;
}
__global__ void marker(uint8_t*data,float*q,unsigned tokens,unsigned cold,size_t bytes){
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<bytes;i+=gridDim.x*blockDim.x){
  const unsigned page=i/AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES;
  const bool value=i%AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES>=AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES/2;
  data[i]=value?(page<cold?0x38:0xb8):0; // V=+1 in cold pages, -1 in hot pages
 }
 for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<kQDim;i+=gridDim.x*blockDim.x)q[i]=0;
}
void run(unsigned tokens,unsigned slots,unsigned seed,bool zeros,bool marker_case=false,bool gap=false){
 const unsigned page_tokens=AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS;
 const size_t page_bytes=AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES;
 const unsigned pages=(tokens+page_tokens-1)/page_tokens;
 const unsigned current=pages-1,last=(tokens-1)%page_tokens+1;
 size_t available=0,total=0;check(cudaMemGetInfo(&available,&total));
 if(available<size_t(pages)*page_bytes+512u*1024u*1024u)throw std::runtime_error("BLOCKED: insufficient free GPU memory plus safety margin");
 Memory m;
 auto*data=m.get<uint8_t>(size_t(pages)*page_bytes);auto*q=m.get<float>(kQDim);
 auto*sv=m.get<float>(kBatch*kAttentionSplitK*kQDim);
 auto*sm=m.get<float>(kBatch*kAttentionSplitK*kHeads);
 auto*sd=m.get<float>(kBatch*kAttentionSplitK*kHeads);
 auto*mx=m.get<float>(kHeads);auto*dn=m.get<float>(kHeads);auto*val=m.get<float>(kQDim);auto*out=m.get<float>(kQDim);
 auto*table=m.get<const uint8_t*>(slots);
 std::vector<const uint8_t*> ptrs(slots,nullptr);std::vector<unsigned> ids(slots,~0u);
 for(unsigned p=pages>slots?pages-slots:0;p<pages;++p){ptrs[p%slots]=data+p*page_bytes;ids[p%slots]=p;}
 if(gap&&current>0&&slots>1){ids[(current-1)%slots]=~0u;ptrs[(current-1)%slots]=nullptr;}
 check(cudaMemcpy(table,ptrs.data(),slots*sizeof(void*),cudaMemcpyHostToDevice));
 init<<<256,256>>>(data,q,size_t(pages)*page_bytes,seed,zeros);check(cudaDeviceSynchronize());
 const unsigned begin=axiom_qwen38::resident_suffix_begin(current,slots,ids.data());
 if(marker_case){marker<<<256,256>>>(data,q,tokens,begin,size_t(pages)*page_bytes);check(cudaDeviceSynchronize());}
 auto launch=[&](bool fast){
  qwen38_stream_init_stats_kernel<<<(kQDim+kThreads-1)/kThreads,kThreads>>>(mx,dn,val);
  for(unsigned p=0;p<(fast?begin:pages);++p){
   const uint8_t*page=data+p*page_bytes;const unsigned n=p==current?last:page_tokens;
   if(!fast){
    qwen38_stream_reference_page_kernel<<<kHeads,kHeadDim,kHeadDim*sizeof(float)>>>(q,page,page+page_bytes/2,n,mx,dn,val);
   }else{
    qwen38_attention_core_gqa_splitk8_kernel<false,kAttentionSplitK><<<dim3(kKvHeads,1,kAttentionSplitK),kAttentionThreads>>>(q,page,page+page_bytes/2,sv,sm,sd,n,0,nullptr,page_tokens);
    qwen38_stream_accumulate_page_kernel<><<<kKvHeads,kAttentionThreads>>>(sv,sm,sd,mx,dn,val);
   }
  }
  if(fast&&begin<pages){unsigned offset=begin*page_tokens,count=tokens-offset;
   qwen38_attention_core_gqa_splitk8_kernel<false,kBatch*kAttentionSplitK,true><<<dim3(kKvHeads,1,kBatch*kAttentionSplitK),kAttentionThreads>>>(q,nullptr,nullptr,sv,sm,sd,count,0,nullptr,count,table,slots,offset);
   qwen38_stream_accumulate_page_kernel<true><<<kKvHeads,kAttentionThreads>>>(sv,sm,sd,mx,dn,val);
  }
  qwen38_stream_finalize_stats_kernel<<<kKvHeads,kAttentionThreads>>>(mx,dn,val,out);check(cudaGetLastError());
 };
 std::vector<float> results[2];double ms[2]{};
 for(int path=0;path<2;++path){
  launch(path);check(cudaDeviceSynchronize()); // warmup, separate from timing
  auto t=std::chrono::steady_clock::now();
  for(int repeat=0;repeat<3;++repeat)launch(path);
  check(cudaDeviceSynchronize());ms[path]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/3;
  results[path].resize(kQDim);check(cudaMemcpy(results[path].data(),out,kQDim*sizeof(float),cudaMemcpyDeviceToHost));
 }
 double max_abs=0,err2=0,ref2=0;unsigned mismatch=0;
 for(unsigned i=0;i<kQDim;++i){double a=results[0][i],b=results[1][i];
  if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite attention");
  double e=std::abs(a-b);max_abs=std::max(max_abs,e);err2+=e*e;ref2+=a*a;mismatch+=a!=b;
 }
 const double relative=std::sqrt(err2/std::max(ref2,1e-30));
 // Predeclared bound: FP32 reduction regrouping before BF16 materialization.
 // Not bit-identical: require <1% aggregate error and <0.002 absolute error.
 bool pass=relative<0.01&&max_abs<0.002;
 if(marker_case){const double expected=2.0*begin*page_tokens/tokens-1.0;
  for(float value:results[1])pass=pass&&std::abs(value-expected)<=0.00391;
 }
 std::printf("tokens=%u slots=%u seed=%u zero=%u marker=%u gap=%u cold=%u abs=%.9g relative_l2=%.9g differing=%u reference_ms=%.3f mixed_ms=%.3f speedup=%.2f pass=%u\n",tokens,slots,seed,zeros,marker_case,gap,begin,max_abs,relative,mismatch,ms[0],ms[1],ms[0]/ms[1],pass);std::fflush(stdout);
 if(!pass)throw std::runtime_error("numerical gate failed");
}
}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--matrix"){
  for(unsigned seed:{13u,71u,901u})for(unsigned tokens:{1u,255u,256u,257u,8191u,8192u,8193u,65535u,65536u,65537u,69103u,131073u})mixed_gate::run(tokens,256,seed,false);
  for(unsigned slots:{1u,8u,64u,256u})for(unsigned tokens:{769u,69103u}){
   mixed_gate::run(tokens,slots,2027,false,true);mixed_gate::run(tokens,slots,2027,false,false,true);
  }
  for(unsigned tokens:{262144u,524288u,1048576u}){mixed_gate::run(tokens,256,2027,false);mixed_gate::run(tokens,256,2027,false,true);}
  mixed_gate::run(69103,256,2027,true);
  std::puts("mixed attention expanded matrix: 59 cases PASS");return 0;
 }
 if(argc==2&&std::string(argv[1])=="--sanity"){
  mixed_gate::run(769,2,2027,false,false,true);mixed_gate::run(769,1,2027,false,true);
  std::puts("mixed attention sanitizer subset PASS");return 0;
 }
 if(argc!=1)throw std::runtime_error("usage: microbench [--matrix|--sanity]");
 for(unsigned seed:{13u,71u})for(unsigned tokens:{255u,256u,257u,65535u,65536u,65537u,69103u})mixed_gate::run(tokens,256,seed,false);
 mixed_gate::run(69103,256,13,true);
 mixed_gate::run(769,1,13,false);mixed_gate::run(131073,256,71,false);
 std::puts("mixed attention synthetic gate PASS");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
