#define XTLLM_LONGCAT 1
#if !defined(XTLLM_LONGCAT_Q4_SHARED) && !defined(XTLLM_LONGCAT_DENSE_Q4) && !defined(XTLLM_LONGCAT_DENSE_OUT_Q4)
#define XTLLM_LONGCAT_Q8_SHARED 1
#endif
#define OVLLM_QWEN35_RUNTIME_ONLY 1
#include "m16_qwen35.cpp"

namespace longcat {
using namespace qwen35;
constexpr uint32_t D=3072, ED=1024, DD=6144, L=14, K=12, E=256, ET=384;
#if defined(XTLLM_LONGCAT_DENSE_Q4) || defined(XTLLM_LONGCAT_DENSE_OUT_Q4)
constexpr TensorFormat OE_W=TensorFormat::q8_row;
constexpr TensorFormat ATTN_W=TensorFormat::q8_row;
constexpr TensorFormat DENSE_W=TensorFormat::q4g64t;
#elif defined(XTLLM_LONGCAT_Q4_SHARED)
constexpr TensorFormat OE_W=TensorFormat::q4g64t;
constexpr TensorFormat ATTN_W=TensorFormat::q4g64t;
constexpr TensorFormat DENSE_W=TensorFormat::q4g64t;
#else
constexpr TensorFormat OE_W=TensorFormat::q8_row;
constexpr TensorFormat ATTN_W=TensorFormat::q8_row;
constexpr TensorFormat DENSE_W=TensorFormat::q8_row;
#endif
constexpr TensorFormat W=ATTN_W;
constexpr uint32_t QL=1536, KVL=512, H=32, QHD=192, VD=128;
constexpr uint64_t REC=5013504;
struct OeHeader { char magic[8]; uint32_t version,header_bytes,tables,dimension,vocabulary,base_rows,row_bytes,reserved; uint64_t q[40]; };
static_assert(sizeof(OeHeader)==360);

class OeFile {
public:
    explicit OeFile(const std::filesystem::path& path): input_(path,std::ios::binary) {
        if(!input_) throw std::runtime_error("Could not open LongCat n-gram store");
        input_.read(reinterpret_cast<char*>(&header_),sizeof(header_));
        if(std::memcmp(header_.magic,"OLCFOE1\0",8)||header_.version!=1||header_.header_bytes!=4096||header_.tables!=12||header_.dimension!=256||header_.vocabulary!=131072||header_.row_bytes!=512)
            throw std::runtime_error("Unsupported LongCat n-gram store");
    }
    void read_rows(const std::vector<uint32_t>& history, void* output) {
        auto* dst=static_cast<uint8_t*>(output); const uint32_t pos=uint32_t(history.size()-1);
        for(uint32_t n=2;n<=4;++n) for(uint32_t k=0;k<4;++k){
            uint32_t table=(n-2)*4+k,mod=uint32_t(header_.q[20+table]); uint64_t id=0,power=1;
            for(uint32_t delta=0;delta<n&&delta<=pos;++delta){uint32_t token=history[pos-delta];if(delta>0&&token==2u)break;id=(id+uint64_t(token)*power)%mod;power=power*131072ull%mod;}
            input_.seekg(std::streamoff(header_.q[table]+id*512ull)); input_.read(reinterpret_cast<char*>(dst+table*512),512);
            if(!input_) throw std::runtime_error("LongCat n-gram row read failed");
        }
    }
private: std::ifstream input_; OeHeader header_{};
};

struct Pipe { VkPipeline embedding{},rms{},quant{},q4{},q4one{},q4r{},q8{},q8r{},oe{},attn_matrix{},attn_matrix_r{},dense_matrix{},dense_matrix_r{},swiglu{},argmax{}; VkPipeline bf16{},mean{},router{},egate{},edown{},reduce{},store{},attention{},add{}; };
class Kernels {
public:
    Kernels(const Runtime&r,const std::filesystem::path&dir):runtime(r),res(create_compute_resources(r,16384)),dummy(create_device_buffer(r,4096)){
        auto load=[&](const char*n){return dsv4::create_dsv4_pipeline(runtime,res,dir/(std::string(n)+".comp.spv"),64);};
        p.embedding=load("dsv4_embedding");p.rms=load("step37_rmsnorm");p.quant=load("dsv4_quantize_q8");
#if defined(XTLLM_LONGCAT_Q4_SHARED) || defined(XTLLM_LONGCAT_DENSE_Q4) || defined(XTLLM_LONGCAT_DENSE_OUT_Q4)
        p.q4=load("dsv4_q4g64t_gemv");p.q4one=load("dsv4_q4g64t_gemv_one_lane");p.q4r=load("dsv4_q4g64t_gemv_residual");
#else
        p.q4=load("dsv4_q8_gemv");p.q4r=load("dsv4_q8_gemv_residual");
#endif
        p.q8=load("dsv4_q8_gemv");p.q8r=load("dsv4_q8_gemv_residual");
#ifdef XTLLM_LONGCAT_DENSE_OUT_Q4
        p.oe=p.q8;p.attn_matrix=p.q8;p.attn_matrix_r=p.q4r;p.dense_matrix=std::getenv("LONGCAT_Q4_ONE_LANE")?p.q4one:p.q4;p.dense_matrix_r=p.q4r;
        p.q4=p.q8;
#elif defined(XTLLM_LONGCAT_DENSE_Q4)
        p.oe=p.q8;p.attn_matrix=p.q8;p.attn_matrix_r=p.q8r;p.dense_matrix=std::getenv("LONGCAT_Q4_ONE_LANE")?p.q4one:p.q4;p.dense_matrix_r=p.q4r;
        p.q4=p.q8;p.q4r=p.q8r;
#else
        p.oe=p.q4;p.attn_matrix=p.q4;p.attn_matrix_r=p.q4r;p.dense_matrix=p.q4;p.dense_matrix_r=p.q4r;
#endif
        p.swiglu=load("step37_swiglu");p.argmax=load("qwen35_greedy_argmax");p.bf16=load("longcat_bf16_rows");p.mean=load("longcat_ngram_mean");p.router=load("longcat_router_top12_mainfast");p.egate=load("longcat_expert_gate_up_q4");p.edown=load("longcat_expert_down_q4");p.reduce=load("longcat_moe_reduce");p.store=load("longcat_mla_store");p.attention=load("longcat_mla_attention");p.add=load("longcat_add_shortcut");
    }
    ~Kernels(){for(auto x:res.pipelines)vkfn::DestroyPipeline(runtime.device,x,nullptr);for(auto x:res.shader_modules)vkfn::DestroyShaderModule(runtime.device,x,nullptr);if(res.descriptor_pool)vkfn::DestroyDescriptorPool(runtime.device,res.descriptor_pool,nullptr);if(res.pipeline_layout)vkfn::DestroyPipelineLayout(runtime.device,res.pipeline_layout,nullptr);if(res.descriptor_layout)vkfn::DestroyDescriptorSetLayout(runtime.device,res.descriptor_layout,nullptr);destroy_buffer(runtime,dummy);}
    VkDescriptorSet set(std::initializer_list<DescriptorRange> ranges){std::array<DescriptorRange,6>a;a.fill(whole(dummy));uint32_t i=0;for(auto x:ranges)a[i++]=x;return dsv4::create_dsv4_set(runtime,res,a);}
    void dispatch(VkCommandBuffer c,VkPipeline x,VkDescriptorSet s,const Push*push,uint32_t gx,uint32_t gy=1){dsv4::dispatch_dsv4(c,res,x,s,push,gx,gy);} Pipe p;
private:const Runtime&runtime;ComputeResources res{};Buffer dummy{};
};

struct AttnSets { VkDescriptorSet in_norm{},hidden_q{},qa{},qa_norm{},qa_q{},qb{},kva{},kva_norm{},kva_q{},kvb{},store{},attention{},ctx_q{},out{}; };
struct DenseSets { VkDescriptorSet norm{},hidden_q{},gate{},up{},act{},act_q{},down{}; };
struct LayerSets { std::array<AttnSets,2> attn; std::array<DenseSets,2> dense; VkDescriptorSet router_gemv{},router{}; std::vector<VkDescriptorSet> egate,edown; };

class Engine {
public:
    Engine(const Runtime&r,const SharedIndex&index,const std::filesystem::path&experts,const std::filesystem::path&oe,const std::filesystem::path&shaders,uint64_t ram,uint32_t slots,uint32_t context_capacity)
      :rt(r),weights(r,index),efile(experts),hcache(efile,ram),dcache(r,slots),oefile(oe),kern(r,shaders),compute(r,r.queue),transfer(r,r.secondary_queue),context_capacity_(context_capacity){if(!context_capacity_)throw std::runtime_error("LongCat context capacity must be positive");if(const char*text=std::getenv("LONGCAT_ACTIVE_TOPK")){active=uint32_t(std::stoul(text));if(active==0||active>K)throw std::runtime_error("LONGCAT_ACTIVE_TOPK must be 1..12");}allocate();rope();sets();staging.resize(K);for(auto&x:staging)x=dsv4::create_host_buffer_uninitialized(rt,REC);oe_stage=dsv4::create_host_buffer_uninitialized(rt,12*512);if(std::getenv("LONGCAT_FILL_RAM_CACHE")){double seconds=0;uint32_t filled=hcache.fill_remaining_uniform(seconds);std::cout<<"RAM cache top-off: "<<filled<<" records, "<<double(hcache.committed_bytes())/(1ull<<30)<<" GiB cache, "<<seconds<<" s\n";}}
    ~Engine(){for(auto&x:staging)destroy_buffer(rt,x);destroy_buffer(rt,oe_stage);for(Buffer*x:{&token,&routing,&word,&oe_bf16,&oe_rows,&oe_q,&oe_proj,&hidden,&norm,&quant,&qa,&qan,&qaq,&kva,&kvan,&kvaq,&kvexp,&context,&ctxq,&dense_gate,&dense_up,&dense_mid,&denseq,&router_logits,&shortcut,&expert_mid,&expertq,&expert_out,&kv_cache,&ropebuf,&logits,&argmax})destroy_buffer(rt,*x);}
    uint32_t run(uint32_t tok,uint32_t pos,std::vector<uint32_t>&history){if(pos>=context_capacity_)throw std::runtime_error("LongCat prompt plus generation exceeds allocated context");*static_cast<uint32_t*>(token.mapped)=tok;flush_buffer(rt,token);history.push_back(tok);oefile.read_rows(history,oe_stage.mapped);dsv4::flush_buffer_range(rt,oe_stage,0,12*512);
        uint64_t ready=transfer.submit([&](VkCommandBuffer c){VkBufferCopy x{0,0,12*512};vkfn::CmdCopyBuffer(c,oe_stage.handle,oe_bf16.handle,1,&x);dsv4::transfer_barrier(c,oe_bf16);});
        uint64_t sig=compute.submit([&](VkCommandBuffer c){Push p0{131072,D,D/4,0};kern.dispatch(c,kern.p.embedding,embed_set,&p0,(D+63)/64);Push pb{12*256,0,0,0};kern.dispatch(c,kern.p.bf16,bf16_set,&pb,(12*256+63)/64);compute_barrier(c);for(uint32_t i=0;i<12;++i){Push pq{256,128,64,64};kern.dispatch(c,kern.p.quant,oe_quant_sets[i],&pq,2);compute_barrier(c);Push pg{D,256,64,0};kern.dispatch(c,kern.p.oe,oe_proj_sets[i],&pg,D/8);compute_barrier(c);}Push pm{D,12,0,0};kern.dispatch(c,kern.p.mean,mean_set,&pm,D/64);compute_barrier(c);},transfer.semaphore(),ready,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);compute.wait(sig);
        for(uint32_t layer=0;layer<L;++layer){attention_block(layer,0,pos);moe(layer);dense(layer,0);attention_block(layer,1,pos);dense(layer,1);sig=compute.submit([&](VkCommandBuffer c){Push p{D,0,0,0};kern.dispatch(c,kern.p.add,add_set,&p,D/64);compute_barrier(c);});compute.wait(sig);}
        sig=compute.submit([&](VkCommandBuffer c){Push p{1,D,float_bits(1e-5f),0};kern.dispatch(c,kern.p.rms,final_norm_set,&p,1);compute_barrier(c);p={D,128,D/4,D/4};kern.dispatch(c,kern.p.quant,final_q_set,&p,D/128);compute_barrier(c);p={131072,D,D/4,0};kern.dispatch(c,kern.p.q8,head_set,&p,(131072+7)/8);compute_barrier(c);p={131072,256,0,0};kern.dispatch(c,kern.p.argmax,argmax_set,&p,256);compute_barrier(c);p={131072,256,1,0};kern.dispatch(c,kern.p.argmax,argmax_set,&p,1);compute_barrier(c);});compute.wait(sig);invalidate_buffer(rt,token);return *static_cast<uint32_t*>(token.mapped);}
    uint64_t ram_bytes()const{return hcache.committed_bytes();}uint64_t vram_bytes()const{return weights.device_bytes()+dcache.device_bytes()+activation_bytes;}uint64_t disk_bytes()const{return hcache.disk_bytes();}uint64_t h2d_bytes()const{return h2d;}uint64_t host_copy_bytes()const{return hostcopy;}uint64_t device_hits()const{return dcache.hits();}uint64_t device_misses()const{return dcache.misses();}uint64_t ram_hits()const{return hcache.hits();}uint64_t ram_misses()const{return hcache.misses();}uint32_t host_slots()const{return hcache.capacity();}uint32_t active_routes()const{return active;}
private:
    Buffer dev(uint64_t n){auto x=create_device_buffer(rt,n);activation_bytes+=x.allocation_size;return x;}
    TensorDevice tensor(const std::string&name,TensorFormat fmt,uint64_t a,uint64_t b0=0){
#if defined(XTLLM_LONGCAT_DENSE_Q4) || defined(XTLLM_LONGCAT_DENSE_OUT_Q4)
        if(name.find(".dense.")!=std::string::npos)fmt=DENSE_W;
#ifdef XTLLM_LONGCAT_DENSE_OUT_Q4
        if(name.find(".attn.")!=std::string::npos && name.size()>=2 &&
           name.compare(name.size()-2,2,".o")==0)fmt=TensorFormat::q4g64t;
#endif
#endif
        auto x=weights.tensor(name);uint32_t rank=b0?2:1;if(x.format!=fmt||x.rank!=rank||x.shape[0]!=a||(b0&&x.shape[1]!=b0))throw std::runtime_error("Unexpected LongCat tensor ABI: "+name);return x;}
    VkDescriptorSet q4set(DescriptorRange a,const TensorDevice&w,DescriptorRange o,DescriptorRange residual={}){return kern.set({a,w.data,w.auxiliary,o,residual.buffer?residual:whole(norm)});}
    void allocate(){token=create_buffer(rt,4);routing=create_buffer(rt,128);word=dev(D*4);oe_bf16=dev(12*512);oe_rows=dev(12*256*4);oe_q=dev(512);oe_proj=dev(12ull*D*4);hidden=dev(D*4);norm=dev(D*4);quant=dev(4224*4);qa=dev(H*QHD*4);qan=dev(QL*4);qaq=dev(2048*4);kva=dev((KVL+64)*4);kvan=dev(KVL*4);kvaq=dev(768*4);kvexp=dev(H*256*4);context=dev(H*VD*4);ctxq=dev(4224*4);dense_gate=dev(DD*4);dense_up=dev(DD*4);dense_mid=dev(DD*4);denseq=dev(6400*4);router_logits=dev(ET*4);shortcut=dev(D*4);expert_mid=dev(uint64_t(K)*ED*4);expertq=dev(4096*4);expert_out=dev(uint64_t(K)*D*4);kv_cache=dev(28ull*context_capacity_*H*320*4);ropebuf=dev(uint64_t(context_capacity_)*64*4);logits=dev(131072ull*4);argmax=dev(512*4);}
    void rope(){std::vector<float>v(uint64_t(context_capacity_)*64);for(uint32_t p=0;p<context_capacity_;++p)for(uint32_t i=0;i<32;++i){double ramp=std::clamp((double(i)-8.0)/9.0,0.0,1.0),mask=1.0-ramp,base=std::pow(1000000.0,double(2*i)/64.0),inv=mask/base+(1.0-mask)/(120.0*base),a=double(p)*inv;v[uint64_t(p)*64+i]=float(std::cos(a));v[uint64_t(p)*64+32+i]=float(std::sin(a));}Buffer stage=dsv4::create_host_buffer_uninitialized(rt,v.size()*4);std::memcpy(stage.mapped,v.data(),v.size()*4);dsv4::flush_buffer_range(rt,stage,0,v.size()*4);dsv4::FiniteQueue q(rt,rt.queue);auto done=q.submit([&](VkCommandBuffer c){VkBufferCopy copy{0,0,v.size()*4};vkfn::CmdCopyBuffer(c,stage.handle,ropebuf.handle,1,&copy);dsv4::transfer_barrier(c,ropebuf);});q.wait(done);destroy_buffer(rt,stage);}
    void sets(){auto emb=tensor("embed",TensorFormat::q8_row,131072,D),head=tensor("lm_head",TensorFormat::q8_row,131072,D),fn=tensor("final_norm",TensorFormat::f32,D);embed_set=kern.set({emb.data,emb.auxiliary,whole(token),whole(word)});bf16_set=kern.set({whole(oe_bf16),whole(oe_rows)});mean_set=kern.set({whole(word),whole(oe_proj),whole(hidden)});for(uint32_t i=0;i<12;++i){oe_quant_sets[i]=kern.set({arena_range(oe_rows,uint64_t(i)*256*4,256*4),whole(oe_q)});auto w=tensor("oe_proj."+std::to_string(i),OE_W,D,256);oe_proj_sets[i]=q4set(whole(oe_q),w,arena_range(oe_proj,uint64_t(i)*D*4,D*4));}final_norm_set=kern.set({whole(hidden),fn.data,whole(norm)});final_q_set=kern.set({whole(norm),whole(quant)});head_set=kern.set({whole(quant),head.data,head.auxiliary,whole(logits)});argmax_set=kern.set({whole(logits),whole(token),whole(argmax)});add_set=kern.set({whole(hidden),whole(shortcut)});moe_reduce=kern.set({whole(expert_out),whole(norm),whole(routing),whole(shortcut)});expert_quant_set=kern.set({whole(expert_mid),whole(expertq)});
        layers.resize(L);for(uint32_t l=0;l<L;++l){auto&r=layers[l];std::string p="layers."+std::to_string(l)+".";for(uint32_t s=0;s<2;++s){auto&as=r.attn[s];auto in=tensor(p+"input_norm."+std::to_string(s),TensorFormat::f32,D),post=tensor(p+"post_norm."+std::to_string(s),TensorFormat::f32,D);as.in_norm=kern.set({whole(hidden),in.data,whole(norm)});as.hidden_q=kern.set({whole(norm),whole(quant)});auto qaw=tensor(p+"attn."+std::to_string(s)+".q_a",W,QL,D),qanorm=tensor(p+"attn."+std::to_string(s)+".q_a_norm",TensorFormat::f32,QL),qbw=tensor(p+"attn."+std::to_string(s)+".q_b",W,H*QHD,QL),kvaw=tensor(p+"attn."+std::to_string(s)+".kv_a",W,KVL+64,D),kvnorm=tensor(p+"attn."+std::to_string(s)+".kv_a_norm",TensorFormat::f32,KVL),kvbw=tensor(p+"attn."+std::to_string(s)+".kv_b",W,H*256,KVL),ow=tensor(p+"attn."+std::to_string(s)+".o",W,D,H*VD);as.qa=q4set(whole(quant),qaw,whole(qa));as.qa_norm=kern.set({whole(qa),qanorm.data,whole(qan)});as.qa_q=kern.set({whole(qan),whole(qaq)});as.qb=q4set(whole(qaq),qbw,whole(qa));as.kva=q4set(whole(quant),kvaw,whole(kva));as.kva_norm=kern.set({arena_range(kva,0,KVL*4),kvnorm.data,whole(kvan)});as.kva_q=kern.set({whole(kvan),whole(kvaq)});as.kvb=q4set(whole(kvaq),kvbw,whole(kvexp));as.store=kern.set({whole(qa),whole(kvexp),whole(kva),whole(kv_cache),whole(ropebuf)});as.attention=kern.set({whole(qa),whole(kv_cache),whole(context)});as.ctx_q=kern.set({whole(context),whole(ctxq)});as.out=q4set(whole(ctxq),ow,whole(hidden),whole(hidden));auto&ds=r.dense[s];ds.norm=kern.set({whole(hidden),post.data,whole(norm)});ds.hidden_q=kern.set({whole(norm),whole(quant)});auto gw=tensor(p+"dense."+std::to_string(s)+".gate",W,DD,D),uw=tensor(p+"dense."+std::to_string(s)+".up",W,DD,D),dw=tensor(p+"dense."+std::to_string(s)+".down",W,D,DD);ds.gate=q4set(whole(quant),gw,whole(dense_gate));ds.up=q4set(whole(quant),uw,whole(dense_up));ds.act=kern.set({whole(dense_gate),whole(dense_up),whole(norm),whole(dense_mid)});ds.act_q=kern.set({whole(dense_mid),whole(denseq)});ds.down=q4set(whole(denseq),dw,whole(hidden),whole(hidden));}
            auto rw=tensor(p+"router",TensorFormat::q8_row,ET,D),rb=tensor(p+"router_bias",TensorFormat::f32,ET);r.router_gemv=kern.set({whole(quant),rw.data,rw.auxiliary,whole(router_logits)});r.router=kern.set({whole(router_logits),rb.data,whole(routing)});r.egate.resize(dcache.slots(l));r.edown.resize(dcache.slots(l));for(uint32_t slot=0;slot<dcache.slots(l);++slot){auto rec=dcache.record(l,slot);r.egate[slot]=kern.set({whole(quant),rec,whole(routing),whole(expert_mid)});r.edown[slot]=kern.set({whole(expertq),rec,whole(routing),whole(expert_out)});}}
    }
    void attention_block(uint32_t l,uint32_t s,uint32_t pos){auto&a=layers[l].attn[s];uint64_t sig=compute.submit([&](VkCommandBuffer c){Push p{1,D,float_bits(1e-5f),0};kern.dispatch(c,kern.p.rms,a.in_norm,&p,1);compute_barrier(c);p={D,128,D/4,D/4};kern.dispatch(c,kern.p.quant,a.hidden_q,&p,D/128);compute_barrier(c);p={QL,D,D/4,0};kern.dispatch(c,kern.p.q4,a.qa,&p,QL/8);p={KVL+64,D,D/4,0};kern.dispatch(c,kern.p.q4,a.kva,&p,(KVL+64)/8);compute_barrier(c);p={1,QL,float_bits(1e-6f),0};kern.dispatch(c,kern.p.rms,a.qa_norm,&p,1);p={1,KVL,float_bits(1e-6f),0};kern.dispatch(c,kern.p.rms,a.kva_norm,&p,1);compute_barrier(c);p={QL,128,QL/4,QL/4};kern.dispatch(c,kern.p.quant,a.qa_q,&p,QL/128);Push pk{KVL,128,KVL/4,KVL/4};kern.dispatch(c,kern.p.quant,a.kva_q,&pk,KVL/128);compute_barrier(c);p={H*QHD,QL,QL/4,0};kern.dispatch(c,kern.p.q4,a.qb,&p,H*QHD/8);p={H*256,KVL,KVL/4,0};kern.dispatch(c,kern.p.q4,a.kvb,&p,H*256/8);compute_barrier(c);p={l*2+s,pos,context_capacity_,0};kern.dispatch(c,kern.p.store,a.store,&p,H);compute_barrier(c);kern.dispatch(c,kern.p.attention,a.attention,&p,H);compute_barrier(c);p={H*VD,128,H*VD/4,H*VD/4};kern.dispatch(c,kern.p.quant,a.ctx_q,&p,H*VD/128);compute_barrier(c);p={D,H*VD,H*VD/4,0};kern.dispatch(c,kern.p.q4r,a.out,&p,D/8);compute_barrier(c);});compute.wait(sig);}
    void dense(uint32_t l,uint32_t s){
        auto&d=layers[l].dense[s];
        uint64_t sig=compute.submit([&](VkCommandBuffer c){
            Push p{1,D,float_bits(1e-5f),0};
            kern.dispatch(c,kern.p.rms,d.norm,&p,1);compute_barrier(c);
            p={D,128,D/4,D/4};
            kern.dispatch(c,kern.p.quant,d.hidden_q,&p,D/128);compute_barrier(c);
            p={DD,D,D/4,0};
            kern.dispatch(c,kern.p.dense_matrix,d.gate,&p,DD/8);
            kern.dispatch(c,kern.p.dense_matrix,d.up,&p,DD/8);compute_barrier(c);
            p={DD,float_bits(3.402823466e38f),0,0};
            kern.dispatch(c,kern.p.swiglu,d.act,&p,DD/64);compute_barrier(c);
            p={DD,128,DD/4,DD/4};
            kern.dispatch(c,kern.p.quant,d.act_q,&p,DD/128);compute_barrier(c);
            p={D,DD,DD/4,0};
            kern.dispatch(c,kern.p.dense_matrix_r,d.down,&p,D/8);compute_barrier(c);
        });
        compute.wait(sig);
    }
    void moe(uint32_t l) {
        auto& r = layers[l];
        auto& post = r.dense[0];
        uint64_t sig = compute.submit([&](VkCommandBuffer c) {
            Push p{1,D,float_bits(1e-5f),0};
            kern.dispatch(c,kern.p.rms,post.norm,&p,1);
            compute_barrier(c);
            p={D,128,D/4,D/4};
            kern.dispatch(c,kern.p.quant,post.hidden_q,&p,D/128);
            compute_barrier(c);
            p={ET,D,D/4,0};
            kern.dispatch(c,kern.p.q8,r.router_gemv,&p,(ET+7)/8);
            compute_barrier(c);
            p={ET,active,0,0};
            kern.dispatch(c,kern.p.router,r.router,&p,1);
            compute_barrier(c);
        });
        compute.wait(sig);
        invalidate_buffer(rt,routing);

        auto* words=static_cast<uint32_t*>(routing.mapped);
        std::array<uint32_t,K> ex{};
        std::array<float,K> w{};
        uint32_t physical=0;
        for(uint32_t i=0;i<active;++i) {
            float f;
            std::memcpy(&f,words+16+i,4);
            if(words[i]<E) {
                ex[physical]=words[i];
                w[physical++]=f;
            }
        }
        const uint32_t filler=physical?ex[0]:0;
        for(uint32_t i=physical;i<K;++i) {
            ex[i]=filler;
            w[i]=0;
        }
        for(uint32_t i=0;i<K;++i) {
            words[i]=ex[i];
            std::memcpy(words+16+i,&w[i],4);
        }
        flush_buffer(rt,routing);

        if(!physical) {
            sig=compute.submit([&](VkCommandBuffer c) {
                Push pr{D,0,0,0};
                kern.dispatch(c,kern.p.reduce,moe_reduce,&pr,D/64);
                compute_barrier(c);
            });
            compute.wait(sig);
            return;
        }

        auto sel=dcache.resolve(l,ex,physical);
        std::array<void*,K> dest{};
        for(uint32_t i=0;i<physical;++i) dest[i]=staging[i].mapped;
        auto src=hcache.resolve_batch(l,ex,sel.misses,dest);
        std::vector<uint32_t> copy;
        for(uint32_t i=0;i<physical;++i) if(sel.misses[i]) {
            if(!src.direct[i]) {
                std::memcpy(staging[i].mapped,src.pointers[i],REC);
                hostcopy+=REC;
            }
            dsv4::flush_buffer_range(rt,staging[i],0,REC);
            copy.push_back(i);
        }
        uint64_t ready=0;
        if(!copy.empty()) {
            ready=transfer.submit([&](VkCommandBuffer c) {
                for(auto i:copy) {
                    VkBufferCopy x{0,uint64_t(sel.slots[i])*REC,REC};
                    vkfn::CmdCopyBuffer(c,staging[i].handle,
                        dcache.arena(l).handle,1,&x);
                }
                dsv4::transfer_barrier(c,dcache.arena(l));
            });
            h2d+=uint64_t(copy.size())*REC;
            transfer.wait(ready);
            for(auto i:copy) if(src.direct[i]&&src.pointers[i]) {
                std::memcpy(const_cast<uint8_t*>(src.pointers[i]),
                    staging[i].mapped,REC);
                hostcopy+=REC;
            }
        }

        sig=compute.submit([&](VkCommandBuffer c) {
            for(uint32_t i=0;i<physical;++i) {
                Push p{i,D/4,0,0};
                kern.dispatch(c,kern.p.egate,r.egate[sel.slots[i]],&p,ED/8);
            }
            compute_barrier(c);
            Push pq{physical*ED,128,physical*ED/4,physical*ED/4};
            kern.dispatch(c,kern.p.quant,expert_quant_set,&pq,physical*ED/128);
            compute_barrier(c);
            for(uint32_t i=0;i<physical;++i) {
                Push p{i,physical*ED/4,0,0};
                kern.dispatch(c,kern.p.edown,r.edown[sel.slots[i]],&p,D/8);
            }
            compute_barrier(c);
            Push pr{D,physical,0,0};
            kern.dispatch(c,kern.p.reduce,moe_reduce,&pr,D/64);
            compute_barrier(c);
        });
        compute.wait(sig);
    }
    const Runtime&rt;DeviceWeights weights;ExpertFile efile;HostExpertCache hcache;DeviceExpertCache dcache;OeFile oefile;Kernels kern;dsv4::FiniteQueue compute,transfer;std::vector<Buffer>staging;Buffer oe_stage{};Buffer token{},routing{},word{},oe_bf16{},oe_rows{},oe_q{},oe_proj{},hidden{},norm{},quant{},qa{},qan{},qaq{},kva{},kvan{},kvaq{},kvexp{},context{},ctxq{},dense_gate{},dense_up{},dense_mid{},denseq{},router_logits{},shortcut{},expert_mid{},expertq{},expert_out{},kv_cache{},ropebuf{},logits{},argmax{};std::vector<LayerSets>layers;std::array<VkDescriptorSet,12>oe_quant_sets{},oe_proj_sets{};VkDescriptorSet embed_set{},bf16_set{},mean_set{},final_norm_set{},final_q_set{},head_set{},argmax_set{},add_set{},moe_reduce{},expert_quant_set{};uint64_t activation_bytes=0,h2d=0,hostcopy=0;uint32_t active=K,context_capacity_=0;
};
}

int main(int argc,char**argv){Runtime rt{};try{if(argc<3){std::cerr<<"usage: xtllm_longcat.exe <runtime-dir> <prompt> [new-tokens]\n";return 2;}std::filesystem::path dir=argv[1];dsv4::ReadOnlyMapping tf((dir/"tokenizer.ovb").string());qwen35::Tokenizer tok(tf);if(std::strcmp(argv[2],"--tokenize")==0){if(argc<4)throw std::runtime_error("tokenize text required");std::vector<uint32_t> ids{47};auto body=tok.encode_text(argv[3]);ids.insert(ids.end(),body.begin(),body.end());ids.push_back(48);std::cout<<"tokens:";for(auto id:ids)std::cout<<' '<<id;std::cout<<'\n';return 0;}const char* shared_name=std::getenv("LONGCAT_SHARED_MODEL");qwen35::SharedIndex index(dir/(shared_name?shared_name:"model-q4g64.ovs"));qwen35::ExpertFile inspect(dir/"experts-q4g64.ovx");rt=create_runtime();uint32_t count=argc>3?uint32_t(std::stoul(argv[3])):16;if(!count)throw std::runtime_error("new-tokens must be positive");double gib=std::getenv("LONGCAT_RAM_GIB")?std::stod(std::getenv("LONGCAT_RAM_GIB")):16.0;uint32_t slots=std::getenv("LONGCAT_DEVICE_SLOTS_PER_LAYER")?uint32_t(std::stoul(std::getenv("LONGCAT_DEVICE_SLOTS_PER_LAYER"))):80;std::vector<uint32_t>prompt=tok.chat_prompt(argv[2],false);uint64_t required_context=uint64_t(prompt.size())+count-1,requested_context=required_context;if(const char* value=std::getenv("LONGCAT_CONTEXT_TOKENS"))requested_context=std::stoull(value);if(requested_context<required_context)throw std::runtime_error("LONGCAT_CONTEXT_TOKENS is smaller than prompt plus requested generation");if(requested_context>UINT32_MAX)throw std::runtime_error("LongCat requested context is too large");uint32_t context_capacity=uint32_t(requested_context);std::vector<uint32_t>history,result;double seconds=0;uint64_t ram=0,vram=0,disk=0,h2d=0,hcopy=0,dh=0,dm=0,rh=0,rm=0;uint32_t hs=0,ar=0;{longcat::Engine e(rt,index,dir/"experts-q4g64.ovx",dir/"ngram-bf16.ove",std::filesystem::absolute(argv[0]).parent_path(),uint64_t(gib*(1ull<<30)),slots,context_capacity);ar=e.active_routes();uint32_t next=0;for(uint32_t i=0;i<prompt.size();++i)next=e.run(prompt[i],i,history);result.push_back(next);std::cout<<tok.decode_piece(next)<<std::flush;auto start=std::chrono::steady_clock::now();for(uint32_t i=1;i<count&&!tok.is_eos(next);++i){next=e.run(next,uint32_t(history.size()),history);result.push_back(next);std::cout<<tok.decode_piece(next)<<std::flush;}seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();ram=e.ram_bytes();vram=e.vram_bytes();disk=e.disk_bytes();h2d=e.h2d_bytes();hcopy=e.host_copy_bytes();dh=e.device_hits();dm=e.device_misses();rh=e.ram_hits();rm=e.ram_misses();hs=e.host_slots();}
    uint64_t timed=result.size()>1?result.size()-1:0;
    double d=timed?double(timed):1;
#ifdef XTLLM_LONGCAT_DENSE_OUT_Q4
    const char* precision="Q4G64T experts/dense/attention-output, Q8 QKV/n-gram/embedding/head/router, BF16 n-gram rows";
#elif defined(XTLLM_LONGCAT_DENSE_Q4)
    const char* precision="Q4G64T experts/dense, Q8 attention/n-gram/embedding/head/router, BF16 n-gram rows";
#elif defined(XTLLM_LONGCAT_Q4_SHARED)
    const char* precision="Q4G64T experts/attention/dense/n-gram projections, Q8 embedding/head/router, BF16 n-gram rows";
#else
    const char* precision="Q4G64T experts, Q8 attention/dense/n-gram projections/embedding/head/router, BF16 n-gram rows";
#endif
    std::cout<<"\nmodel: LongCat-Flash-Lite-Sparse\nprecision: "<<precision
             <<"\nactive MoE routes: "<<ar<<" / "<<longcat::K
             <<"\nallocated context capacity: "<<context_capacity<<" tokens"
             <<"\nRAM budget/actual GiB: "<<gib<<" / "<<double(ram)/(1ull<<30)<<" ("<<hs<<" expert records)"
             <<"\npeak device allocation estimate GiB: "<<double(vram)/(1ull<<30)
             <<"\ndecode throughput: "<<(seconds?timed/seconds:0)<<" tok/s"
             <<"\ndevice hits/misses: "<<dh<<'/'<<dm
             <<"\nRAM hits/misses: "<<rh<<'/'<<rm
             <<"\nexpert SSD / host-copy / H2D bytes per output: "<<disk/d<<" / "<<hcopy/d<<" / "<<h2d/d
             <<"\ntoken ids:";
    for(auto x:result)std::cout<<' '<<x;
    std::cout<<'\n';
    vkfn::DestroyDevice(rt.device,nullptr);vkfn::DestroyInstance(rt.instance,nullptr);FreeLibrary(rt.loader);return 0;
}catch(const std::exception&e){std::cerr<<"LongCat runtime error: "<<e.what()<<'\n';if(rt.device)vkfn::DestroyDevice(rt.device,nullptr);if(rt.instance)vkfn::DestroyInstance(rt.instance,nullptr);if(rt.loader)FreeLibrary(rt.loader);return 1;}}
