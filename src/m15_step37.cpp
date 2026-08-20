#define OVLLM_DSV4_RUNTIME_ONLY
#include "m13_deepseek_v4.cpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <future>
#include <unordered_map>

// Step-3.7-Flash text-backbone executor.  All command buffers are finite and
// every host/device rendezvous is a bounded Vulkan timeline wait.
namespace step37 {

constexpr uint32_t kDim=4096, kMoeDim=1280, kLayers=45, kMoeFirst=3;
constexpr uint32_t kMoeLayers=42, kExperts=288, kTopK=8, kVocabulary=128896;
constexpr uint32_t kKvHeads=8, kHeadDim=128, kMaximumContext=2048;
constexpr uint64_t kHeaderBytes=4096, kExpertRecordBytes=8355840;
constexpr uint64_t kGateScale=0, kGateWeight=163840, kUpScale=2785280;
constexpr uint64_t kUpWeight=2949120, kDownScale=5570560, kDownWeight=5734400;

using dsv4::SharedHeader;
using dsv4::GroupEntry;
using dsv4::TensorEntry;
using dsv4::ExpertHeader;
using dsv4::TensorFormat;

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct TensorDevice {
    DescriptorRange data{}, auxiliary{};
    TensorFormat format=TensorFormat::f32;
    uint32_t rank=0;
    std::array<uint64_t,8> shape{};
};

static void read_at(std::ifstream& input, uint64_t offset, void* output,
                    size_t bytes, const char* label) {
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(static_cast<char*>(output), static_cast<std::streamsize>(bytes));
    if (!input) throw std::runtime_error(std::string("Step shared read failed: ")+label);
}

class SharedIndex {
public:
    explicit SharedIndex(const std::filesystem::path& path):path_(path){
        std::ifstream input(path,std::ios::binary);
        if(!input) throw std::runtime_error("Could not open Step shared container");
        read_at(input,0,&header_,sizeof(header_),"header");
        const bool main=std::memcmp(header_.magic,"OVS37SH\0",8)==0;
        const bool mtp=std::memcmp(header_.magic,"OVS37MT\0",8)==0;
        if((!main&&!mtp)||header_.version!=1||header_.header_bytes!=kHeaderBytes||
           header_.tensor_entry_bytes!=sizeof(TensorEntry)||header_.dimension!=kDim||
           header_.vocabulary!=kVocabulary||(main&&(header_.moe_dimension!=kMoeDim||
           header_.layers!=kLayers||header_.experts!=kExperts||header_.top_k!=kTopK||
           header_.expert_record_bytes!=kExpertRecordBytes))||(mtp&&(header_.layers!=3||
           header_.group_count!=3)))throw std::runtime_error("Unsupported Step-3.7 shared container");
        const uint64_t actual=std::filesystem::file_size(path);
        if(header_.file_bytes!=actual || (main&&header_.group_count!=kLayers+1) ||
           header_.tensor_count==0 || header_.tensor_count>4096)
            throw std::runtime_error("Invalid Step shared container bounds");
        groups_.resize(static_cast<size_t>(header_.group_count));
        entries_.resize(static_cast<size_t>(header_.tensor_count));
        read_at(input,header_.group_table_offset,groups_.data(),groups_.size()*sizeof(GroupEntry),"groups");
        read_at(input,header_.tensor_table_offset,entries_.data(),entries_.size()*sizeof(TensorEntry),"tensors");
        for(uint32_t gi=0;gi<groups_.size();++gi){
            const auto& g=groups_[gi];
            if(g.data_begin<kHeaderBytes || g.data_end<=g.data_begin || g.data_end>actual ||
               (g.data_begin&4095u)||(g.data_end&4095u)||
               uint64_t(g.first_tensor)+g.tensor_count>entries_.size())
                throw std::runtime_error("Invalid Step shared group");
            for(uint32_t i=0;i<g.tensor_count;++i){
                const TensorEntry& e=entries_[g.first_tensor+i];
                size_t n=0;while(n<sizeof(e.name)&&e.name[n])++n;
                if(n==sizeof(e.name))throw std::runtime_error("Unterminated Step tensor name");
                Entry x{};x.entry=e;x.group=gi;
                if(!by_name_.emplace(std::string(e.name,n),x).second)
                    throw std::runtime_error("Duplicate Step tensor");
            }
        }
    }
    struct Entry{TensorEntry entry{};uint32_t group=0;};
    const Entry& require(const std::string& name)const{
        auto it=by_name_.find(name);if(it==by_name_.end())throw std::runtime_error("Missing Step tensor: "+name);return it->second;
    }
    const SharedHeader& header()const{return header_;}
    const std::vector<GroupEntry>& groups()const{return groups_;}
    const std::filesystem::path& path()const{return path_;}
private:
    std::filesystem::path path_;SharedHeader header_{};
    std::vector<GroupEntry> groups_;std::vector<TensorEntry> entries_;
    std::unordered_map<std::string,Entry> by_name_;
};

static HANDLE open_unbuffered(const std::filesystem::path& path){
    HANDLE h=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,
                         FILE_FLAG_NO_BUFFERING|FILE_FLAG_RANDOM_ACCESS,nullptr);
    if(h==INVALID_HANDLE_VALUE)throw std::runtime_error("Could not open unbuffered Step weight file");
    return h;
}

static void read_unbuffered(HANDLE file,uint64_t offset,void* destination,uint64_t bytes){
    if((offset&4095u)||(bytes&4095u)||(reinterpret_cast<uintptr_t>(destination)&4095u))
        throw std::runtime_error("Unaligned Step unbuffered read");
    LARGE_INTEGER where{};where.QuadPart=offset;
    if(!SetFilePointerEx(file,where,nullptr,FILE_BEGIN))throw std::runtime_error("Step seek failed");
    uint8_t* out=static_cast<uint8_t*>(destination);
    while(bytes){DWORD part=static_cast<DWORD>(std::min<uint64_t>(bytes,64ull*1024*1024));DWORD got=0;
        if(!ReadFile(file,out,part,&got,nullptr)||got!=part)throw std::runtime_error("Step unbuffered read failed");
        out+=part;bytes-=part;
    }
}

class DeviceWeights {
public:
    DeviceWeights(const Runtime& runtime,const SharedIndex& index,uint32_t group_limit=UINT32_MAX):runtime_(runtime),index_(index),transfer_(runtime,runtime.queue){
        HANDLE file=open_unbuffered(index.path());
        Buffer staging=dsv4::create_host_buffer_uninitialized(runtime,64ull*1024*1024);
        try{
            groups_.resize(index.groups().size());
            for(uint32_t gi=0;gi<groups_.size()&&gi<group_limit;++gi){
                const auto& group=index.groups()[gi];const uint64_t bytes=group.data_end-group.data_begin;
                groups_[gi]=create_device_buffer(runtime,bytes);device_bytes_+=groups_[gi].allocation_size;
                for(uint64_t done=0;done<bytes;done+=staging.size){
                    uint64_t part=std::min<uint64_t>(staging.size,bytes-done);
                    read_unbuffered(file,group.data_begin+done,staging.mapped,part);
                    dsv4::flush_buffer_range(runtime,staging,0,part);
                    uint64_t signal=transfer_.submit([&](VkCommandBuffer command){
                        VkBufferCopy copy{0,done,part};vkfn::CmdCopyBuffer(command,staging.handle,groups_[gi].handle,1,&copy);
                        dsv4::transfer_barrier(command,groups_[gi]);
                    });transfer_.wait(signal);
                }
            }
        }catch(...){CloseHandle(file);destroy_buffer(runtime,staging);throw;}
        CloseHandle(file);destroy_buffer(runtime,staging);
    }
    ~DeviceWeights(){for(auto& b:groups_)destroy_buffer(runtime_,b);}
    TensorDevice tensor(const std::string& name)const{
        const auto& x=index_.require(name);const auto& g=index_.groups()[x.group];const auto& e=x.entry;
        auto make=[&](uint64_t off,uint64_t bytes){
            if(!bytes)return whole(groups_[x.group]);
            if(off<g.data_begin||off+bytes>g.data_end)throw std::runtime_error("Step tensor outside group");
            return arena_range(groups_[x.group],off-g.data_begin,bytes);
        };
        TensorDevice t;t.data=make(e.data_offset,e.data_bytes);t.auxiliary=make(e.auxiliary_offset,e.auxiliary_bytes);
        t.format=static_cast<TensorFormat>(e.dtype);t.rank=e.rank;
        std::copy(std::begin(e.shape),std::end(e.shape),t.shape.begin());return t;
    }
    uint64_t device_bytes()const{return device_bytes_;}
private:
    const Runtime& runtime_;const SharedIndex& index_;dsv4::FiniteQueue transfer_;
    std::vector<Buffer> groups_;uint64_t device_bytes_=0;
};

class ExpertFile {
public:
    explicit ExpertFile(const std::filesystem::path& path):path_(path){
        std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("Could not open Step experts");
        read_at(input,0,&header_,sizeof(header_),"expert header");
        if(std::memcmp(header_.magic,"OVS37EX\0",8)!=0||header_.version!=1||
           header_.header_bytes!=kHeaderBytes||header_.dimension!=kDim||
           header_.moe_dimension!=kMoeDim||header_.layers!=kMoeLayers||
           header_.experts!=kExperts||header_.record_bytes!=kExpertRecordBytes||
           header_.file_bytes!=std::filesystem::file_size(path))
            throw std::runtime_error("Unsupported Step expert container");
        file_=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,
                          FILE_FLAG_NO_BUFFERING|FILE_FLAG_RANDOM_ACCESS|FILE_FLAG_OVERLAPPED,nullptr);
        if(file_==INVALID_HANDLE_VALUE)throw std::runtime_error("Could not open overlapped Step expert file");
    }
    ~ExpertFile(){if(file_!=INVALID_HANDLE_VALUE)CloseHandle(file_);}
    uint64_t offset(uint32_t layer,uint32_t expert)const{
        if(layer<kMoeFirst||layer>=kLayers||expert>=kExperts)throw std::runtime_error("Invalid Step expert key");
        return header_.core_offset+(uint64_t(layer-kMoeFirst)*kExperts+expert)*kExpertRecordBytes;
    }
    void read_batch(const std::vector<uint64_t>& offsets,const std::vector<void*>& outputs){
        if(offsets.size()!=outputs.size()||offsets.size()>kTopK)throw std::runtime_error("Invalid Step I/O batch");
        if(offsets.empty())return;
        std::array<OVERLAPPED,kTopK> operations{};std::array<HANDLE,kTopK> events{};events.fill(nullptr);
        try{
            for(uint32_t i=0;i<offsets.size();++i){
                events[i]=CreateEventW(nullptr,TRUE,FALSE,nullptr);if(!events[i])throw std::runtime_error("Step I/O event creation failed");
                operations[i].Offset=DWORD(offsets[i]);operations[i].OffsetHigh=DWORD(offsets[i]>>32);operations[i].hEvent=events[i];
                BOOL ok=ReadFile(file_,outputs[i],DWORD(kExpertRecordBytes),nullptr,&operations[i]);
                if(!ok&&GetLastError()!=ERROR_IO_PENDING)throw std::runtime_error("Step overlapped expert read submission failed");
            }
            DWORD waited=WaitForMultipleObjects(DWORD(offsets.size()),events.data(),TRUE,60000);
            if(waited!=WAIT_OBJECT_0)throw std::runtime_error("Step overlapped expert read timed out");
            for(uint32_t i=0;i<offsets.size();++i){DWORD got=0;if(!GetOverlappedResult(file_,&operations[i],&got,FALSE)||got!=kExpertRecordBytes)throw std::runtime_error("Step overlapped expert read failed");}
        }catch(...){for(HANDLE event:events)if(event)CloseHandle(event);throw;}
        for(HANDLE event:events)if(event)CloseHandle(event);
    }
private:std::filesystem::path path_;ExpertHeader header_{};HANDLE file_=INVALID_HANDLE_VALUE;
};

class HostExpertCache {
public:
    struct Batch{std::array<const uint8_t*,kTopK> pointers{};std::array<bool,kTopK> direct{};};
    HostExpertCache(ExpertFile& file,uint64_t budget_bytes):file_(file){
        const uint64_t reserve=budget_bytes>512ull*1024*1024?budget_bytes-512ull*1024*1024:0;
        slots_=static_cast<uint32_t>(std::min<uint64_t>(reserve/kExpertRecordBytes,kMoeLayers*kExperts));
        if(slots_<kTopK)throw std::runtime_error("Step RAM budget is too small");
        bytes_=uint64_t(slots_)*kExpertRecordBytes;
        base_=static_cast<uint8_t*>(VirtualAlloc(nullptr,bytes_,MEM_RESERVE,PAGE_READWRITE));
        if(!base_)throw std::runtime_error("Could not reserve Step RAM cache");
        entries_.resize(slots_);locations_.assign(kMoeLayers*kExperts,-1);frequency_.assign(locations_.size(),0);
    }
    ~HostExpertCache(){if(base_)VirtualFree(base_,0,MEM_RELEASE);}
    Batch resolve_batch(uint32_t layer,const std::array<uint32_t,kTopK>& experts,const std::array<bool,kTopK>& needed,const std::array<void*,kTopK>& direct_destinations){
        Batch result{};std::vector<bool> reserved(slots_);std::vector<uint64_t> offsets;std::vector<void*> destinations;
        for(uint32_t rank=0;rank<kTopK;++rank)if(needed[rank]){
            uint32_t expert=experts[rank],key=(layer-kMoeFirst)*kExperts+expert;++frequency_[key];int32_t location=locations_[key];
            if(location>=0){++hits_;entries_[location].age=++clock_;reserved[location]=true;result.pointers[rank]=base_+uint64_t(location)*kExpertRecordBytes;continue;}
            ++misses_;uint32_t victim=UINT32_MAX;
            for(uint32_t s=0;s<slots_;++s)if(!reserved[s]&&(victim==UINT32_MAX||entries_[s].key<0||
                (entries_[victim].key>=0&&(lru_?entries_[s].age<entries_[victim].age:
                (frequency_[entries_[s].key]<frequency_[entries_[victim].key]||
                (frequency_[entries_[s].key]==frequency_[entries_[victim].key]&&entries_[s].age<entries_[victim].age)))))){victim=s;if(entries_[s].key<0)break;}
            if(victim==UINT32_MAX)throw std::runtime_error("No Step RAM cache victim");reserved[victim]=true;Entry& slot=entries_[victim];if(slot.key>=0)locations_[slot.key]=-1;uint8_t* pointer=base_+uint64_t(victim)*kExpertRecordBytes;
            if(!slot.committed){if(!VirtualAlloc(pointer,kExpertRecordBytes,MEM_COMMIT,PAGE_READWRITE))throw std::runtime_error("Step RAM cache commit failed");slot.committed=true;++committed_;}
            slot.key=static_cast<int32_t>(key);slot.age=++clock_;locations_[key]=static_cast<int32_t>(victim);result.pointers[rank]=pointer;result.direct[rank]=true;offsets.push_back(file_.offset(layer,expert));destinations.push_back(direct_destinations[rank]);
        }
        file_.read_batch(offsets,destinations);disk_bytes_+=uint64_t(offsets.size())*kExpertRecordBytes;return result;
    }
    uint64_t hits()const{return hits_;}uint64_t misses()const{return misses_;}
    uint64_t disk_bytes()const{return disk_bytes_;}uint64_t committed_bytes()const{return uint64_t(committed_)*kExpertRecordBytes;}
    uint32_t capacity()const{return slots_;}
    void reset_metrics(){hits_=0;misses_=0;disk_bytes_=0;}
private:
    struct Entry{int32_t key=-1;uint64_t age=0;bool committed=false;};
    ExpertFile& file_;uint8_t* base_=nullptr;uint64_t bytes_=0;uint32_t slots_=0,committed_=0;
    std::vector<Entry> entries_;std::vector<int32_t> locations_;std::vector<uint32_t> frequency_;
    uint64_t clock_=0,hits_=0,misses_=0,disk_bytes_=0;bool lru_=std::getenv("STEP_CACHE_LRU")!=nullptr;
};

class DeviceExpertCache {
public:
    DeviceExpertCache(const Runtime& runtime,uint32_t slots):runtime_(runtime),slots_(slots){
        layers_.resize(kMoeLayers);
        for(auto& layer:layers_){layer.arena=create_device_buffer(runtime,uint64_t(slots)*kExpertRecordBytes);layer.entries.resize(slots);device_bytes_+=layer.arena.allocation_size;}
    }
    ~DeviceExpertCache(){for(auto& l:layers_)destroy_buffer(runtime_,l.arena);}
    struct Selection{std::array<uint32_t,kTopK> slots{};std::array<bool,kTopK> misses{};};
    Selection resolve(uint32_t layer,const std::array<uint32_t,kTopK>& experts){
        Layer& cache=layers_.at(layer-kMoeFirst);Selection out{};std::vector<bool> reserved(slots_);
        for(uint32_t r=0;r<kTopK;++r){++cache.frequency[experts[r]];out.slots[r]=UINT32_MAX;
            for(uint32_t s=0;s<slots_;++s)if(cache.entries[s].expert==int32_t(experts[r])){out.slots[r]=s;reserved[s]=true;cache.entries[s].age=++clock_;++hits_;break;}}
        for(uint32_t r=0;r<kTopK;++r)if(out.slots[r]==UINT32_MAX){++misses_;uint32_t victim=UINT32_MAX;
            for(uint32_t s=0;s<slots_;++s)if(!reserved[s]&&(victim==UINT32_MAX||cache.entries[s].expert<0||
               (cache.entries[victim].expert>=0&&(lru_?cache.entries[s].age<cache.entries[victim].age:
               (cache.frequency[cache.entries[s].expert]<cache.frequency[cache.entries[victim].expert]||
               (cache.frequency[cache.entries[s].expert]==cache.frequency[cache.entries[victim].expert]&&cache.entries[s].age<cache.entries[victim].age)))))){victim=s;if(cache.entries[s].expert<0)break;}
            if(victim==UINT32_MAX)throw std::runtime_error("No Step device expert victim");cache.entries[victim].expert=int32_t(experts[r]);cache.entries[victim].age=++clock_;reserved[victim]=true;out.slots[r]=victim;out.misses[r]=true;}
        return out;
    }
    std::array<bool,kTopK> missing(uint32_t layer,const std::array<uint32_t,kTopK>& experts)const{std::array<bool,kTopK> result{};const Layer& cache=layers_.at(layer-kMoeFirst);for(uint32_t r=0;r<kTopK;++r){result[r]=true;for(const Entry& entry:cache.entries)if(entry.expert==int32_t(experts[r])){result[r]=false;break;}}return result;}
    DescriptorRange record(uint32_t layer,uint32_t slot)const{return arena_range(layers_.at(layer-kMoeFirst).arena,uint64_t(slot)*kExpertRecordBytes,kExpertRecordBytes);}
    Buffer& arena(uint32_t layer){return layers_.at(layer-kMoeFirst).arena;}
    uint32_t slots()const{return slots_;}uint64_t hits()const{return hits_;}uint64_t misses()const{return misses_;}uint64_t device_bytes()const{return device_bytes_;}
    void reset_metrics(){hits_=0;misses_=0;}
private:
    struct Entry{int32_t expert=-1;uint64_t age=0;};struct Layer{Buffer arena{};std::vector<Entry> entries;std::array<uint32_t,kExperts> frequency{};};
    const Runtime& runtime_;uint32_t slots_;std::vector<Layer> layers_;uint64_t clock_=0,hits_=0,misses_=0,device_bytes_=0;bool lru_=std::getenv("STEP_CACHE_LRU")!=nullptr;
};

struct Pipelines{VkPipeline embedding{},rms{},quant{},q4{},q4_residual{},q8{},swiglu{},router{},qk{},store_v{},attention{},head_gate{},expert_gate{},expert_down{},reduce{},argmax{},mtp_fuse{};};
class Kernels{
public:
    Kernels(const Runtime& runtime,const std::filesystem::path& directory):runtime_(runtime),resources_(create_compute_resources(runtime,8192)),dummy_(create_device_buffer(runtime,4096)){
        auto load=[&](const char* name){return dsv4::create_dsv4_pipeline(runtime_,resources_,directory/(std::string(name)+".comp.spv"),64);};
        p_.embedding=load("dsv4_embedding");p_.rms=load("step37_rmsnorm");p_.quant=load("dsv4_quantize_q8");
        p_.q4=load("dsv4_q4g64t_gemv");p_.q4_residual=load("dsv4_q4g64t_gemv_residual");p_.q8=load("dsv4_q8_gemv");p_.swiglu=load("step37_swiglu");
        p_.router=load("step37_router_top8");p_.qk=load("step37_qk_rope_cache");p_.store_v=load("step37_store_value");p_.attention=load("step37_attention");p_.head_gate=load("step37_head_gate");
        p_.expert_gate=load("step37_expert_gate_up_q4");p_.expert_down=load("step37_expert_down_q4");p_.reduce=load("step37_reduce");p_.argmax=load("dsv4_greedy_argmax");p_.mtp_fuse=load("step37_mtp_fuse");
    }
    ~Kernels(){for(auto p:resources_.pipelines)vkfn::DestroyPipeline(runtime_.device,p,nullptr);for(auto m:resources_.shader_modules)vkfn::DestroyShaderModule(runtime_.device,m,nullptr);if(resources_.descriptor_pool)vkfn::DestroyDescriptorPool(runtime_.device,resources_.descriptor_pool,nullptr);if(resources_.pipeline_layout)vkfn::DestroyPipelineLayout(runtime_.device,resources_.pipeline_layout,nullptr);if(resources_.descriptor_layout)vkfn::DestroyDescriptorSetLayout(runtime_.device,resources_.descriptor_layout,nullptr);destroy_buffer(runtime_,dummy_);}
    VkDescriptorSet set(std::initializer_list<DescriptorRange> list){std::array<DescriptorRange,6> a; a.fill(whole(dummy_));uint32_t i=0;for(auto x:list){if(i<6)a[i++]=x;}return dsv4::create_dsv4_set(runtime_,resources_,a);}
    void dispatch(VkCommandBuffer c,VkPipeline p,VkDescriptorSet s,const void* pc,uint32_t x,uint32_t y=1){dsv4::dispatch_dsv4(c,resources_,p,s,pc,x,y);}
    const Pipelines& p()const{return p_;}DescriptorRange dummy()const{return whole(dummy_);}
private:const Runtime& runtime_;ComputeResources resources_{};Buffer dummy_{};Pipelines p_{};
};

struct Push{uint32_t a,b,c,d;};

class StepEngine{
public:
    StepEngine(const Runtime& runtime,const SharedIndex& index,const std::filesystem::path& expert_path,
               const std::filesystem::path& shader_dir,uint64_t ram_budget,uint32_t device_slots)
      :runtime_(runtime),weights_(runtime,index),expert_file_(expert_path),host_cache_(expert_file_,ram_budget),
       device_cache_(runtime,device_slots),kernels_(runtime,shader_dir),compute_(runtime,runtime.queue),transfer_(runtime,runtime.secondary_queue){
        allocate_buffers();make_rope();build_sets(index);
        staging_.resize(2*kTopK);for(auto& b:staging_)b=dsv4::create_host_buffer_uninitialized(runtime,kExpertRecordBytes);
        selected_slots_.fill(0);
    }
    ~StepEngine(){for(auto& b:staging_)destroy_buffer(runtime_,b);destroy_all();}
    std::vector<uint32_t> generate(const dsv4::Tokenizer& tokenizer,const std::vector<uint32_t>& prompt,uint32_t count){
        uint32_t position=0,next=0;std::vector<uint32_t> output;
        for(uint32_t token:prompt){next=run_token(token,position++);}
        reset_decode_metrics();
        auto start=std::chrono::steady_clock::now();
        for(uint32_t i=0;i<count;++i){output.push_back(next);std::cout<<tokenizer.decode_piece(next)<<std::flush;if(next==tokenizer.eos()||i+1==count)break;next=run_token(next,position++);}
        decode_seconds_=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return output;
    }
    uint64_t device_hits()const{return device_cache_.hits();}uint64_t device_misses()const{return device_cache_.misses();}
    uint64_t ram_hits()const{return host_cache_.hits();}uint64_t ram_misses()const{return host_cache_.misses();}
    uint64_t disk_bytes()const{return host_cache_.disk_bytes();}uint64_t ram_bytes()const{return host_cache_.committed_bytes()+uint64_t(staging_.size())*kExpertRecordBytes;}
    uint64_t vram_bytes()const{return weights_.device_bytes()+device_cache_.device_bytes()+activation_device_bytes_;}
    uint64_t transfer_bytes()const{return transfer_bytes_;}double decode_seconds()const{return decode_seconds_;}
    uint64_t prediction_matches()const{return prediction_matches_;}uint64_t prediction_total()const{return prediction_total_;}
    double pre_seconds()const{return pre_seconds_;}double acquire_seconds()const{return acquire_seconds_;}double post_seconds()const{return post_seconds_;}
    uint32_t host_slots()const{return host_cache_.capacity();}uint32_t device_slots()const{return device_cache_.slots();}
    uint32_t process(uint32_t token,uint32_t position){return run_token(token,position);}
    std::array<uint32_t,2> verify2(const std::array<uint32_t,2>& tokens,uint32_t position){
        if(position+1>=kMaximumContext)throw std::runtime_error("Step two-token verifier context cap reached");
        for(uint32_t row=0;row<2;++row)embed_token(tokens[row],&batch_hidden_[row]);
        for(uint32_t l=0;l<kLayers;++l)run_layer2(l,position);
        std::array<uint32_t,2> result{};for(uint32_t row=0;row<2;++row){
            uint64_t signal=compute_.submit([&](VkCommandBuffer c){record_hidden_load(c,batch_hidden_[row]);record_final(c);});compute_.wait(signal);
            invalidate_buffer(runtime_,token_);result[row]=*static_cast<uint32_t*>(token_.mapped);
        }return result;
    }
    void accept_verified(uint32_t row){if(row>=2)throw std::runtime_error("invalid verifier row");uint64_t signal=compute_.submit([&](VkCommandBuffer c){record_hidden_load(c,batch_hidden_[row]);});compute_.wait(signal);}
    DescriptorRange hidden_range()const{return whole(hidden_);}TensorDevice embedding_tensor()const{return weights_.tensor("embed");}
    void reset_decode_metrics(){host_cache_.reset_metrics();device_cache_.reset_metrics();transfer_bytes_=0;prediction_matches_=prediction_total_=0;pre_seconds_=0;acquire_seconds_=0;post_seconds_=0;}
private:
    Buffer device(uint64_t bytes){Buffer b=create_device_buffer(runtime_,bytes);activation_device_bytes_+=b.allocation_size;return b;}
    void allocate_buffers(){
        token_=create_buffer(runtime_,4);routing_=create_buffer(runtime_,64);hidden_=device(kDim*4ull);for(auto& b:batch_hidden_)b=device(kDim*4ull);for(auto& b:batch_shared_)b=device(kDim*4ull);normalized_=device(kDim*4ull);
        q_=device(12288ull*4);k_=device(1024ull*4);v_=device(1024ull*4);gates_=device(96ull*4);context_=device(12288ull*4);
        quant_=device(3200ull*4);ffn_gate_=device(11264ull*4);ffn_up_=device(11264ull*4);ffn_=device(11264ull*4);
        ffn_quant_=device(3000ull*4);shared_out_=device(kDim*4ull);router_logits_=device(kExperts*4ull);
        expert_intermediate_=device(uint64_t(kTopK)*kMoeDim*4);expert_outputs_=device(uint64_t(kTopK)*kDim*4);
        logits_=device(uint64_t(kVocabulary)*4);argmax_=device(512ull*4);
        kv_cache_=device(uint64_t(kLayers)*2*kMaximumContext*kKvHeads*kHeadDim*4);
        rope_full_=device(uint64_t(kMaximumContext)*128*4);rope_slide_=device(uint64_t(kMaximumContext)*128*4);
    }
    void make_rope(){
        auto table=[&](bool full){std::vector<float> v(uint64_t(kMaximumContext)*128);
              const double rotary_dimension=full?64.0:128.0;
              for(uint32_t pos=0;pos<kMaximumContext;++pos)for(uint32_t i=0;i<64;++i){double inv=std::pow(full?5000000.0:10000.0,-2.0*double(i)/rotary_dimension);
                if(full){double wave=2.0*3.141592653589793/inv;if(wave>131072.0)inv/=2.0;else if(wave>=4096.0){double smooth=(131072.0/wave-1.0)/(32.0-1.0);inv=(1.0-smooth)*inv/2.0+smooth*inv;}}
                v[uint64_t(pos)*128+i]=float(std::cos(double(pos)*inv));v[uint64_t(pos)*128+64+i]=float(std::sin(double(pos)*inv));}return v;};
        Buffer stage=dsv4::create_host_buffer_uninitialized(runtime_,uint64_t(kMaximumContext)*128*4);dsv4::FiniteQueue q(runtime_,runtime_.queue);
        for(uint32_t which=0;which<2;++which){auto values=table(which==0);std::memcpy(stage.mapped,values.data(),values.size()*4);dsv4::flush_buffer_range(runtime_,stage,0,values.size()*4);Buffer& dst=which==0?rope_full_:rope_slide_;
            auto signal=q.submit([&](VkCommandBuffer c){VkBufferCopy copy{0,0,values.size()*4};vkfn::CmdCopyBuffer(c,stage.handle,dst.handle,1,&copy);dsv4::transfer_barrier(c,dst);});q.wait(signal);}destroy_buffer(runtime_,stage);
    }
    struct LayerSets{VkDescriptorSet in_norm{},q{},k{},v{},g{},qk{},store_v{},attention{},head_gate{},context_quant{},o{},post_norm{},hidden_quant{},router_gemv{},router{};
        VkDescriptorSet gate{},up{},swiglu{},ffn_quant{},down{};std::vector<VkDescriptorSet> expert_gate,expert_down;};
    VkDescriptorSet q4set(DescriptorRange activation,const TensorDevice& w,DescriptorRange output,DescriptorRange residual={}){
        return kernels_.set({activation,w.data,w.auxiliary,output,residual.buffer?residual:kernels_.dummy()});}
    void build_sets(const SharedIndex&){
        auto embed=weights_.tensor("embed"),fn=weights_.tensor("final_norm"),lm=weights_.tensor("lm_head");
        embed_set_=kernels_.set({embed.data,embed.auxiliary,whole(token_),whole(hidden_)});
        final_norm_set_=kernels_.set({whole(hidden_),fn.data,whole(normalized_)});final_quant_set_=kernels_.set({whole(normalized_),whole(quant_)});
        lm_set_=kernels_.set({whole(quant_),lm.data,lm.auxiliary,whole(logits_)});argmax_set_=kernels_.set({whole(logits_),whole(token_),whole(argmax_)});
        layers_.resize(kLayers);
        for(uint32_t l=0;l<kLayers;++l){auto& s=layers_[l];std::string p="layers."+std::to_string(l)+".";
            auto in=weights_.tensor(p+"input_norm"),post=weights_.tensor(p+"post_norm"),qn=weights_.tensor(p+"q_norm"),kn=weights_.tensor(p+"k_norm");
            auto qw=weights_.tensor(p+"q_proj"),kw=weights_.tensor(p+"k_proj"),vw=weights_.tensor(p+"v_proj"),gw=weights_.tensor(p+"g_proj"),ow=weights_.tensor(p+"o_proj");
            s.in_norm=kernels_.set({whole(hidden_),in.data,whole(normalized_)});s.hidden_quant=kernels_.set({whole(normalized_),whole(quant_)});
            s.q=q4set(whole(quant_),qw,whole(q_));s.k=q4set(whole(quant_),kw,whole(k_));s.v=q4set(whole(quant_),vw,whole(v_));s.g=q4set(whole(quant_),gw,whole(gates_));
            s.qk=kernels_.set({whole(q_),whole(k_),qn.data,kn.data,whole(kv_cache_),l%4==0?whole(rope_full_):whole(rope_slide_)});
            s.store_v=kernels_.set({whole(v_),whole(kv_cache_)});s.attention=kernels_.set({whole(q_),whole(kv_cache_),whole(context_)});s.head_gate=kernels_.set({whole(context_),whole(gates_)});
            s.context_quant=kernels_.set({whole(context_),whole(quant_)});s.o=q4set(whole(quant_),ow,whole(hidden_),whole(hidden_));s.post_norm=kernels_.set({whole(hidden_),post.data,whole(normalized_)});
            if(l<kMoeFirst){auto gate=weights_.tensor(p+"dense_gate_proj"),up=weights_.tensor(p+"dense_up_proj"),down=weights_.tensor(p+"dense_down_proj");
                s.gate=q4set(whole(quant_),gate,whole(ffn_gate_));s.up=q4set(whole(quant_),up,whole(ffn_up_));s.swiglu=kernels_.set({whole(ffn_gate_),whole(ffn_up_),kernels_.dummy(),whole(ffn_)});s.ffn_quant=kernels_.set({whole(ffn_),whole(ffn_quant_)});s.down=q4set(whole(ffn_quant_),down,whole(hidden_),whole(hidden_));
            }else{auto router=weights_.tensor(p+"router"),bias=weights_.tensor(p+"router_bias"),gate=weights_.tensor(p+"shared_gate_proj"),up=weights_.tensor(p+"shared_up_proj"),down=weights_.tensor(p+"shared_down_proj");
                s.router_gemv=kernels_.set({whole(quant_),router.data,router.auxiliary,whole(router_logits_)});s.router=kernels_.set({whole(router_logits_),bias.data,whole(routing_)});
                s.gate=q4set(whole(quant_),gate,whole(ffn_gate_));s.up=q4set(whole(quant_),up,whole(ffn_up_));s.swiglu=kernels_.set({whole(ffn_gate_),whole(ffn_up_),kernels_.dummy(),whole(ffn_)});s.ffn_quant=kernels_.set({whole(ffn_),whole(ffn_quant_)});s.down=q4set(whole(ffn_quant_),down,whole(shared_out_));
                s.expert_gate.resize(device_cache_.slots());s.expert_down.resize(device_cache_.slots());
                for(uint32_t slot=0;slot<device_cache_.slots();++slot){auto record=device_cache_.record(l,slot);s.expert_gate[slot]=kernels_.set({whole(quant_),record,whole(routing_),whole(expert_intermediate_)});s.expert_down[slot]=kernels_.set({whole(ffn_quant_),record,whole(routing_),whole(expert_outputs_)});}
            }
        }
        swiglu_reduce_set_=kernels_.set({whole(expert_outputs_),whole(shared_out_),whole(hidden_)});
        expert_quant_set_=kernels_.set({whole(expert_intermediate_),whole(ffn_quant_)});
    }
    uint32_t run_token(uint32_t token,uint32_t position){if(position>=kMaximumContext)throw std::runtime_error("Step context cap reached");
        *static_cast<uint32_t*>(token_.mapped)=token;flush_buffer(runtime_,token_);
        auto signal=compute_.submit([&](VkCommandBuffer c){Push p{kVocabulary,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().embedding,embed_set_,&p,(kDim+63)/64);});compute_.wait(signal);
        if(position==0&&std::getenv("STEP_DEBUG_LAYERS"))debug_hidden("embed",UINT32_MAX);
        for(uint32_t l=0;l<kLayers;++l){auto before=std::chrono::steady_clock::now();signal=compute_.submit([&](VkCommandBuffer c){record_pre(c,l,position);});compute_.wait(signal);pre_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
            if(l>=kMoeFirst){invalidate_buffer(runtime_,routing_);std::array<uint32_t,kTopK> experts{};auto* r=static_cast<uint32_t*>(routing_.mapped);for(uint32_t i=0;i<kTopK;++i)experts[i]=r[i];
                auto& previous=previous_routes_[l-kMoeFirst];if(previous_valid_[l-kMoeFirst]){prediction_total_+=kTopK;for(uint32_t expert:experts)for(uint32_t old:previous)if(expert==old){++prediction_matches_;break;}}previous=experts;previous_valid_[l-kMoeFirst]=true;
                auto post_begin=std::chrono::steady_clock::now();compute_.submit([&](VkCommandBuffer c){record_shared(c,l);});
                before=std::chrono::steady_clock::now();auto selection=device_cache_.resolve(l,experts);selected_slots_=selection.slots;std::vector<uint32_t> copied;
                std::array<void*,kTopK> direct{};for(uint32_t rank=0;rank<kTopK;++rank)direct[rank]=staging_[rank].mapped;
                auto sources=host_cache_.resolve_batch(l,experts,selection.misses,direct);
                for(uint32_t rank=0;rank<kTopK;++rank)if(selection.misses[rank]){if(!sources.direct[rank])std::memcpy(staging_[rank].mapped,sources.pointers[rank],kExpertRecordBytes);dsv4::flush_buffer_range(runtime_,staging_[rank],0,kExpertRecordBytes);copied.push_back(rank);}
                uint64_t ready=0;if(!copied.empty()){ready=transfer_.submit([&](VkCommandBuffer c){for(uint32_t rank:copied){VkBufferCopy copy{0,uint64_t(selected_slots_[rank])*kExpertRecordBytes,kExpertRecordBytes};vkfn::CmdCopyBuffer(c,staging_[rank].handle,device_cache_.arena(l).handle,1,&copy);transfer_bytes_+=kExpertRecordBytes;}dsv4::transfer_barrier(c,device_cache_.arena(l));});}
                acquire_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();before=std::chrono::steady_clock::now();
                signal=compute_.submit([&](VkCommandBuffer c){record_experts(c,l);},ready?transfer_.semaphore():VK_NULL_HANDLE,ready,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                bool fill=false;for(uint32_t rank:copied)fill=fill||sources.direct[rank];if(fill){transfer_.wait(ready);for(uint32_t rank:copied)if(sources.direct[rank])std::memcpy(const_cast<uint8_t*>(sources.pointers[rank]),staging_[rank].mapped,kExpertRecordBytes);}
                compute_.wait(signal);post_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-post_begin).count();
            }else{auto before2=std::chrono::steady_clock::now();signal=compute_.submit([&](VkCommandBuffer c){record_post(c,l);});compute_.wait(signal);post_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before2).count();}
            if(position==0&&std::getenv("STEP_DEBUG_LAYERS"))debug_hidden("layer",l);
        }
        signal=compute_.submit([&](VkCommandBuffer c){Push rms{1,kDim,float_bits(1e-5f),0};kernels_.dispatch(c,kernels_.p().rms,final_norm_set_,&rms,1);compute_barrier(c);Push q{kDim,128,kDim/4,kDim/4};kernels_.dispatch(c,kernels_.p().quant,final_quant_set_,&q,kDim/128);compute_barrier(c);Push gem{kVocabulary,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().q8,lm_set_,&gem,(kVocabulary+7)/8);compute_barrier(c);Push a{kVocabulary,256,0,0};kernels_.dispatch(c,kernels_.p().argmax,argmax_set_,&a,256);compute_barrier(c);a={kVocabulary,256,1,0};kernels_.dispatch(c,kernels_.p().argmax,argmax_set_,&a,1);});compute_.wait(signal);
        if(std::getenv("STEP_DEBUG_LOGITS")) debug_logits(position);
        invalidate_buffer(runtime_,token_);return *static_cast<uint32_t*>(token_.mapped);
    }
    void record_device_save(VkCommandBuffer c,const Buffer& source,const Buffer& destination,VkDeviceSize bytes){
        VkBufferMemoryBarrier before{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};before.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;before.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;before.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;before.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;before.buffer=source.handle;before.offset=0;before.size=source.size;
        vkfn::CmdPipelineBarrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&before,0,nullptr);
        VkBufferCopy copy{0,0,bytes};vkfn::CmdCopyBuffer(c,source.handle,destination.handle,1,&copy);
        VkBufferMemoryBarrier after{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};after.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;after.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_SHADER_READ_BIT;after.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;after.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;after.buffer=destination.handle;after.offset=0;after.size=destination.size;
        vkfn::CmdPipelineBarrier(c,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&after,0,nullptr);
    }
    void record_device_load(VkCommandBuffer c,const Buffer& source,const Buffer& destination,VkDeviceSize bytes){VkBufferCopy copy{0,0,bytes};vkfn::CmdCopyBuffer(c,source.handle,destination.handle,1,&copy);dsv4::transfer_barrier(c,destination);}
    void record_hidden_save(VkCommandBuffer c,const Buffer& destination){record_device_save(c,hidden_,destination,uint64_t(kDim)*4);}
    void record_hidden_load(VkCommandBuffer c,const Buffer& source){record_device_load(c,source,hidden_,uint64_t(kDim)*4);}
    void embed_token(uint32_t token,Buffer* save){*static_cast<uint32_t*>(token_.mapped)=token;flush_buffer(runtime_,token_);uint64_t signal=compute_.submit([&](VkCommandBuffer c){Push p{kVocabulary,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().embedding,embed_set_,&p,(kDim+63)/64);if(save)record_hidden_save(c,*save);});compute_.wait(signal);}
    void record_final(VkCommandBuffer c){Push rms{1,kDim,float_bits(1e-5f),0};kernels_.dispatch(c,kernels_.p().rms,final_norm_set_,&rms,1);compute_barrier(c);Push q{kDim,128,kDim/4,kDim/4};kernels_.dispatch(c,kernels_.p().quant,final_quant_set_,&q,kDim/128);compute_barrier(c);Push gem{kVocabulary,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().q8,lm_set_,&gem,(kVocabulary+7)/8);compute_barrier(c);Push a{kVocabulary,256,0,0};kernels_.dispatch(c,kernels_.p().argmax,argmax_set_,&a,256);compute_barrier(c);a={kVocabulary,256,1,0};kernels_.dispatch(c,kernels_.p().argmax,argmax_set_,&a,1);}
    void record_moe_input_quant(VkCommandBuffer c,uint32_t l){auto& s=layers_[l];Push p{1,kDim,float_bits(1e-5f),0};kernels_.dispatch(c,kernels_.p().rms,s.post_norm,&p,1);compute_barrier(c);p={kDim,128,kDim/4,kDim/4};kernels_.dispatch(c,kernels_.p().quant,s.hidden_quant,&p,kDim/128);compute_barrier(c);}
    void run_layer(uint32_t l,uint32_t position,const Buffer* load,Buffer* save){
        auto before=std::chrono::steady_clock::now();uint64_t signal=compute_.submit([&](VkCommandBuffer c){if(load)record_hidden_load(c,*load);record_pre(c,l,position);});compute_.wait(signal);pre_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
        if(l>=kMoeFirst){invalidate_buffer(runtime_,routing_);std::array<uint32_t,kTopK> experts{};auto* r=static_cast<uint32_t*>(routing_.mapped);for(uint32_t i=0;i<kTopK;++i)experts[i]=r[i];
            auto& previous=previous_routes_[l-kMoeFirst];if(previous_valid_[l-kMoeFirst]){prediction_total_+=kTopK;for(uint32_t expert:experts)for(uint32_t old:previous)if(expert==old){++prediction_matches_;break;}}previous=experts;previous_valid_[l-kMoeFirst]=true;
            auto post_begin=std::chrono::steady_clock::now();compute_.submit([&](VkCommandBuffer c){record_shared(c,l);});
            before=std::chrono::steady_clock::now();auto selection=device_cache_.resolve(l,experts);selected_slots_=selection.slots;std::vector<uint32_t> copied;std::array<void*,kTopK> direct{};for(uint32_t rank=0;rank<kTopK;++rank)direct[rank]=staging_[rank].mapped;
            auto sources=host_cache_.resolve_batch(l,experts,selection.misses,direct);for(uint32_t rank=0;rank<kTopK;++rank)if(selection.misses[rank]){if(!sources.direct[rank])std::memcpy(staging_[rank].mapped,sources.pointers[rank],kExpertRecordBytes);dsv4::flush_buffer_range(runtime_,staging_[rank],0,kExpertRecordBytes);copied.push_back(rank);}
            uint64_t ready=0;if(!copied.empty()){ready=transfer_.submit([&](VkCommandBuffer c){for(uint32_t rank:copied){VkBufferCopy copy{0,uint64_t(selected_slots_[rank])*kExpertRecordBytes,kExpertRecordBytes};vkfn::CmdCopyBuffer(c,staging_[rank].handle,device_cache_.arena(l).handle,1,&copy);transfer_bytes_+=kExpertRecordBytes;}dsv4::transfer_barrier(c,device_cache_.arena(l));});}
            acquire_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();before=std::chrono::steady_clock::now();signal=compute_.submit([&](VkCommandBuffer c){record_experts(c,l);if(save)record_hidden_save(c,*save);},ready?transfer_.semaphore():VK_NULL_HANDLE,ready,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            bool fill=false;for(uint32_t rank:copied)fill=fill||sources.direct[rank];if(fill){transfer_.wait(ready);for(uint32_t rank:copied)if(sources.direct[rank])std::memcpy(const_cast<uint8_t*>(sources.pointers[rank]),staging_[rank].mapped,kExpertRecordBytes);}compute_.wait(signal);post_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-post_begin).count();
        }else{auto before2=std::chrono::steady_clock::now();signal=compute_.submit([&](VkCommandBuffer c){record_post(c,l);if(save)record_hidden_save(c,*save);});compute_.wait(signal);post_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before2).count();}
    }
    void run_layer2(uint32_t l,uint32_t position){
        if(l<kMoeFirst){for(uint32_t row=0;row<2;++row)run_layer(l,position+row,&batch_hidden_[row],&batch_hidden_[row]);return;}
        std::array<std::array<uint32_t,kTopK>,2> experts{};std::array<std::array<uint32_t,16>,2> routes{};
        for(uint32_t row=0;row<2;++row){auto before=std::chrono::steady_clock::now();uint64_t signal=compute_.submit([&](VkCommandBuffer c){record_hidden_load(c,batch_hidden_[row]);record_pre(c,l,position+row);record_hidden_save(c,batch_hidden_[row]);});compute_.wait(signal);pre_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
            invalidate_buffer(runtime_,routing_);std::memcpy(routes[row].data(),routing_.mapped,64);for(uint32_t r=0;r<kTopK;++r)experts[row][r]=routes[row][r];auto& previous=previous_routes_[l-kMoeFirst];if(previous_valid_[l-kMoeFirst]){prediction_total_+=kTopK;for(uint32_t expert:experts[row])for(uint32_t old:previous)if(expert==old){++prediction_matches_;break;}}previous=experts[row];previous_valid_[l-kMoeFirst]=true;
            auto shared_begin=std::chrono::steady_clock::now();signal=compute_.submit([&](VkCommandBuffer c){record_shared(c,l);record_device_save(c,shared_out_,batch_shared_[row],uint64_t(kDim)*4);});compute_.wait(signal);post_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-shared_begin).count();
        }
        auto selection0=device_cache_.resolve(l,experts[0]);std::array<void*,kTopK> direct0{};for(uint32_t r=0;r<kTopK;++r)direct0[r]=staging_[r].mapped;auto acquire_begin=std::chrono::steady_clock::now();auto sources0=host_cache_.resolve_batch(l,experts[0],selection0.misses,direct0);std::vector<uint32_t> copied0;
        for(uint32_t r=0;r<kTopK;++r)if(selection0.misses[r]){if(!sources0.direct[r])std::memcpy(staging_[r].mapped,sources0.pointers[r],kExpertRecordBytes);dsv4::flush_buffer_range(runtime_,staging_[r],0,kExpertRecordBytes);copied0.push_back(r);}uint64_t ready0=0;if(!copied0.empty())ready0=transfer_.submit([&](VkCommandBuffer c){for(uint32_t r:copied0){VkBufferCopy copy{0,uint64_t(selection0.slots[r])*kExpertRecordBytes,kExpertRecordBytes};vkfn::CmdCopyBuffer(c,staging_[r].handle,device_cache_.arena(l).handle,1,&copy);transfer_bytes_+=kExpertRecordBytes;}dsv4::transfer_barrier(c,device_cache_.arena(l));});acquire_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-acquire_begin).count();
        std::memcpy(routing_.mapped,routes[0].data(),64);flush_buffer(runtime_,routing_);selected_slots_=selection0.slots;auto expert0_begin=std::chrono::steady_clock::now();uint64_t done0=compute_.submit([&](VkCommandBuffer c){record_hidden_load(c,batch_hidden_[0]);record_moe_input_quant(c,l);record_device_load(c,batch_shared_[0],shared_out_,uint64_t(kDim)*4);record_experts(c,l);record_hidden_save(c,batch_hidden_[0]);},ready0?transfer_.semaphore():VK_NULL_HANDLE,ready0,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        if(ready0)transfer_.wait(ready0);for(uint32_t r:copied0)if(sources0.direct[r])std::memcpy(const_cast<uint8_t*>(sources0.pointers[r]),staging_[r].mapped,kExpertRecordBytes);
        auto needed1=device_cache_.missing(l,experts[1]);std::array<void*,kTopK> direct1{};for(uint32_t r=0;r<kTopK;++r)direct1[r]=staging_[kTopK+r].mapped;auto future1=std::async(std::launch::async,[this,l,e=experts[1],needed1,direct1](){return host_cache_.resolve_batch(l,e,needed1,direct1);});compute_.wait(done0);post_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-expert0_begin).count();
        acquire_begin=std::chrono::steady_clock::now();auto sources1=future1.get();auto selection1=device_cache_.resolve(l,experts[1]);for(uint32_t r=0;r<kTopK;++r)if(selection1.misses[r]!=needed1[r])throw std::runtime_error("Step batch cache prediction changed unexpectedly");std::vector<uint32_t> copied1;
        for(uint32_t r=0;r<kTopK;++r)if(selection1.misses[r]){Buffer& stage=staging_[kTopK+r];if(!sources1.direct[r])std::memcpy(stage.mapped,sources1.pointers[r],kExpertRecordBytes);dsv4::flush_buffer_range(runtime_,stage,0,kExpertRecordBytes);copied1.push_back(r);}uint64_t ready1=0;if(!copied1.empty())ready1=transfer_.submit([&](VkCommandBuffer c){for(uint32_t r:copied1){VkBufferCopy copy{0,uint64_t(selection1.slots[r])*kExpertRecordBytes,kExpertRecordBytes};vkfn::CmdCopyBuffer(c,staging_[kTopK+r].handle,device_cache_.arena(l).handle,1,&copy);transfer_bytes_+=kExpertRecordBytes;}dsv4::transfer_barrier(c,device_cache_.arena(l));});acquire_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-acquire_begin).count();
        std::memcpy(routing_.mapped,routes[1].data(),64);flush_buffer(runtime_,routing_);selected_slots_=selection1.slots;auto expert1_begin=std::chrono::steady_clock::now();uint64_t done1=compute_.submit([&](VkCommandBuffer c){record_hidden_load(c,batch_hidden_[1]);record_moe_input_quant(c,l);record_device_load(c,batch_shared_[1],shared_out_,uint64_t(kDim)*4);record_experts(c,l);record_hidden_save(c,batch_hidden_[1]);},ready1?transfer_.semaphore():VK_NULL_HANDLE,ready1,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        if(ready1)transfer_.wait(ready1);for(uint32_t r:copied1)if(sources1.direct[r])std::memcpy(const_cast<uint8_t*>(sources1.pointers[r]),staging_[kTopK+r].mapped,kExpertRecordBytes);compute_.wait(done1);post_seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-expert1_begin).count();
    }
    void debug_logits(uint32_t position){
        constexpr VkDeviceSize logits_bytes=uint64_t(kVocabulary)*sizeof(float);
        constexpr VkDeviceSize hidden_bytes=uint64_t(kDim)*sizeof(float);
        uint64_t signal=compute_.submit([&](VkCommandBuffer c){
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
            vkfn::CmdPipelineBarrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
            VkBufferCopy copies[2]={{0,0,logits_bytes},{0,logits_bytes,hidden_bytes}};
            vkfn::CmdCopyBuffer(c,logits_.handle,staging_[0].handle,1,&copies[0]);
            vkfn::CmdCopyBuffer(c,hidden_.handle,staging_[0].handle,1,&copies[1]);
        });compute_.wait(signal);invalidate_buffer(runtime_,staging_[0]);
        const float* logits=static_cast<const float*>(staging_[0].mapped);const float* hidden=logits+kVocabulary;
        std::array<uint32_t,5> top{};top.fill(UINT32_MAX);uint64_t bad=0;float hmin=3.402823466e+38f,hmax=-3.402823466e+38f;double hs=0;
        for(uint32_t i=0;i<kVocabulary;++i){if(!std::isfinite(logits[i])){++bad;continue;}for(uint32_t j=0;j<top.size();++j)if(top[j]==UINT32_MAX||logits[i]>logits[top[j]]){for(uint32_t q=top.size()-1;q>j;--q)top[q]=top[q-1];top[j]=i;break;}}
        for(uint32_t i=0;i<kDim;++i){if(!std::isfinite(hidden[i])){++bad;continue;}hmin=std::min(hmin,hidden[i]);hmax=std::max(hmax,hidden[i]);hs+=double(hidden[i])*hidden[i];}
        std::cerr<<"Step debug position "<<position<<" hidden[min,max,rms]="<<hmin<<','<<hmax<<','<<std::sqrt(hs/kDim)<<" bad="<<bad<<" top:";
        for(uint32_t i:top)std::cerr<<' '<<i<<'='<<(i==UINT32_MAX?0.0f:logits[i]);std::cerr<<'\n';
    }
    void debug_hidden(const char* label,uint32_t layer){
        constexpr VkDeviceSize bytes=uint64_t(kDim)*sizeof(float);uint64_t signal=compute_.submit([&](VkCommandBuffer c){
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
            vkfn::CmdPipelineBarrier(c,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
            VkBufferCopy copy{0,0,bytes};vkfn::CmdCopyBuffer(c,hidden_.handle,staging_[0].handle,1,&copy);
        });compute_.wait(signal);invalidate_buffer(runtime_,staging_[0]);const float* v=static_cast<const float*>(staging_[0].mapped);
        float lo=3.402823466e+38f,hi=-3.402823466e+38f;uint32_t ilo=0,ihi=0;double ss=0;uint32_t bad=0;for(uint32_t i=0;i<kDim;++i){if(!std::isfinite(v[i])){++bad;continue;}if(v[i]<lo){lo=v[i];ilo=i;}if(v[i]>hi){hi=v[i];ihi=i;}ss+=double(v[i])*v[i];}
        std::cerr<<"Step debug "<<label;if(layer!=UINT32_MAX)std::cerr<<' '<<layer;std::cerr<<" min="<<lo<<'@'<<ilo<<" max="<<hi<<'@'<<ihi<<" rms="<<std::sqrt(ss/kDim)<<" bad="<<bad<<'\n';
    }
    void record_pre(VkCommandBuffer c,uint32_t l,uint32_t position){auto& s=layers_[l];uint32_t heads=l%4==0?64:96,qsize=heads*kHeadDim;
        Push p{1,kDim,float_bits(1e-5f),0};kernels_.dispatch(c,kernels_.p().rms,s.in_norm,&p,1);compute_barrier(c);p={kDim,128,kDim/4,kDim/4};kernels_.dispatch(c,kernels_.p().quant,s.hidden_quant,&p,kDim/128);compute_barrier(c);
        p={qsize,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().q4,s.q,&p,(qsize+7)/8);p={1024,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().q4,s.k,&p,128);kernels_.dispatch(c,kernels_.p().q4,s.v,&p,128);p={heads,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().q4,s.g,&p,(heads+7)/8);compute_barrier(c);
        p={l,position,heads,l%4==0?32u:64u};kernels_.dispatch(c,kernels_.p().qk,s.qk,&p,heads);p={l,position,0,0};kernels_.dispatch(c,kernels_.p().store_v,s.store_v,&p,16);compute_barrier(c);
        p={l,position,heads,l%4==0?0u:512u};kernels_.dispatch(c,kernels_.p().attention,s.attention,&p,heads);compute_barrier(c);p={qsize,heads,0,0};kernels_.dispatch(c,kernels_.p().head_gate,s.head_gate,&p,(qsize+63)/64);compute_barrier(c);
        p={qsize,128,(qsize+3)/4,(qsize+3)/4};kernels_.dispatch(c,kernels_.p().quant,s.context_quant,&p,(qsize+127)/128);compute_barrier(c);p={kDim,qsize,(qsize+3)/4,0};kernels_.dispatch(c,kernels_.p().q4_residual,s.o,&p,kDim/8);compute_barrier(c);
        p={1,kDim,float_bits(1e-5f),0};kernels_.dispatch(c,kernels_.p().rms,s.post_norm,&p,1);compute_barrier(c);p={kDim,128,kDim/4,kDim/4};kernels_.dispatch(c,kernels_.p().quant,s.hidden_quant,&p,kDim/128);compute_barrier(c);
        if(l>=kMoeFirst){p={kExperts,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().q8,s.router_gemv,&p,(kExperts+7)/8);compute_barrier(c);p={kExperts,kTopK,float_bits(3.0f),0};kernels_.dispatch(c,kernels_.p().router,s.router,&p,1);}
    }
    void record_shared(VkCommandBuffer c,uint32_t l){auto& s=layers_[l];uint32_t width=l<kMoeFirst?11264:kMoeDim;Push p{width,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().q4,s.gate,&p,(width+7)/8);kernels_.dispatch(c,kernels_.p().q4,s.up,&p,(width+7)/8);compute_barrier(c);const float shared_limit=l>=43?16.0f:3.402823466e+38f;p={width,float_bits(shared_limit),0,0};kernels_.dispatch(c,kernels_.p().swiglu,s.swiglu,&p,(width+63)/64);compute_barrier(c);p={width,128,(width+3)/4,(width+3)/4};kernels_.dispatch(c,kernels_.p().quant,s.ffn_quant,&p,(width+127)/128);compute_barrier(c);
        if(l<kMoeFirst){p={kDim,width,(width+3)/4,0};kernels_.dispatch(c,kernels_.p().q4_residual,s.down,&p,kDim/8);return;}p={kDim,kMoeDim,(kMoeDim+3)/4,0};kernels_.dispatch(c,kernels_.p().q4,s.down,&p,kDim/8);
    }
    void record_experts(VkCommandBuffer c,uint32_t l){auto& s=layers_[l];Push p{};
        const float expert_limit=l>=43?7.0f:3.402823466e+38f;for(uint32_t r=0;r<kTopK;++r){p={r,kDim/4,float_bits(expert_limit),0};kernels_.dispatch(c,kernels_.p().expert_gate,s.expert_gate[selected_slots_[r]],&p,kMoeDim/8);}compute_barrier(c);
        p={kTopK*kMoeDim,128,kTopK*kMoeDim/4,kTopK*kMoeDim/4};kernels_.dispatch(c,kernels_.p().quant,expert_quant_set_,&p,kTopK*kMoeDim/128);compute_barrier(c);
        for(uint32_t r=0;r<kTopK;++r){p={r,kTopK*kMoeDim/4,0,0};kernels_.dispatch(c,kernels_.p().expert_down,s.expert_down[selected_slots_[r]],&p,kDim/8);}compute_barrier(c);p={kDim,kTopK,0,0};kernels_.dispatch(c,kernels_.p().reduce,swiglu_reduce_set_,&p,kDim/64);
    }
    void record_post(VkCommandBuffer c,uint32_t l){record_shared(c,l);if(l>=kMoeFirst){compute_barrier(c);record_experts(c,l);}}
    void destroy_all(){for(auto& b:batch_shared_)destroy_buffer(runtime_,b);for(auto& b:batch_hidden_)destroy_buffer(runtime_,b);for(Buffer* b:{&rope_slide_,&rope_full_,&kv_cache_,&argmax_,&logits_,&expert_outputs_,&expert_intermediate_,&router_logits_,&shared_out_,&ffn_quant_,&ffn_,&ffn_up_,&ffn_gate_,&quant_,&context_,&gates_,&v_,&k_,&q_,&normalized_,&hidden_,&routing_,&token_})destroy_buffer(runtime_,*b);}
    const Runtime& runtime_;DeviceWeights weights_;ExpertFile expert_file_;HostExpertCache host_cache_;DeviceExpertCache device_cache_;Kernels kernels_;dsv4::FiniteQueue compute_,transfer_;
    Buffer token_{},routing_{},hidden_{};std::array<Buffer,2> batch_hidden_{},batch_shared_{};Buffer normalized_{},q_{},k_{},v_{},gates_{},context_{},quant_{},ffn_gate_{},ffn_up_{},ffn_{},ffn_quant_{},shared_out_{},router_logits_{},expert_intermediate_{},expert_outputs_{},logits_{},argmax_{},kv_cache_{},rope_full_{},rope_slide_{};
    std::vector<Buffer> staging_;std::vector<LayerSets> layers_;std::array<uint32_t,kTopK> selected_slots_{};
    std::array<std::array<uint32_t,kTopK>,kMoeLayers> previous_routes_{};std::array<bool,kMoeLayers> previous_valid_{};
    VkDescriptorSet embed_set_{},final_norm_set_{},final_quant_set_{},lm_set_{},argmax_set_{},swiglu_reduce_set_{},expert_quant_set_{};
    uint64_t activation_device_bytes_=0,transfer_bytes_=0,prediction_matches_=0,prediction_total_=0;double decode_seconds_=0,pre_seconds_=0,acquire_seconds_=0,post_seconds_=0;
};

class MtpOneEngine{
public:
    MtpOneEngine(const Runtime& runtime,const SharedIndex& index,const TensorDevice& embedding,const DescriptorRange& main_hidden,const std::filesystem::path& shaders)
      :runtime_(runtime),weights_(runtime,index,1),kernels_(runtime,shaders),compute_(runtime,runtime.queue),main_hidden_(main_hidden){allocate();rope();sets(embedding);}
    ~MtpOneEngine(){for(Buffer* b:{&rope_,&kv_,&argmax_,&logits_,&ffn_quant_,&ffn_,&up_,&gate_,&quant_,&context_,&gates_,&v_,&k_,&q_,&normalized_,&hidden_,&concat_,&embedding_,&token_})destroy_buffer(runtime_,*b);}
    uint32_t process(uint32_t token,uint32_t position){if(position>=kMaximumContext)throw std::runtime_error("Step MTP context cap reached");auto started=std::chrono::steady_clock::now();*static_cast<uint32_t*>(token_.mapped)=token;flush_buffer(runtime_,token_);
        uint64_t signal=compute_.submit([&](VkCommandBuffer c){Push p{kVocabulary,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().embedding,embedding_set_,&p,kDim/64);compute_barrier(c);
            p={kDim,float_bits(1e-5f),0,0};kernels_.dispatch(c,kernels_.p().mtp_fuse,fuse_set_,&p,1);compute_barrier(c);p={2*kDim,128,2*kDim/4,2*kDim/4};kernels_.dispatch(c,kernels_.p().quant,concat_quant_,&p,(2*kDim)/128);compute_barrier(c);p={kDim,2*kDim,2*kDim/4,0};kernels_.dispatch(c,kernels_.p().q4,eh_,&p,kDim/8);compute_barrier(c);
            p={1,kDim,float_bits(1e-5f),0};kernels_.dispatch(c,kernels_.p().rms,input_norm_,&p,1);compute_barrier(c);p={kDim,128,kDim/4,kDim/4};kernels_.dispatch(c,kernels_.p().quant,hidden_quant_,&p,kDim/128);compute_barrier(c);
            p={12288,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().q4,q_set_,&p,12288/8);p={1024,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().q4,k_set_,&p,128);kernels_.dispatch(c,kernels_.p().q4,v_set_,&p,128);p={96,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().q4,g_set_,&p,12);compute_barrier(c);
            p={0,position,96,64};kernels_.dispatch(c,kernels_.p().qk,qk_set_,&p,96);p={0,position,0,0};kernels_.dispatch(c,kernels_.p().store_v,store_v_,&p,16);compute_barrier(c);p={0,position,96,512};kernels_.dispatch(c,kernels_.p().attention,attention_,&p,96);compute_barrier(c);p={12288,96,0,0};kernels_.dispatch(c,kernels_.p().head_gate,head_gate_,&p,12288/64);compute_barrier(c);
            p={12288,128,12288/4,12288/4};kernels_.dispatch(c,kernels_.p().quant,context_quant_,&p,12288/128);compute_barrier(c);p={kDim,12288,12288/4,0};kernels_.dispatch(c,kernels_.p().q4_residual,o_set_,&p,kDim/8);compute_barrier(c);
            p={1,kDim,float_bits(1e-5f),0};kernels_.dispatch(c,kernels_.p().rms,post_norm_,&p,1);compute_barrier(c);p={kDim,128,kDim/4,kDim/4};kernels_.dispatch(c,kernels_.p().quant,hidden_quant_,&p,kDim/128);compute_barrier(c);p={11264,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().q4,gate_set_,&p,11264/8);kernels_.dispatch(c,kernels_.p().q4,up_set_,&p,11264/8);compute_barrier(c);p={11264,float_bits(3.402823466e+38f),0,0};kernels_.dispatch(c,kernels_.p().swiglu,swiglu_,&p,11264/64);compute_barrier(c);p={11264,128,11264/4,11264/4};kernels_.dispatch(c,kernels_.p().quant,ffn_quant_set_,&p,11264/128);compute_barrier(c);p={kDim,11264,11264/4,0};kernels_.dispatch(c,kernels_.p().q4_residual,down_set_,&p,kDim/8);compute_barrier(c);
            p={1,kDim,float_bits(1e-5f),0};kernels_.dispatch(c,kernels_.p().rms,head_norm_,&p,1);compute_barrier(c);p={kDim,128,kDim/4,kDim/4};kernels_.dispatch(c,kernels_.p().quant,head_quant_,&p,kDim/128);compute_barrier(c);p={kVocabulary,kDim,kDim/4,0};kernels_.dispatch(c,kernels_.p().q8,head_,&p,(kVocabulary+7)/8);compute_barrier(c);p={kVocabulary,256,0,0};kernels_.dispatch(c,kernels_.p().argmax,argmax_set_,&p,256);compute_barrier(c);p={kVocabulary,256,1,0};kernels_.dispatch(c,kernels_.p().argmax,argmax_set_,&p,1);
        });compute_.wait(signal);invalidate_buffer(runtime_,token_);seconds_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();++passes_;return *static_cast<uint32_t*>(token_.mapped);}
    double seconds()const{return seconds_;}uint64_t passes()const{return passes_;}uint64_t device_bytes()const{return weights_.device_bytes()+device_bytes_;}
    void reset_metrics(){seconds_=0;passes_=0;}
private:
    Buffer device(uint64_t n){Buffer b=create_device_buffer(runtime_,n);device_bytes_+=b.allocation_size;return b;}
    VkDescriptorSet q4(DescriptorRange a,const TensorDevice& w,DescriptorRange o,DescriptorRange residual={}){return kernels_.set({a,w.data,w.auxiliary,o,residual.buffer?residual:kernels_.dummy()});}
    void allocate(){token_=create_buffer(runtime_,4);embedding_=device(kDim*4ull);concat_=device(2*kDim*4ull);hidden_=device(kDim*4ull);normalized_=device(kDim*4ull);q_=device(12288ull*4);k_=device(1024ull*4);v_=device(1024ull*4);gates_=device(96ull*4);context_=device(12288ull*4);quant_=device(3200ull*4);gate_=device(11264ull*4);up_=device(11264ull*4);ffn_=device(11264ull*4);ffn_quant_=device(3000ull*4);logits_=device(uint64_t(kVocabulary)*4);argmax_=device(512ull*4);kv_=device(uint64_t(2)*kMaximumContext*kKvHeads*kHeadDim*4);rope_=device(uint64_t(kMaximumContext)*128*4);}
    void rope(){std::vector<float> table(uint64_t(kMaximumContext)*128);for(uint32_t pos=0;pos<kMaximumContext;++pos)for(uint32_t i=0;i<64;++i){double inv=std::pow(10000.0,-2.0*double(i)/128.0);table[uint64_t(pos)*128+i]=float(std::cos(pos*inv));table[uint64_t(pos)*128+64+i]=float(std::sin(pos*inv));}Buffer stage=dsv4::create_host_buffer_uninitialized(runtime_,table.size()*4);std::memcpy(stage.mapped,table.data(),table.size()*4);dsv4::flush_buffer_range(runtime_,stage,0,table.size()*4);dsv4::FiniteQueue queue(runtime_,runtime_.queue);auto done=queue.submit([&](VkCommandBuffer c){VkBufferCopy copy{0,0,table.size()*4};vkfn::CmdCopyBuffer(c,stage.handle,rope_.handle,1,&copy);dsv4::transfer_barrier(c,rope_);});queue.wait(done);destroy_buffer(runtime_,stage);}
    void sets(const TensorDevice& embed){auto t=[&](const char* n){return weights_.tensor(std::string("mtp.0.")+n);};embedding_set_=kernels_.set({embed.data,embed.auxiliary,whole(token_),whole(embedding_)});fuse_set_=kernels_.set({whole(embedding_),main_hidden_,t("enorm").data,t("hnorm").data,whole(concat_)});concat_quant_=kernels_.set({whole(concat_),whole(quant_)});eh_=q4(whole(quant_),t("eh_proj"),whole(hidden_));input_norm_=kernels_.set({whole(hidden_),t("input_norm").data,whole(normalized_)});hidden_quant_=kernels_.set({whole(normalized_),whole(quant_)});q_set_=q4(whole(quant_),t("q_proj"),whole(q_));k_set_=q4(whole(quant_),t("k_proj"),whole(k_));v_set_=q4(whole(quant_),t("v_proj"),whole(v_));g_set_=q4(whole(quant_),t("g_proj"),whole(gates_));qk_set_=kernels_.set({whole(q_),whole(k_),t("q_norm").data,t("k_norm").data,whole(kv_),whole(rope_)});store_v_=kernels_.set({whole(v_),whole(kv_)});attention_=kernels_.set({whole(q_),whole(kv_),whole(context_)});head_gate_=kernels_.set({whole(context_),whole(gates_)});context_quant_=kernels_.set({whole(context_),whole(quant_)});o_set_=q4(whole(quant_),t("o_proj"),whole(hidden_),whole(hidden_));post_norm_=kernels_.set({whole(hidden_),t("post_norm").data,whole(normalized_)});gate_set_=q4(whole(quant_),t("gate_proj"),whole(gate_));up_set_=q4(whole(quant_),t("up_proj"),whole(up_));swiglu_=kernels_.set({whole(gate_),whole(up_),kernels_.dummy(),whole(ffn_)});ffn_quant_set_=kernels_.set({whole(ffn_),whole(ffn_quant_)});down_set_=q4(whole(ffn_quant_),t("down_proj"),whole(hidden_),whole(hidden_));head_norm_=kernels_.set({whole(hidden_),t("head_norm").data,whole(normalized_)});head_quant_=kernels_.set({whole(normalized_),whole(quant_)});auto hw=t("head");head_=kernels_.set({whole(quant_),hw.data,hw.auxiliary,whole(logits_)});argmax_set_=kernels_.set({whole(logits_),whole(token_),whole(argmax_)});}
    const Runtime& runtime_;DeviceWeights weights_;Kernels kernels_;dsv4::FiniteQueue compute_;DescriptorRange main_hidden_{};uint64_t device_bytes_=0,passes_=0;double seconds_=0;
    Buffer token_{},embedding_{},concat_{},hidden_{},normalized_{},q_{},k_{},v_{},gates_{},context_{},quant_{},gate_{},up_{},ffn_{},ffn_quant_{},logits_{},argmax_{},kv_{},rope_{};
    VkDescriptorSet embedding_set_{},fuse_set_{},concat_quant_{},eh_{},input_norm_{},hidden_quant_{},q_set_{},k_set_{},v_set_{},g_set_{},qk_set_{},store_v_{},attention_{},head_gate_{},context_quant_{},o_set_{},post_norm_{},gate_set_{},up_set_{},swiglu_{},ffn_quant_set_{},down_set_{},head_norm_{},head_quant_{},head_{},argmax_set_{};
};

static uint64_t ram_budget(){const char* text=std::getenv("STEP_RAM_GIB");double gib=text?std::stod(text):16.0;if(gib<2||gib>56)throw std::runtime_error("STEP_RAM_GIB must be 2..56");return uint64_t(gib*1024.0*1024.0*1024.0);}
static uint32_t device_slots(){const char* text=std::getenv("STEP_DEVICE_SLOTS_PER_LAYER");uint32_t n=text?uint32_t(std::stoul(text)):12;if(n<kTopK||n>24)throw std::runtime_error("STEP_DEVICE_SLOTS_PER_LAYER must be 8..24");return n;}

} // namespace step37

int main(int argc,char** argv){Runtime runtime{};try{
    if(argc<3){std::cerr<<"usage: amd_step37.exe <runtime-dir> <prompt|--inspect|--tokenize> [new-tokens]\n";return 2;}
    std::filesystem::path dir=argv[1];dsv4::ReadOnlyMapping tokfile((dir/"tokenizer.ovb").string());dsv4::Tokenizer tokenizer(tokfile);
    if(std::strcmp(argv[2],"--tokenize")==0){if(argc<4)throw std::runtime_error("tokenize text required");auto ids=tokenizer.chat_prompt(argv[3],true);std::cout<<"tokens:";for(auto x:ids)std::cout<<' '<<x;std::cout<<"\n";return 0;}
    step37::SharedIndex index(dir/"model-q4g64.ovs");step37::ExpertFile inspect_experts(dir/"experts-q4g64.ovx");
    if(std::strcmp(argv[2],"--inspect")==0){std::cout<<"Step-3.7 runtime containers validated\n";return 0;}
    runtime=create_runtime();std::cout<<"Vulkan device: "<<runtime.properties.deviceName<<"\n";
    uint32_t count=argc>=4?uint32_t(std::stoul(argv[3])):8;uint64_t budget=step37::ram_budget();uint32_t slots=step37::device_slots();
    const bool mtp1=std::getenv("STEP_MTP1")!=nullptr;std::vector<uint32_t> result;double decode=0,pre=0,acq=0,post=0,mtp_seconds=0,verify_seconds=0;uint64_t dh=0,dm=0,rh=0,rm=0,disk=0,traffic=0,ram=0,vram=0,pm=0,pt=0,mtp_match=0,mtp_total=0,accepted_main=0;uint32_t hslots=0;
    {step37::StepEngine engine(runtime,index,dir/"experts-q4g64.ovx",std::filesystem::absolute(argv[0]).parent_path(),budget,slots);
     std::cout<<"precision: Q4G64T experts/shared, Q8 global/router\nRAM budget: "<<double(budget)/(1024.0*1024*1024)<<" GiB\nexpert slots device/RAM: "<<slots<<" per layer / "<<engine.host_slots()<<" global\n";
     if(mtp1){step37::SharedIndex mtp_index(dir/"mtp-q4g64.ovs");step37::MtpOneEngine mtp(runtime,mtp_index,engine.embedding_tensor(),engine.hidden_range(),std::filesystem::absolute(argv[0]).parent_path());auto prompt=tokenizer.chat_prompt(argv[2],true);uint32_t position=0,next=0;for(uint32_t i=0;i<prompt.size();++i){next=engine.process(prompt[i],position++);if(i+1<prompt.size())mtp.process(prompt[i+1],position);}engine.reset_decode_metrics();mtp.reset_metrics();auto started=std::chrono::steady_clock::now();if(count&&next!=tokenizer.eos()){result.push_back(next);std::cout<<tokenizer.decode_piece(next)<<std::flush;}while(result.size()<count&&next!=tokenizer.eos()){
        uint32_t draft=mtp.process(next,position);auto verify_started=std::chrono::steady_clock::now();auto exact=engine.verify2({next,draft},position);verify_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-verify_started).count();++mtp_total;const bool matched=draft==exact[0];if(matched)++mtp_match;
        engine.accept_verified(0);next=exact[0];++position;if(next==tokenizer.eos())break;result.push_back(next);++accepted_main;std::cout<<tokenizer.decode_piece(next)<<std::flush;if(result.size()>=count)break;
        if(matched){mtp.process(next,position);engine.accept_verified(1);next=exact[1];++position;if(next==tokenizer.eos())break;result.push_back(next);++accepted_main;std::cout<<tokenizer.decode_piece(next)<<std::flush;}
     }decode=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();mtp_seconds=mtp.seconds();vram=engine.vram_bytes()+mtp.device_bytes();}
     else{result=engine.generate(tokenizer,tokenizer.chat_prompt(argv[2],true),count);decode=engine.decode_seconds();vram=engine.vram_bytes();}
     pre=engine.pre_seconds();acq=engine.acquire_seconds();post=engine.post_seconds();dh=engine.device_hits();dm=engine.device_misses();rh=engine.ram_hits();rm=engine.ram_misses();disk=engine.disk_bytes();traffic=engine.transfer_bytes();ram=engine.ram_bytes();hslots=engine.host_slots();pm=engine.prediction_matches();pt=engine.prediction_total();}
    const uint64_t timed_tokens=result.size()>1?result.size()-1:0;
    std::cout<<"\ntoken ids:";for(auto x:result)std::cout<<' '<<x;std::cout<<"\ndecode throughput: "<<(decode>0?timed_tokens/decode:0)<<" tok/s ("<<timed_tokens<<" timed token transitions)\n";
    std::cout<<"device hits/misses: "<<dh<<'/'<<dm<<"\nRAM hits/misses: "<<rh<<'/'<<rm<<"\nexpert disk/VRAM traffic GiB: "<<double(disk)/(1ull<<30)<<" / "<<double(traffic)/(1ull<<30)<<"\n";
    std::cout<<"previous-token route overlap: "<<pm<<'/'<<pt<<" ("<<(pt?100.0*pm/pt:0.0)<<"%)\n";
    if(mtp1)std::cout<<"MTP-1 draft acceptance: "<<mtp_match<<'/'<<mtp_total<<" ("<<(mtp_total?100.0*mtp_match/mtp_total:0.0)<<"%), accepted/main pass: "<<(mtp_total?double(accepted_main)/mtp_total:0.0)<<", draft / verification seconds: "<<mtp_seconds<<" / "<<verify_seconds<<"\n";
    std::cout<<"attention+router / acquisition / expert+shared s: "<<pre<<" / "<<acq<<" / "<<post<<"\npeak explicit host model RAM GiB: "<<double(ram)/(1ull<<30)<<" (capacity "<<hslots<<" records)\npeak device allocation estimate GiB: "<<double(vram)/(1ull<<30)<<"\n";
    vkfn::DestroyDevice(runtime.device,nullptr);vkfn::DestroyInstance(runtime.instance,nullptr);FreeLibrary(runtime.loader);return 0;
 }catch(const std::exception& e){std::cerr<<"Step-3.7 runtime error: "<<e.what()<<"\n";if(runtime.device)vkfn::DestroyDevice(runtime.device,nullptr);if(runtime.instance)vkfn::DestroyInstance(runtime.instance,nullptr);if(runtime.loader)FreeLibrary(runtime.loader);return 1;}}
