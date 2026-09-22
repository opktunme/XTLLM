#define OVLLM_QWEN38_RUNTIME_ONLY
#include "../src/qwen38_flash_next.cpp"
#include <random>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    Runtime runtime{};
    try {
        _putenv_s("QWEN38_DWARFSTAR", "");
        _putenv_s("QWEN38_DWARF_ROUTER", "");
        runtime = create_runtime();
        {
            qwen38::Kernels original(runtime, argv[1]);
            _putenv_s("QWEN38_DWARF_ROUTER", "1");
            qwen38::Kernels parallel(runtime, argv[1]);
            Buffer logits=create_buffer(runtime,512*4), a=create_buffer(runtime,36*4), b=create_buffer(runtime,36*4);
            auto sa=original.set({whole(logits),whole(a)});
            auto sb=parallel.set({whole(logits),whole(b)});
            dsv4::FiniteQueue queue(runtime,runtime.queue);
            std::mt19937 rng(381024);
            for (uint32_t trial=0;trial<100;++trial) {
                auto* values=static_cast<float*>(logits.mapped);
                for(uint32_t i=0;i<512;++i) {
                    float value=float(int(rng()%20001)-10000)*0.003f;
                    if(trial==0) value=0;
                    if(trial==1) value=-float(i);
                    if(trial==2) value=float(i);
                    if(trial==3) value=float(i%7); // ties spanning lanes/groups
                    if(trial==4) value=-std::numeric_limits<float>::max();
                    if(trial==5) value=-1;
                    if(trial==6 && i%17==0) value=std::numeric_limits<float>::quiet_NaN();
                    values[i]=value;
                }
                flush_buffer(runtime,logits);
                const auto signal=queue.submit([&](VkCommandBuffer command) {
                    qwen38::Push push{512,10,0,0};
                    original.dispatch(command,original.p().router,sa,&push,1);
                    parallel.dispatch(command,parallel.p().router,sb,&push,1);
                    compute_barrier(command);
                });
                queue.wait(signal);
                invalidate_buffer(runtime,a); invalidate_buffer(runtime,b);
                auto* x=static_cast<uint32_t*>(a.mapped); auto* y=static_cast<uint32_t*>(b.mapped);
                for(uint32_t rank=0;rank<10;++rank)
                    if(x[rank]!=y[rank] || x[16+rank]!=y[16+rank])
                        throw std::runtime_error("Router mismatch trial="+std::to_string(trial)+" rank="+std::to_string(rank));
            }
            destroy_buffer(runtime,b); destroy_buffer(runtime,a); destroy_buffer(runtime,logits);
            std::cout << "PASS: 100 router fixtures; all top-10 IDs and weight bits match original, including ties/sentinels/mixed NaNs.\n";
        }
        vkfn::DestroyDevice(runtime.device,nullptr);
        vkfn::DestroyInstance(runtime.instance,nullptr);
        FreeLibrary(runtime.loader);
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
