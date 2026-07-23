from vllm import LLM

if __name__ == '__main__':
    llm = LLM(
        model="./opt-125m",
        dtype="float32",
        gpu_memory_utilization=0.3
    )
    outputs = llm.generate("The capital of France is")
    for o in outputs:
        print(o.outputs[0].text)

# export VLLM_CPU_KVCACHE_SPACE=4
# python test_cpu.py