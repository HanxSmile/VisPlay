import argparse
import time
import torch
import threading
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from flask import Flask, request, jsonify
import os
import json
from vllm_utils import build_prompt_and_images, process_single


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='5000')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
    parser.add_argument('--gpu_mem_util', type=float, default=0.8,
                        help='The maximum GPU memory utilization fraction for vLLM.')
    parser.add_argument('--max_model_len', type=int, default=20480,
                        help='模型最大长度 (默认: 8192)')
    parser.add_argument('--max_tokens', type=int, default=4096,
                        help='最大生成token数 (默认: 4096)')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='采样温度 (默认: 0.0)')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p采样 (默认: 0.95)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='每个prompt的采样个数n (默认: 1)')
    args = parser.parse_args()
    return args


args = parse_args()

##  vLLM Initialization
processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

# 初始化 vLLM
llm = LLM(
    model=args.model_path,
    tensor_parallel_size=args.tensor_parallel_size,
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    trust_remote_code=True,
    dtype="bfloat16",
    limit_mm_per_prompt={"image": 10},
)

# 采样参数
sampling_params = SamplingParams(
    max_tokens=args.max_tokens,
    temperature=args.temperature,
    top_p=args.top_p,
    repetition_penalty=1.0,
    n=args.num_samples,
)

# ---------------------- GPU Idle Utilization Thread ---------------------- #
# (This section remains unchanged)
stop_event = threading.Event()  # Event to stop the thread globally
pause_event = threading.Event()  # Event to pause the thread during requests


def gpu_idle_worker():
    '''
    This worker occupies the GPU with a continuous matrix multiplication loop when idle,
    preventing potential performance drops from GPU power state changes.
    '''
    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] Paused.')
                running = False
            time.sleep(0.1)  # Sleep briefly while paused
            continue
        else:
            if not running:
                print('[idle_worker] Resumed.')
                running = True
        try:
            # A simple but effective way to keep the GPU busy
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[idle_worker] Caught a RuntimeError: {e}. Sleeping for 1s...')
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')


idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
idle_thread.start()

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)


@app.route('/hello', methods=['GET'])
def hello():
    '''The main processing endpoint: reads a task file, invokes vLLM, consolidates answers, and writes results.'''

    # --- Pause the GPU idle worker to free up resources ---
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received request for task file: {name}')

    # ---------- Load Data ----------
    with open(name, 'r', encoding="utf-8") as f:
        data = json.load(f)
    os.remove(name)

    system_prompts = [item["system"] for item in data]
    questions = [item['question'] for item in data]
    types = [item['types'] for item in data]
    image_list = [item['image'] for item in data]

    valid_chats = [build_prompt_and_images(imgs, sys_p, p, processor) for imgs, sys_p, p in
                   zip(image_list, system_prompts, questions)]

    responses = llm.generate(valid_chats, sampling_params=sampling_params, use_tqdm=True)

    print('[server] Generation completed.')

    results_all = []
    response_idx = 0
    for q, t in zip(questions, types):
        try:
            response = responses[response_idx]
            response_idx += 1
            item = process_single(q, t, response)
            results_all.append(item)

        except Exception as e:
            # Catch any other unexpected exceptions from within process_single.
            print(f'[server] CRITICAL: An unhandled error occurred while processing question: {q}')
            print(f'[server] Error details: {e}')
            results_all.append({
                'question': q,
                'type': t,
                'score': -1,
                'results': [],
                'error': f'unhandled exception in process_single: {str(e)}'
            })
    print('[server] All results have been processed.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    # --- Resume the GPU idle worker ---
    pause_event.clear()
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})


# ------------------------- Main Application Entrypoint --------------------------- #
# (This section remains unchanged)
if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        # Gracefully shut down the background thread on exit
        stop_event.set()
        idle_thread.join()
        print('[main] Application shutdown complete.')
