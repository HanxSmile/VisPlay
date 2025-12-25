import io
import base64
from PIL import Image
from typing import List, Tuple
from qwen_vl_utils import process_vision_info
from task_formatter import TaskFormatter
from collections import Counter

task_formatter_obj = TaskFormatter()


def build_prompt_and_images(
        image_list: List[str],
        system_prompt: str,
        prompt: str,
        processor
) -> Tuple[str, List]:
    """
    构建 vLLM generate 需要的 prompt 和图像

    Args:
        image_list: 图像路径列表
        ground_truth: 包含 findings 和 impression 的字典
        prompt_type: prompt类型 (cot, no_cot, cot_with_bbox)
        processor: Qwen processor 用于 apply_chat_template

    Returns:
        prompt: 格式化的prompt字符串
        image_inputs: 处理后的图像输入列表
    """

    # 构建图像内容 - 包含路径和像素限制参数
    if isinstance(image_list, str):
        image_list = [image_list]
    image_content = []
    for image_path in image_list:
        image_content.append({
            "type": "image",
            "image": base64_to_pil(image_path),
            "max_pixels": 1500 ** 2,
            "min_pixels": 256 * 28 * 28,
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": image_content + [{"type": "text", "text": prompt}]}
    ]

    # 使用 processor 的 chat template 格式化
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 使用 qwen_vl_utils 处理图像（会自动根据 max_pixels/min_pixels 调整）
    image_inputs, video_inputs = process_vision_info(messages)

    return prompt, image_inputs


def base64_to_pil(b64_string):
    # 去掉 "data:image/png;base64," 头（如果有的话）
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


# ---------- Results Post-Processing (Core Refactoring & Optimization Here) ----------
def process_single(question, type_, response):
    '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
    results = [task_formatter_obj(type_, out.text) for out in response.outputs]
    results = [_ for _ in results if _]

    answer_counts = Counter(results)

    if not answer_counts:
        majority_ans, max_count = '', 0
    else:
        majority_ans = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_ans]

    score = max_count / len(results) if results else 0.0

    return {
        'question': question,
        'answer': list(majority_ans) if isinstance(majority_ans, tuple) else [majority_ans],
        'score': score,
        'results': [list(_) if isinstance(_, tuple) else [_] for _ in results]
    }
