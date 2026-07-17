import json
from collections import defaultdict
from vllm import LLM, SamplingParams

def run_vllm_base_diagnostic(model_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", eval_dataset_path="datasets/dev_frozen.jsonl"):
    """
    Evaluates the un-adapted base model via local vLLM pipeline, 
    breaking down accuracy by task_type to solve the haircut equations.
    """
    print(f"Initializing base model via vLLM pipeline: {model_id}")
    
    # Instantiate native base model in vLLM (mimics production runner)
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        tensor_parallel_size=1,  # Set to match your available GPU configuration
        gpu_memory_utilization=0.90,
        dtype="bfloat16"
    )

    # Force strict deterministic greedy settings to eliminate noise
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128
    )

    prompts = []
    ground_truths = []
    categories = []

    print(f"Reading records from {eval_dataset_path}...")
    with open(eval_dataset_path, "r") as f:
        for line in f:
            record = json.loads(line)
            prompts.append(record["prompt"])
            ground_truths.append(record.get("answer", "").strip())
            categories.append(record.get("task_type", "unknown"))

    print(f"Generating completions for {len(prompts)} validation items...")
    # Batch process everything instantly 
    outputs = llm.generate(prompts, sampling_params)

    category_totals = defaultdict(int)
    category_correct = defaultdict(int)

    for output, gt, cat in zip(outputs, ground_truths, categories):
        prediction = output.outputs[0].text.strip()
        is_correct = (prediction == gt)
        
        category_totals[cat] += 1
        if is_correct:
            category_correct[cat] += 1

    print("\n================ LOCAL BASE EVALUATION BREAKDOWN ================")
    total_samples = 0
    total_correct = 0
    
    for cat in sorted(category_totals.keys()):
        correct = category_correct[cat]
        total = category_totals[cat]
        acc = correct / total if total > 0 else 0.0
        total_samples += total
        total_correct += correct
        print(f"Category: {cat:<25} | Accuracy: {acc:.2%} ({correct}/{total})")
        
    global_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print("----------------------------------------------------------------")
    print(f"Overall Local Base Accuracy: {global_acc:.2%}")
    print("================================================================\n")

if __name__ == "__main__":
    run_vllm_base_diagnostic()
