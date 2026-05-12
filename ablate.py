import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import argparse

def measure_loss(model, tokenizer, samples, device):
    """
    Measure average loss on a list of samples.
    """
    total_loss = 0
    total_tokens = 0
    
    for sample in samples:
        inputs = tokenizer(sample, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            num_tokens = inputs["input_ids"].numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss

def ablate_tensors(model_id, samples_path, output_path, limit_samples=None):
    print(f"[N] loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    device = model.device

    with open(samples_path, 'r') as f:
        samples = json.load(f)
    
    if limit_samples:
        print(f"[N] limiting to {limit_samples} samples per domain")
        for domain in samples:
            samples[domain] = samples[domain][:limit_samples]

    # Filter tensors: we only care about weight tensors in layers
    tensor_names = [n for n, p in model.named_parameters() if "layers" in n and "weight" in n]
    print(f"[N] total layers weight tensors found: {len(tensor_names)}")

    results = []
    
    # Domains
    domains = ["code", "math", "chat", "summarization"]
    
    # For baseline
    baselines = {}
    print("[N] calculating baseline losses")
    for domain in domains:
        baselines[domain] = measure_loss(model, tokenizer, samples[domain], device)
        print(f"[N] {domain} baseline loss: {baselines[domain]:.4f}")

    # Ablation loop
    print("[N] starting ablation")
    for i, name in enumerate(tqdm(tensor_names)):
        if i % 10 == 0:
            print(f"[N] processing tensor {i}/{len(tensor_names)}: {name}")
        
        module_path = name.split(".")
        param_name = module_path[-1]
        parent_path = ".".join(module_path[:-1])
        
        # Access parent module
        parent = dict(model.named_modules())[parent_path]
        original_weight = getattr(parent, param_name).data.clone()
        
        tensor_impacts = {}
        
        # Zero out the weight
        getattr(parent, param_name).data.zero_()
        
        for domain in domains:
            ablated_loss = measure_loss(model, tokenizer, samples[domain], device)
            # Impact is the absolute increase in loss
            impact = ablated_loss - baselines[domain]
            tensor_impacts[domain] = float(impact)
            
        # Restore
        getattr(parent, param_name).data.copy_(original_weight)
        
        # Classification
        # Shared: high impact across all
        impact_vals = list(tensor_impacts.values())
        max_impact = max(impact_vals)
        max_domain = max(tensor_impacts, key=tensor_impacts.get)
        
        others = [v for k, v in tensor_impacts.items() if k != max_domain]
        avg_others = np.mean(others) if others else 0
        
        classification = "sparse"
        domain_label = None
        
        # Thresholds for loss impact (Loss increase > 0.1 is usually significant for single sample)
        if all(v > 0.2 for v in impact_vals):
            classification = "shared"
        elif max_impact > 0.2 and (max_impact - avg_others) > 0.1:
            classification = f"{max_domain}-critical"
            domain_label = max_domain
        elif max_impact < 0.05:
            classification = "sparse"
        else:
            classification = "mixed"
            if max_impact > 0.2:
                domain_label = max_domain

        results.append({
            "name": name,
            "shape": list(original_weight.shape),
            "size_bytes": original_weight.numel() * 2, # FP16
            "impact": tensor_impacts,
            "classification": classification,
            "domain": domain_label
        })

    # Summary
    summary = {
        "total_tensors": len(results),
        "shared_tensors": len([r for r in results if r["classification"] == "shared"]),
        "domain_counts": {d: len([r for r in results if r["domain"] == d]) for d in domains},
        "shared_bytes": sum(r["size_bytes"] for r in results if r["classification"] == "shared"),
        "total_bytes": sum(r["size_bytes"] for r in results)
    }
    summary["shared_percent"] = (summary["shared_bytes"] / summary["total_bytes"] * 100) if summary["total_bytes"] > 0 else 0

    output = {
        "model": model_id,
        "calibration_samples": {d: len(samples[d]) for d in domains},
        "tensors": results,
        "summary": summary
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[N] ablation complete. saved to {output_path}")
    print(f"[N] shared %: {summary['shared_percent']:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--samples", type=str, default="calibration_samples.json")
    parser.add_argument("--output", type=str, default="sensitivity_map_tensors.json")
    parser.add_argument("--limit-samples", type=int, default=None)
    args = parser.parse_args()
    
    ablate_tensors(args.model, args.samples, args.output, args.limit_samples)
