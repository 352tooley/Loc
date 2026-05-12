import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import argparse

class MOELoader:
    def __init__(self, model_id, device="cpu"):
        self.model_id = model_id
        self.device = device
        print(f"[N] loading MoE base: {model_id}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            print(f"[N] fast tokenizer failed: {e}. trying slow tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
            except Exception as e2:
                 print(f"[N] slow tokenizer failed: {e2}. trying fallback to Qwen2Tokenizer...")
                 from transformers import Qwen2Tokenizer
                 self.tokenizer = Qwen2Tokenizer.from_pretrained(model_id)
        
        # Load model, but we want to simulate selective expert loading
        # For real selective loading in transformers, we'd need custom layer implementations.
        # Here we simulate by loading the full model and tracking expert usage.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        self.expert_usage = {}
        self.hooks = []

    def _register_expert_hooks(self):
        def get_hook(layer_idx):
            def hook(module, input, output):
                # output is usually (router_logits, expert_indices) or similar
                # For Qwen2MoE, it's often in the MoE layer
                # We need to find where the routing happens.
                # Usually: router_logits = gate(input)
                # We'll hook into the gate/router module.
                pass
            return hook

        # Find gate/router modules
        for name, module in self.model.named_modules():
            # Specifically target the sparse experts router gate
            if "mlp.gate" in name and "shared" not in name:
                # print(f"[N] found router: {name}")
                def gate_hook(name=name):
                    def hook(module, input, output):
                        # output is [batch, seq, num_experts]
                        # We want to see which experts were selected (top-k)
                        num_experts = output.shape[-1]
                        k = min(self.model.config.num_experts_per_tok, num_experts)
                        
                        probs = torch.softmax(output, dim=-1)
                        top_k_indices = torch.topk(probs, k=k, dim=-1).indices
                        unique_experts = torch.unique(top_k_indices).tolist()
                        
                        if name not in self.expert_usage:
                            self.expert_usage[name] = []
                        self.expert_usage[name].extend(unique_experts)
                    return hook
                self.hooks.append(module.register_forward_hook(gate_hook(name)))
        print(f"[N] registered {len(self.hooks)} sparse router hooks")

    def predict_experts(self, samples):
        self.expert_usage = {}
        self._register_expert_hooks()
        
        print("[N] running prediction passes")
        for sample in tqdm(samples):
            inputs = self.tokenizer(sample, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model(**inputs)
        
        # Cleanup hooks
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
        # Aggregate experts per layer
        predicted = {}
        for name, usage in self.expert_usage.items():
            unique_used = sorted(list(set(usage)))
            predicted[name] = unique_used
            
        return predicted

    def get_expert_stats(self, predicted):
        total_experts = 0
        used_experts = 0
        # Assuming 8 experts per layer for Qwen MoE Tiny?
        # Let's check the config
        num_experts = self.model.config.num_experts if hasattr(self.model.config, "num_experts") else 8
        num_layers = self.model.config.num_hidden_layers
        
        total_possible = num_layers * num_experts
        used_count = sum(len(v) for v in predicted.values())
        
        return {
            "total_possible": total_possible,
            "used_count": used_count,
            "used_percent": (used_count / total_possible) * 100
        }

def run_moe_test(model_path, samples_path):
    loader = MOELoader(model_path, device="auto")
    
    with open(samples_path, 'r') as f:
        samples = json.load(f)
    
    domain_results = {}
    for domain, domain_samples in samples.items():
        print(f"[N] predicting for domain: {domain}")
        # Use first 10 samples for prediction
        predicted = loader.predict_experts(domain_samples[:10])
        stats = loader.get_expert_stats(predicted)
        domain_results[domain] = {
            "stats": stats,
            "experts": predicted
        }
        print(f"[N] {domain} expert usage: {stats['used_percent']:.2f}%")

    output_path = "moe_expert_prediction.json"
    with open(output_path, 'w') as f:
        json.dump(domain_results, f, indent=2)
    print(f"[N] results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/qwen3-moe-tiny")
    parser.add_argument("--samples", type=str, default="calibration_samples.json")
    args = parser.parse_args()
    
    run_moe_test(args.model, args.samples)
