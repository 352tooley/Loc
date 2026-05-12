import json
import numpy as np

def analyze_overlap(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    domains = list(results.keys())
    
    # Collect all used experts per domain across all layers
    domain_experts = {}
    for domain in domains:
        all_experts = set()
        for layer, experts in results[domain]["experts"].items():
            for e in experts:
                all_experts.add(f"{layer}_{e}")
        domain_experts[domain] = all_experts
        print(f"[N] {domain} total unique expert-layers: {len(all_experts)}")

    # Overlap matrix
    print("\n[N] Overlap Matrix (Jaccard Similarity):")
    print("      " + " ".join([d[:4] for d in domains]))
    for d1 in domains:
        row = f"{d1[:4]} "
        for d2 in domains:
            intersection = len(domain_experts[d1].intersection(domain_experts[d2]))
            union = len(domain_experts[d1].union(domain_experts[d2]))
            sim = intersection / union if union > 0 else 0
            row += f"{sim:.2f} "
        print(row)

    # Intersection of all domains
    common = set.intersection(*domain_experts.values())
    print(f"\n[N] Experts common to ALL domains: {len(common)} ({len(common)/1440*100:.2f}%)")
    
    # Unique to each domain
    for domain in domains:
        others = set().union(*[domain_experts[d] for d in domains if d != domain])
        unique = domain_experts[domain] - others
        print(f"[N] Experts unique to {domain}: {len(unique)} ({len(unique)/1440*100:.2f}%)")

if __name__ == "__main__":
    analyze_overlap("moe_expert_prediction.json")
