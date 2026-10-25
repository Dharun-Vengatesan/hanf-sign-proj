"""
Model Export Pipeline for Unity
Exports the trained ASL neural network to:
1. Assets/Models/asl_model.onnx (ONNX for Unity Sentis / ONNX Runtime)
2. Assets/StreamingAssets/asl_weights.json (Standalone Pure C# Inference in Unity)
"""

import os
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import json
import torch
import numpy as np
import onnx
from train_model import ASLSignNet

def export_asl_model(model_path='python/models/best_asl_model.pth',
                     onnx_out='Assets/Models/asl_model.onnx',
                     json_out='Assets/StreamingAssets/asl_weights.json'):
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(onnx_out), exist_ok=True)
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    
    print(f"Loading checkpoint: {model_path}...")
    model = ASLSignNet()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    model.eval()
    
    # 1. Export to ONNX
    dummy_input = torch.randn(1, 63, dtype=torch.float32)
    
    print(f"Exporting to ONNX format: {onnx_out}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_out,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['landmarks'],
        output_names=['logits'],
        dynamic_axes={
            'landmarks': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model integrity
    onnx_model = onnx.load(onnx_out)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification PASSED!")
    
    # 2. Export to JSON Weights for Pure C# Engine
    print(f"Exporting layer weights to JSON: {json_out}...")
    state = model.state_dict()
    
    weights_dict = {
        "architecture": "MLP",
        "input_features": 63,
        "classes": [chr(i) for i in range(ord('A'), ord('Z') + 1)],
        "layers": {
            "fc1": {
                "weight": state['fc1.weight'].numpy().flatten().tolist(), # [128 * 63]
                "bias": state['fc1.bias'].numpy().flatten().tolist()       # [128]
            },
            "fc2": {
                "weight": state['fc2.weight'].numpy().flatten().tolist(), # [64 * 128]
                "bias": state['fc2.bias'].numpy().flatten().tolist()       # [64]
            },
            "fc3": {
                "weight": state['fc3.weight'].numpy().flatten().tolist(), # [26 * 64]
                "bias": state['fc3.bias'].numpy().flatten().tolist()       # [26]
            }
        }
    }
    
    with open(json_out, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    print("JSON weights export PASSED!")
    
    # 3. Numerical Verification: Test with dummy input
    with torch.no_grad():
        py_out = model(dummy_input).numpy().flatten()
        
    # Test JSON weights forward pass in Python to simulate pure C# execution
    w1 = np.array(weights_dict['layers']['fc1']['weight']).reshape(128, 63)
    b1 = np.array(weights_dict['layers']['fc1']['bias'])
    w2 = np.array(weights_dict['layers']['fc2']['weight']).reshape(64, 128)
    b2 = np.array(weights_dict['layers']['fc2']['bias'])
    w3 = np.array(weights_dict['layers']['fc3']['weight']).reshape(26, 64)
    b3 = np.array(weights_dict['layers']['fc3']['bias'])
    
    x = dummy_input.numpy().flatten()
    h1 = np.maximum(0, w1 @ x + b1) # ReLU
    h2 = np.maximum(0, w2 @ h1 + b2) # ReLU
    json_out_logits = w3 @ h2 + b3
    
    diff = np.max(np.abs(py_out - json_out_logits))
    print(f"Max numerical difference between PyTorch and exported weights: {diff:.8f}")
    assert diff < 1e-5, "Exported weights mismatch!"
    
    print("\nAll model artifacts exported and verified successfully!")
    print(f"  1. {onnx_out} ({os.path.getsize(onnx_out) / 1024:.1f} KB)")
    print(f"  2. {json_out} ({os.path.getsize(json_out) / 1024:.1f} KB)")

if __name__ == '__main__':
    export_asl_model()
