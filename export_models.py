import torch
import os
import onnx
from affective_head import AffectiveHead, TEMPORAL_DIM
from gaze_head import GazeHead

def export_affective_model():
    print("Exporting AffectiveHead components to ONNX... (2 models: CNN and TCN)")
    head = AffectiveHead()
    
    # Export 1: The CNN feature extractor
    if head.feature_extractor is not None:
        head.feature_extractor.eval()
        dummy_face = torch.randn(1, 1, 48, 48).to(head.device)
        cnn_onnx_path = os.path.join("models", "affective_cnn.onnx")
        
        torch.onnx.export(
            head.feature_extractor, 
            (dummy_face,), 
            cnn_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_face'],
            output_names=['cnn_embedding']
        )
        onnx.checker.check_model(onnx.load(cnn_onnx_path))
        print(f"Successfully exported Affective CNN to {cnn_onnx_path}")
    
    # Export 2: The TCN layer
    head.tcn.eval()
    dummy_seq = torch.randn(1, 15, head.embed_dim + TEMPORAL_DIM).to(head.device)
    tcn_onnx_path = os.path.join("models", "affective_tcn.onnx")
    
    torch.onnx.export(
        head.tcn, 
        (dummy_seq,), 
        tcn_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_feature_seq'],
        output_names=['stress_logit']
    )
    onnx.checker.check_model(onnx.load(tcn_onnx_path))
    print(f"Successfully exported Affective TCN to {tcn_onnx_path}")

def export_gaze_model():
    print("Exporting GazeHead to ONNX...")
    head = GazeHead()
    head.model.eval()
    
    # Seq_len=15, input_dim=10
    dummy_seq = torch.randn(1, 15, 10).to(head.device)
    dummy_pose = torch.randn(1, 3).to(head.device)
    
    onnx_path = os.path.join("models", "gaze_hybrid.onnx")
    
    torch.onnx.export(
        head.model, 
        (dummy_seq, dummy_pose), 
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_seq', 'input_pose'],
        output_names=['screen_coords']
    )
    
    # Verify the exported ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"Successfully exported and validated GazeHead to {onnx_path}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    try:
        export_affective_model()
        export_gaze_model()
        print("\nAll models successfully exported to ONNX format. Pre-quantization requirement met.")
    except Exception as e:
        print(f"\nError exporting models: {e}")
