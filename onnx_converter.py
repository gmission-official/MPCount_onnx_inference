import torch
from models import DGModel_final

def convert_to_onnx(model_path, onnx_path, input_shape=(1, 3, 512, 512), device='cuda'):

    model = DGModel_final().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    dummy_input = torch.randn(input_shape).to(device)

    torch.onnx.export(
        model,                      
        dummy_input,                
        onnx_path,                  
        export_params=True,         
        opset_version=11,           
        do_constant_folding=True,   
        input_names=['input'],      
        output_names=['output'],    
        dynamic_axes={             
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model has been converted to ONNX format and saved to {onnx_path}")

if __name__ == "__main__":

    model_path = 'weights/sta.pth'  
    onnx_path = 'weights/sta.onnx'               
    input_shape = (1, 3, 512, 512)                
    device = 'cuda'                           

    convert_to_onnx(model_path, onnx_path, input_shape, device)