import os
import sys
import torch
import cv2
import numpy as np

venv_packages = os.path.join(os.getcwd(), ".venv", "lib", "python3.12", "site-packages")
if venv_packages not in sys.path:
    sys.path.insert(0, venv_packages)

try:
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    print("Архитектура UNet успешно импортирована!")
except ModuleNotFoundError:
    print("Ошибка: не найден пакет dynamic_network_architectures.")
    sys.exit(1)

def run_direct():
    input_path = "sample.jpg"
    if not os.path.exists(input_path):
        print(f"Положите тестовый файл {input_path} в корень папки!")
        return
        
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = img.shape
    
    img_resized = cv2.resize(img, (1280, 768), interpolation=cv2.INTER_AREA)
    
    img_input = (img_resized - img_resized.mean()) / (img_resized.std() + 1e-8)
    
    tensor = torch.from_numpy(img_input).float().unsqueeze(0).unsqueeze(0)
    
    model = PlainConvUNet(
        input_channels=1,
        n_stages=8,
        features_per_stage=[32, 64, 128, 256, 512, 512, 512, 512],
        conv_op=torch.nn.Conv2d,
        kernel_sizes=[[3, 3]] * 8,
        strides=[[1, 1]] + [[2, 2]] * 7,
        n_conv_per_stage=[2] * 8,
        n_conv_per_stage_decoder=[2] * 7,
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm2d,
        norm_op_kwargs={'eps': 1e-05, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        num_classes=3
    )
    
    fold_dir = "data/nnUNet_results/Dataset001_CTG/nnUNetTrainer__nnUNetPlans__2d/fold_0"
    checkpoint_path = os.path.join(fold_dir, "checkpoint_final.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(fold_dir, "checkpoint_latest.pth")
        
    if not os.path.exists(checkpoint_path):
        print("Ошибка: веса модели не найдены.")
        return

    print(f"Загрузка весов из: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['network_weights'])
    model.eval()
    
    print("Прямой многоклассовый проход нейросети...")
    with torch.no_grad():
        output = model(tensor)

        mask = torch.argmax(output.squeeze(0), dim=0).numpy().astype(np.uint8)
        
    line_mask = np.where(mask == 1, 255, 0).astype(np.uint8)
    
    mask_final = cv2.resize(line_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    cv2.imwrite("nnunet_ctg_output.png", mask_final)
    print("Успех! Маска графиков сохранена в nnunet_ctg_output.png")

if __name__ == "__main__":
    run_direct()
