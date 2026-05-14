import os
import sys
import torch

venv_packages = os.path.join(os.getcwd(), ".venv", "lib", "python3.12", "site-packages")
if venv_packages not in sys.path:
    sys.path.insert(0, venv_packages)

os.environ["nnUNet_raw"] = os.path.join(os.getcwd(), "data", "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(os.getcwd(), "data", "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(os.getcwd(), "data", "nnUNet_results")

if torch.backends.mps.is_available():
    device_type = "mps"
    print("Обнаружен графический чип Apple Silicon! Обучение пойдет на MPS.")
else:
    device_type = "cpu"
    print("Внимание: MPS недоступен, обучение пойдет на CPU.")

try:
    from nnunetv2.run.run_training import get_trainer_from_args
    print("Модуль обучения nnU-Net v2 успешно импортирован")
except ModuleNotFoundError as e:
    print(f"Ошибка импорта: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print(f"Старт обучения модели для КТГ")
    
    trainer = get_trainer_from_args(
        dataset_name_or_id="1",
        configuration="2d",
        fold=0,
        plans_identifier="nnUNetPlans",
        device=torch.device(device_type)
    )
    
    trainer.num_epochs = 35
    print(f" Количество эпох принудительно изменено на: {trainer.num_epochs}")
    
    trainer.save_every = 1
    
    trainer.initialize()
    trainer.run_training()
    
    print("\nОбучение успешно завершено")
