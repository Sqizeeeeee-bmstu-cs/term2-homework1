import os
import sys


venv_packages = os.path.join(os.getcwd(), ".venv", "lib", "python3.12", "site-packages")
if venv_packages not in sys.path:
    sys.path.insert(0, venv_packages)

os.environ["nnUNet_raw"] = os.path.join(os.getcwd(), "data", "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(os.getcwd(), "data", "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(os.getcwd(), "data", "nnUNet_results")

try:

    from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry
    print("Модуль планирования nnU-Net v2 успешно импортирован!")
except ModuleNotFoundError as e:
    print(f"Ошибка импорта: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("Запуск планирования и предобработки для Dataset001_CTG...")
    
    sys.argv = [
        "nnUNetv2_plan_and_preprocess",
        "-d", "1",
        "-gpu_memory_target", "8",
        "--verify_dataset_integrity"
    ]
    
    plan_and_preprocess_entry()
    print("\nПредобработка успешно завершена!")
