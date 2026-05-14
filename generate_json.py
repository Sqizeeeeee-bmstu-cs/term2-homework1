import json
import os

def create_dataset_json():
    dataset_dir = "data/nnUNet_raw/Dataset001_CTG"
    
    dataset_info = {
        "channel_names": {
            "0": "RGB"
        },
        "labels": {
            "background": 0,
            "line": 1
        },
        "numTrainingCases": 0,
        "file_ending": ".png"
    }
    
    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)
    
    print(f"Файл успешно создан по пути: {json_path}")

if __name__ == "__main__":
    create_dataset_json()
