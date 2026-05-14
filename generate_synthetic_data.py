import os
import random
import numpy as np
import cv2

COLOR_BG = (196, 206, 207)
COLOR_GRID = (169, 167, 194)
COLOR_LINE = (153, 158, 168)
COLOR_TEXT = (120, 120, 120)

def draw_dashed_line(img, mask, pt1, pt2, color, thickness):
    """Рисует линию со случайными обрывами (эффект плохой печати)."""
    if random.random() < 0.08:
        return
        
    cv2.line(img, pt1, pt2, color, thickness)

    cv2.line(mask, pt1, pt2, 1, thickness)

def generate_multiclass_case(case_id, width=1280, height=768):
    img = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)
    

    grid_space = 24
    for x in range(0, width, grid_space):
        cv2.line(img, (x, 0), (x, height), COLOR_GRID, 1)
    for y in range(0, height, grid_space):
        cv2.line(img, (0, y), (width, y), COLOR_GRID, 1)
        
    mid_y = height // 2
    
    points_fhr = []
    current_y = 220
    for x in range(0, width, 2):
        current_y += random.uniform(-7, 7)
        current_y = 0.88 * current_y + 0.12 * 220
        points_fhr.append((x, int(np.clip(current_y, 40, mid_y - 40))))
        
    points_toco = []
    num_contractions = random.randint(1, 3)
    centers = [random.randint(150, width-150) for _ in range(num_contractions)]
    for x in range(0, width, 2):
        base_y = 650 + random.uniform(-3, 3)
        for center in centers:
            base_y -= 140 * np.exp(-((x - center) / 50) ** 2)
        points_toco.append((x, int(np.clip(base_y, mid_y + 40, height - 40))))

    line_thickness = random.choice([2, 3])
    for i in range(len(points_fhr) - 1):
        draw_dashed_line(img, mask, points_fhr[i], points_fhr[i+1], COLOR_LINE, line_thickness)
        draw_dashed_line(img, mask, points_toco[i], points_toco[i+1], COLOR_LINE, line_thickness)

    x_columns = [200, 520, 830]
    for x_col in x_columns:
        x_pos = x_col + random.randint(-20, 20)
        
        for y_pos in range(30, height - 30, 40):

            cv2.putText(img, "140", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1)
            cv2.putText(mask, "140", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2, 1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    gauss = np.random.normal(0, random.uniform(2, 5), img.shape)
    img = np.clip(img + gauss, 0, 255).astype(np.uint8)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_name = f"ctg_synth_{case_id:03d}_0000.png"
    mask_name = f"ctg_synth_{case_id:03d}.png"
    
    cv2.imwrite(os.path.join("data/nnUNet_raw/Dataset001_CTG/imagesTr", img_name), img_gray)
    cv2.imwrite(os.path.join("data/nnUNet_raw/Dataset001_CTG/labelsTr", mask_name), mask)

if __name__ == "__main__":

    os.makedirs("data/nnUNet_raw/Dataset001_CTG/imagesTr", exist_ok=True)
    os.makedirs("data/nnUNet_raw/Dataset001_CTG/labelsTr", exist_ok=True)
    print("Генерация 100 многоклассовых КТГ-лент с обрывами и текстом...")
    for i in range(1, 101):
        generate_multiclass_case(i)
    print("Датасет успешно сгенерирован!")
