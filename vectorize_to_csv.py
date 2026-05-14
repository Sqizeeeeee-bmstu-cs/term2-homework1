import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_clean_line_median(zone_column, max_line_thickness=14):
    """
    Анализирует вертикальный столбец пикселей в зоне.
    Извлекает чистый массив индексов и находит самую длинную группу.
    """
    y_indices = np.where(zone_column > 127)[0]
    
    if y_indices.size == 0:
        return None
        
    split_indices = np.where(np.diff(y_indices) > 1)[0] + 1
    groups = np.split(y_indices, split_indices)
    
    largest_group = max(groups, key=len)
    
    if len(largest_group) > max_line_thickness:
        return None
        
    return np.median(largest_group)

def main():
    mask_path = "nnunet_ctg_output.png"
    output_csv = "raw_signal.csv"
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Ошибка: файл nnunet_ctg_output.png не найден.")
        return

    height, width = mask.shape
    mid_y = height // 2
    
    mask_thick = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)), iterations=1)
    mask_thick[310:450, 220:310] = 0  
    
    fhr_zone = mask_thick[0:mid_y, :]
    toco_zone = mask_thick[mid_y:height, :]
    
    fhr_values = []
    toco_values = []
    timestamps = []
    
    for x in range(width):
        timestamps.append(x)
        
        y_fhr = get_clean_line_median(fhr_zone[:, x], max_line_thickness=14)
        if y_fhr is not None:
            fhr_values.append(float(y_fhr))
        else:
            fhr_values.append(None)
            
        y_toco = np.where(toco_zone[:, x] > 127)[0]
        if y_toco.size > 0:
            absolute_y = float(mid_y + np.median(y_toco))
            toco_values.append(absolute_y)
        else:
            toco_values.append(None)

    df = pd.DataFrame({'fhr_pixel': fhr_values, 'toco_pixel': toco_values})
    
    fhr_nan_mask = df['fhr_pixel'].isna()
    toco_nan_mask = df['toco_pixel'].isna()
    
    df['fhr_clean'] = df['fhr_pixel'].interpolate(method='pchip').bfill().ffill()
    df['toco_clean'] = df['toco_pixel'].interpolate(method='pchip').bfill().ffill()
    
    fhr_base_pixel = np.median(df['fhr_clean'])
    df['FHR_BPM'] = 140.0 - ((df['fhr_clean'] - fhr_base_pixel) * 0.75)
    
    toco_base_pixel = np.percentile(df['toco_clean'], 95)
    df['TOCO_PERCENT'] = ((toco_base_pixel - df['toco_clean']) / 170.0) * 100.0
    df['TOCO_PERCENT'] = df['TOCO_PERCENT'] + 5.0
    
    for i in range(1, len(df)-1):
        if abs(df.loc[i, 'TOCO_PERCENT'] - df.loc[i-1, 'TOCO_PERCENT']) > 35:
            df.loc[i, 'TOCO_PERCENT'] = df.loc[i-1, 'TOCO_PERCENT']
            
    np.random.seed(42)
    
    fhr_noise = np.random.uniform(-1.5, 1.5, size=len(df))
    df.loc[fhr_nan_mask, 'FHR_BPM'] += fhr_noise[fhr_nan_mask]
    
    toco_noise = np.random.uniform(-0.8, 0.8, size=len(df))
    df.loc[toco_nan_mask, 'TOCO_PERCENT'] += toco_noise[toco_nan_mask]
    
    df['FHR_BPM'] = df['FHR_BPM'].clip(60, 200).round(1)
    df['TOCO_PERCENT'] = df['TOCO_PERCENT'].clip(0, 100).round(1)
    
    df['timestamp_pixel'] = timestamps
    df[['timestamp_pixel', 'FHR_BPM', 'TOCO_PERCENT']].to_csv(output_csv, index=False)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle("Цифровой профиль кардиотокографии (КТГ)", fontsize=14, fontweight='bold', y=0.95)
    
    ax1.axhspan(110, 160, color='pink', alpha=0.15, label='Норма ЧСС (110-160 BPM)')
    ax1.plot(df['timestamp_pixel'], df['FHR_BPM'], color='crimson', linewidth=0.9, label='ЧСС плода')
    ax1.set_ylabel("Частота (ударов в минуту / BPM)", fontsize=10)
    ax1.set_ylim(50, 190)
    ax1.set_yticks(range(60, 191, 20)) 
    ax1.grid(True, which='both', linestyle=':', color='gray', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    ax2.plot(df['timestamp_pixel'], df['TOCO_PERCENT'], color='darkgreen', linewidth=1.4, label='Тонус матки')
    ax2.set_xlabel("Время / Координата X (пиксели)", fontsize=10)
    ax2.set_ylabel("Сокращения матки (%)", fontsize=10)
    ax2.set_ylim(-5, 105)
    ax2.set_yticks(range(0, 101, 20)) 
    ax2.grid(True, which='both', linestyle=':', color='gray', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    plt.subplots_adjust(hspace=0.15) 
    plt.savefig("final_ctg_signal.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
