import os
import shutil
import zipfile
import glob
import numpy as np
import pandas as pd
import cv2
import scipy.stats as stats
from skimage.feature import graycomatrix, graycoprops

zip_path = "Leaves.zip"
extracted_folder = "flavia_raw"

if os.path.exists(zip_path):
    print("Descompactando o dataset Flavia...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)

ranges = [
    (1001, 1059, 1),  (1060, 1122, 2),  (1552, 1616, 3),  (1123, 1194, 4),
    (1195, 1267, 5),  (1268, 1323, 6),  (1324, 1385, 7),  (1386, 1437, 8),
    (1497, 1551, 9),  (1438, 1496, 10), (2001, 2050, 11), (2051, 2113, 12),
    (2114, 2165, 13), (2166, 2230, 14), (2231, 2290, 15), (2291, 2358, 16),
    (2359, 2423, 17), (2424, 2485, 18), (2486, 2546, 19), (2547, 2612, 20),
    (2616, 2675, 21), (3001, 3055, 22), (3056, 3110, 23), (3111, 3175, 24),
    (3176, 3229, 25), (3230, 3281, 26), (3282, 3334, 27), (3335, 3389, 28),
    (3390, 3446, 29), (3447, 3510, 30), (3511, 3563, 31), (3564, 3622, 32)
]

output_structured = "flavia_by_class"
os.makedirs(output_structured, exist_ok=True)

for i in range(1, 33):
    os.makedirs(os.path.join(output_structured, f"class_{i:02d}"), exist_ok=True)

for root, _, files in os.walk(extracted_folder):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            try:
                img_num = int(os.path.splitext(file)[0])
                for start, end, class_id in ranges:
                    if start <= img_num <= end:
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(output_structured, f"class_{class_id:02d}", file)
                        shutil.copy(src_path, dst_path)
                        break
            except ValueError:
                continue

def segment_leaf(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, None, None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    leaf_gray = cv2.bitwise_and(gray, gray, mask=mask)
    return gray, mask, leaf_gray

def extract_all_features(image_path):
    gray, mask, leaf_gray = segment_leaf(image_path)
    if gray is None or np.sum(mask) == 0:
        return None

    gray_q = (leaf_gray / 4).astype(np.uint8)
    glcm = graycomatrix(gray_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=64, symmetric=True, normed=True)
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    correlation   = graycoprops(glcm, 'correlation').mean()
    homogeneity   = graycoprops(glcm, 'homogeneity').mean()
    asm           = graycoprops(glcm, 'ASM').mean()
    energy        = graycoprops(glcm, 'energy').mean()

    pixels = leaf_gray[mask > 0]
    mean_val = np.mean(pixels)
    std_val  = np.std(pixels)
    var_val  = np.var(pixels)
    smoothness = 1.0 - (1.0 / (1.0 + var_val))
    third_moment = stats.skew(pixels) * (std_val ** 3)
    hist_p, _ = np.histogram(pixels, bins=256, range=(0, 256), density=True)
    hist_p = hist_p[hist_p > 0]
    entropy_val = -np.sum(hist_p * np.log2(hist_p))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    c = max(contours, key=cv2.contourArea)
    
    area_contorno = cv2.contourArea(c)
    perimetro_contorno = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    area_casco = cv2.contourArea(hull)
    perimetro_casco = cv2.arcLength(hull, True)
    
    if len(c) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        a, b = max(MA, ma)/2.0, min(MA, ma)/2.0
        eccentricity = np.sqrt(1 - (b/a)**2) if a > 0 else 0
    else:
        eccentricity = 0

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(h) / w if w > 0 else 0

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
    (_, _), r_circunscrito = cv2.minEnclosingCircle(c)
    elongation = 1 - (max_val / r_circunscrito) if r_circunscrito > 0 else 0

    return {
        'Dissimilaridade': dissimilarity,
        'Correlacao': correlation,
        'Homogeneidade': homogeneity,
        'ASM': asm,
        'Energia': energy,
        'Excentricidade': eccentricity,
        'Razao_Aspecto': aspect_ratio,
        'Alongamento': elongation,
        'Solidez': area_contorno / area_casco if area_casco > 0 else 0,
        'Fator_Isoperimetrico': (4 * np.pi * area_contorno) / (perimetro_contorno ** 2) if perimetro_contorno > 0 else 0,
        'Convexidade': perimetro_casco / perimetro_contorno if perimetro_contorno > 0 else 0,
        'Media_Intensidade': mean_val,
        'Desvio_Padrao': std_val,
        'Suavidade': smoothness,
        'Terceiro_Momento': third_moment,
        'Entropia': entropy_val
    }

data_records = []
classes = sorted([d for d in os.listdir(output_structured) if os.path.isdir(os.path.join(output_structured, d)) and d.startswith("class_")])

print("\n" + "="*60)
print("INICIANDO EXTRAÇÃO DE ATRIBUTOS BRUTOS POR CLASSE")
print("="*60)

for class_name in classes:
    class_num = int(class_name.split("_")[1])
    class_path = os.path.join(output_structured, class_name)
    img_files = glob.glob(os.path.join(class_path, "*.jpg")) + glob.glob(os.path.join(class_path, "*.png"))
    
    count = 0
    for img_path in img_files:
        feats = extract_all_features(img_path)
        if feats is not None:
            feats['Arquivo'] = os.path.basename(img_path)
            feats['Classe'] = class_name
            data_records.append(feats)
            count += 1

    print(f"Classe: {class_name} | Número: {class_num:02d} | Imagens Processadas: {count}")

df = pd.DataFrame(data_records)
df.to_csv("atributos_flavia_32_classes.csv", index=False)

print("="*60)
print("Extração concluída com sucesso!")
print("Planilha com atributos brutos criada: 'atributos_flavia_32_classes.csv'")
print("="*60)

print("\n" + "="*70)
print("ANÁLISE EXPLORATÓRIA DO CONJUNTO DE DADOS (DATASET FLAVIA)")
print("="*70)

total_imagens = len(df)
print(f"• Quantidade total de imagens processadas: {total_imagens}")

frequencia_classes = df['Classe'].value_counts().sort_index()
proporcao_classes = (df['Classe'].value_counts(normalize=True).sort_index() * 100).round(2)

df_distribuicao = pd.DataFrame({
    'Qtd_Imagens': frequencia_classes,
    'Percentual (%)': proporcao_classes
})

print("\n• Distribuição das imagens por classe:")
print(df_distribuicao.to_string())

media_imagens = frequencia_classes.mean()
std_imagens = frequencia_classes.std()
min_imagens = frequencia_classes.min()
max_imagens = frequencia_classes.max()
cv = (std_imagens / media_imagens) * 100

print("\n• Análise Estatística de Balanceamento:")
print(f"  - Média de imagens por classe: {media_imagens:.2f}")
print(f"  - Desvio Padrão: {std_imagens:.2f}")
print(f"  - Menor classe (Mínimo): {min_imagens} imagens")
print(f"  - Maior classe (Máximo): {max_imagens} imagens")
print(f"  - Coeficiente de Variação (CV): {cv:.2f}%")

if cv < 15:
    status_balanceamento = "ALTAMENTE BALANCEADO (variação insignificante entre as classes)."
elif cv <= 30:
    status_balanceamento = "MODERADAMENTE BALANCEADO (pequenas variações na quantidade de imagens)."
else:
    status_balanceamento = "DESBALANCEADO (diferença expressiva na representatividade das classes)."

print(f"\n• Diagnóstico do Conjunto: O dataset é considerado {status_balanceamento}")
print("="*70 + "\n")