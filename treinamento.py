import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

csv_path = "atributos_flavia_32_classes.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Arquivo '{csv_path}' não encontrado. Execute a extração primeiro.")

print("Carregando dataset de atributos...")
df = pd.read_csv(csv_path)

X_df = df.drop(columns=['Arquivo', 'Classe'])
y = df['Classe'].values

cols_forma = [c for c in X_df.columns if c in [
    'Excentricidade', 'Razao_Aspecto', 'Alongamento', 'Solidez', 
    'Fator_Isoperimetrico', 'Convexidade'
]]

cols_glcm = [c for c in X_df.columns if c in [
    'Dissimilaridade', 'Correlacao', 'Homogeneidade', 'ASM', 'Energia'
]]

cols_lbp = [c for c in X_df.columns if c.startswith('LBP_')]

cols_hu = [c for c in X_df.columns if c.startswith('Hu_')]

combinacoes_atributos = {
    'GLCM + Forma': cols_glcm + cols_forma,
    'LBP + Forma': cols_lbp + cols_forma,
    'Hu + Forma': cols_hu + cols_forma
}

classificadores = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Gaussian Naive Bayes': GaussianNB(),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
}

def calcular_especificidade(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    especificidades = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        esp = tn / (tn + fp) if (tn + fp) > 0 else 0
        especificidades.append(esp)
    return np.mean(especificidades)

metricas = {
    'acuracia': make_scorer(accuracy_score),
    'sensibilidade': make_scorer(recall_score, average='macro'),
    'especificidade': make_scorer(calcular_especificidade),
    'f1_score': make_scorer(f1_score, average='macro')
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

resultados_finais = []

print("\n" + "=" * 80)
print("INICIANDO TREINAMENTO E AVALIAÇÃO DOS MODELOS")
print("=" * 80)

for nome_comb, cols in combinacoes_atributos.items():
    X_sub = X_df[cols].values
    
    for nome_clf, clf in classificadores.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        
        cv_results = cross_validate(
            pipeline, X_sub, y, cv=skf, scoring=metricas, n_jobs=-1
        )
        
        acc_m, acc_std = np.mean(cv_results['test_acuracia']), np.std(cv_results['test_acuracia'])
        sens_m, sens_std = np.mean(cv_results['test_sensibilidade']), np.std(cv_results['test_sensibilidade'])
        esp_m, esp_std = np.mean(cv_results['test_especificidade']), np.std(cv_results['test_especificidade'])
        f1_m, f1_std = np.mean(cv_results['test_f1_score']), np.std(cv_results['test_f1_score'])
        
        resultados_finais.append({
            'Combinação de Atributos': nome_comb,
            'Classificador': nome_clf,
            'Acurácia': f"{acc_m*100:.2f}% ± {acc_std*100:.2f}%",
            'Sensibilidade': f"{sens_m*100:.2f}% ± {sens_std*100:.2f}%",
            'Especificidade': f"{esp_m*100:.2f}% ± {esp_std*100:.2f}%",
            'F1-Score': f"{f1_m*100:.2f}% ± {f1_std*100:.2f}%"
        })
        
        print(f"[{nome_comb}] | {nome_clf:20s} -> Acurácia: {acc_m*100:.2f}% ± {acc_std*100:.2f}%")

df_tabela = pd.DataFrame(resultados_finais)

output_dir = "resultados_32_treinamentos"
os.makedirs(output_dir, exist_ok=True)
df_tabela.to_csv(os.path.join(output_dir, "tabela_comparativa_resultados.csv"), index=False)

print("\n" + "=" * 80)
print("TABELA COMPARATIVA CONSOLIDADA DE RESULTADOS (2.10)")
print("=" * 80)
print(df_tabela.to_string(index=False))
print("\n" + "=" * 80)
print(f"Tabela salva com sucesso em: '{output_dir}/tabela_comparativa_resultados.csv'")
print("=" * 80)