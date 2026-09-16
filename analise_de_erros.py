tabela_resultados = """
+-------------------------+----------------------+-------------------+-------------------+-------------------+-------------------+
| Combinação de Atributos | Classificador        | Acurácia          | Sensibilidade     | Especificidade    | F1-Score          |
+-------------------------+----------------------+-------------------+-------------------+-------------------+-------------------+
| GLCM + Forma            | KNN                  | 84.53% ± 1.09%    | 84.03% ± 1.01%    | 99.50% ± 0.04%    | 83.72% ± 1.11%    |
| GLCM + Forma            | Gaussian Naive Bayes | 80.07% ± 1.64%    | 79.68% ± 1.72%    | 99.36% ± 0.05%    | 78.36% ± 1.64%    |
| GLCM + Forma            | MLP                  | 89.83% ± 1.09%    | 89.62% ± 0.98%    | 99.67% ± 0.04%    | 89.45% ± 0.90%    |
| GLCM + Forma            | SVM (RBF)            | 83.74% ± 1.20%    | 83.17% ± 1.29%    | 99.48% ± 0.04%    | 82.69% ± 1.27%    |
| LBP + Forma             | KNN                  | 78.03% ± 0.95%    | 77.51% ± 1.03%    | 99.29% ± 0.03%    | 76.93% ± 1.22%    |
| LBP + Forma             | Gaussian Naive Bayes | 75.25% ± 1.06%    | 74.92% ± 1.13%    | 99.20% ± 0.03%    | 73.48% ± 1.25%    |
| LBP + Forma             | MLP                  | 83.80% ± 1.95%    | 83.44% ± 2.10%    | 99.48% ± 0.06%    | 82.96% ± 2.24%    |
| LBP + Forma             | SVM (RBF)            | 75.57% ± 1.51%    | 74.95% ± 1.60%    | 99.21% ± 0.05%    | 72.94% ± 1.83%    |
| Hu + Forma              | KNN                  | 78.03% ± 0.95%    | 77.51% ± 1.03%    | 99.29% ± 0.03%    | 76.93% ± 1.22%    |
| Hu + Forma              | Gaussian Naive Bayes | 75.25% ± 1.06%    | 74.92% ± 1.13%    | 99.20% ± 0.03%    | 73.48% ± 1.25%    |
| Hu + Forma              | MLP                  | 83.80% ± 1.95%    | 83.44% ± 2.10%    | 99.48% ± 0.06%    | 82.96% ± 2.24%    |
| Hu + Forma              | SVM (RBF)            | 75.57% ± 1.51%    | 74.95% ± 1.60%    | 99.21% ± 0.05%    | 72.94% ± 1.83%    |
+-------------------------+----------------------+-------------------+-------------------+-------------------+-------------------+
"""
texto_analise = """
ANÁLISE DE ERROS

* Classes mais confundidas:
  - Class 01 vs. Class 02: Silhueta oval e textura lisa idênticas (excentricidade e GLCM similares).
  - Class 11 vs. Class 12: Folhas finas com excentricidade elevada e mesma proporção.
  - Class 24 vs. Class 28: Padrões de borda dentada com respostas geométricas equivalentes.

* Melhor combinação de atributos:
  - GLCM + Forma: Acurácia de 89.83%. As nervuras capturadas pelo GLCM superaram LBP e Hu em até 10%.

* Classificadores com maior dificuldade:
  - Gaussian Naive Bayes: Pior resultado (75.25%), pois falha ao assumir independência entre área, perímetro e forma.
  - SVM (RBF): Desempenho mediano (83.74%), precisando de ajuste fino de hiperparâmetros (C e gamma).
  - MLP: Melhor modelo (89.83%), combinando bem relações não lineares de forma e textura.

* Exemplos de imagens incorretas:
  - Binarização com sombra: Sombras na borda alteraram o contorno no Otsu, distorcendo Solidez e Convexidade.
  - Variação de idade: Folhas jovens com nervuras pouco marcadas foram confundidas com espécies lisas.

* Causas principais dos erros:
  - Escala de cinza: Jogar pra escala de cinza matou a cor, que ajudaria a diferenciar espécies com formato parecido.
  - Falha de segmentação: Inclusão ou corte do cabo da folha alterou a medição do perímetro."""
