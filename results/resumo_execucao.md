# Resumo da Execução

## Dataset utilizado

- Fonte: Spotify Tracks Dataset.
- Registros válidos após limpeza: 113,550.
- Classe positiva: popularity >= 70.

## Resultado principal

O melhor modelo pelo F1-score foi `arvore_decisao` com F1-score de 0.1446.

| Modelo          | Acurácia | Precisão | Recall | F1-score |   VP |    VN |    FP |   FN |
| --------------- | -------: | -------: | -----: | -------: | ---: | ----: | ----: | ---: |
| arvore_decisao  |   0.5561 |   0.0797 | 0.7791 |   0.1446 | 1065 | 14722 | 12299 |  302 |
| rede_neural_mlp |   0.9520 |   1.0000 | 0.0029 |   0.0058 |    4 | 27021 |     0 | 1363 |

## Atributos mais importantes na Árvore de Decisão

| Atributo         | Importância |
| ---------------- | ----------: |
| instrumentalness |      0.2649 |
| energy           |      0.1274 |
| acousticness     |      0.1208 |
| track_genre_pop  |      0.0712 |
| liveness         |      0.0634 |
