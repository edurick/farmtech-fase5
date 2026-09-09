"""Gera a estrutura do notebook; executar depois com scripts/verificar.py."""
from pathlib import Path
import textwrap
import nbformat as nbf
ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
def code(text): cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))
md('''
# FarmTech Solutions · Fase 5
## Produtividade agrícola: previsão e descoberta de padrões
**Eduardo Rick · RM573745 · entrega individual · 08/09/2026**

Esta análise utiliza exclusivamente `dados/crop_yield.csv`, fornecido no portal da atividade.
O objetivo é estimar `Yield` a partir da cultura e de quatro condições climáticas, comparar cinco
algoritmos de regressão e investigar grupos de condições associados à produtividade.

**Roteiro:** 1. Dados e qualidade → 2. Protocolo de avaliação → 3. Exploração →
4. Cinco modelos → 5. Teste e limitações → 6. Clusters → 7. Outliers → 8. Previsão e conclusão.

**Como executar:** Python 3.12, dependências de `requirements.txt`, kernel do ambiente criado
conforme o README e **Restart Kernel and Run All Cells**. A base é local; não é necessário
acessar a internet depois de instalar as dependências. O notebook funciona a partir da raiz do repositório.
O sufixo `pbl_fase4` no nome foi mantido por exigência literal do enunciado da Fase 5.
''')
code('''
from pathlib import Path
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from IPython.display import display, Markdown
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

SEED = 42
ROOT = Path.cwd()
if not (ROOT / 'dados/crop_yield.csv').exists():
    raise FileNotFoundError('Abra o notebook a partir da raiz do repositório, junto à pasta dados.')
OUT = ROOT / 'resultados'
OUT.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid', context='notebook', palette='colorblind')
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 160, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold'})
pd.set_option('display.max_columns', 15)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')
def figura(nome):
    plt.tight_layout()
    plt.savefig(OUT / f'{nome}.png', bbox_inches='tight')
    plt.show()
print({'Python': platform.python_version(), 'pandas': pd.__version__,
       'numpy': np.__version__, 'scikit-learn': sklearn.__version__, 'semente': SEED})
''')
md('''
## 1. Base, dicionário e auditoria

| Coluna original | Papel | Unidade declarada no enunciado |
|---|---|---|
| Crop | Entrada categórica | Cultura |
| Precipitation (mm day-1) | Entrada numérica | mm/dia |
| Specific Humidity at 2 Meters (g/kg) | Entrada numérica | g/kg |
| Relative Humidity at 2 Meters (%) | Entrada numérica | % |
| Temperature at 2 Meters (C) | Entrada numérica | °C |
| Yield | Alvo da regressão | toneladas/hectare |

**Ressalva sobre unidades:** os valores de precipitação (aproximadamente 1.935 a 3.086)
e de rendimento (5.249 a 203.399) são incompatíveis com uma leitura agronômica literal usual
das unidades declaradas. O CSV não fornece documentação para confirmar se há agregação temporal
ou outra unidade de origem. Não suponho conversões. As métricas e gráficos de rendimento usam
**unidades originais de Yield (u.o.)**, sem afirmar que sejam toneladas/ha. Essa incerteza impede
usar os resultados diretamente para orçamento de produção em uma fazenda de 200 hectares.

Apesar de o contexto mencionar solo, não existem pH, nutrientes, localização, ano ou identificador
de fazenda nesta base. Não é possível avaliar esses efeitos nem inventar essas informações.
''')
code('''
path = ROOT / 'dados/crop_yield.csv'
raw = pd.read_csv(path)
expected = ['Crop', 'Precipitation (mm day-1)', 'Specific Humidity at 2 Meters (g/kg)',
            'Relative Humidity at 2 Meters (%)', 'Temperature at 2 Meters (C)', 'Yield']
assert raw.columns.tolist() == expected, 'O esquema do CSV difere da base esperada.'
assert len(raw) > 0, 'A base está vazia.'
assert not raw.isna().any().any(), 'Há valores ausentes: revisar antes de modelar.'
assert not raw.duplicated().any(), 'Há duplicatas exatas: investigar a origem antes de prosseguir.'
assert np.isfinite(raw[expected[1:]].to_numpy(dtype=float)).all(), 'Há números não finitos.'
assert raw['Crop'].str.strip().ne('').all(), 'Há culturas vazias.'
assert raw['Relative Humidity at 2 Meters (%)'].between(0, 100).all()
assert raw[['Precipitation (mm day-1)', 'Specific Humidity at 2 Meters (g/kg)', 'Yield']].ge(0).all().all()

rename = dict(zip(expected, ['cultura', 'precipitacao', 'umidade_especifica',
                             'umidade_relativa', 'temperatura', 'rendimento']))
df = raw.rename(columns=rename)
clima = ['precipitacao', 'umidade_especifica', 'umidade_relativa', 'temperatura']
# A chave representa igualdade das quatro condições, sem inferir que seja um ano.
df['cenario'] = pd.factorize(pd.MultiIndex.from_frame(df[clima]), sort=True)[0]
assert df.groupby('cenario').size().eq(4).all()
assert df.groupby('cenario')['cultura'].nunique().eq(4).all()
sha = hashlib.sha256(path.read_bytes()).hexdigest()
print(f'{len(df)} registros | {df.cultura.nunique()} culturas | {df.cenario.nunique()} cenários')
print('SHA-256 da base original:', sha)
display(raw.head())
display(pd.DataFrame({'tipo': raw.dtypes.astype(str), 'ausentes': raw.isna().sum(),
                      'valores_distintos': raw.nunique()}))
display(df.groupby('cultura').size().rename('registros').to_frame())
''')
md('''
## 2. Protocolo definido antes da análise dos resultados

Cada um dos 39 cenários aparece em quatro culturas. Uma divisão aleatória por linha pode colocar
condições idênticas nos dois conjuntos. Para avaliar condições ainda não vistas, uso uma divisão
por **grupo climático**, com cerca de 20% dos grupos no teste e semente fixa 42.

Somente o treino participa da exploração detalhada e da seleção dos modelos. A validação cruzada
usa **GroupKFold com cinco folds**, sempre mantendo o cenário inteiro no mesmo fold. Padronização,
codificação e transformação do alvo são ajustadas dentro de cada pipeline/fold. O teste fica
reservado até a comparação final. A busca é pequena para limitar custo e sobreajuste.

Os clusters e outliers serão uma análise descritiva separada, posterior à avaliação preditiva;
não fornecerão atributos ao modelo nem motivarão exclusões/reajustes com base no teste.
''')
code('''
features = ['cultura'] + clima
X, y, groups = df[features], df['rendimento'], df['cenario']
train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                            random_state=SEED).split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]
assert set(g_train).isdisjoint(set(g_test))
cv = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=g_train))
for tr, va in cv:
    assert set(g_train.iloc[tr]).isdisjoint(set(g_train.iloc[va]))
    assert set(g_train.iloc[tr]).isdisjoint(set(g_test))
split = df[['cenario', 'cultura']].copy()
split['conjunto'] = np.where(split.index.isin(train_idx), 'treino', 'teste')
split.to_csv(OUT / 'divisao_dados.csv', index_label='linha_csv_zero_based')
display(split.groupby('conjunto').agg(registros=('cenario', 'size'), cenarios=('cenario', 'nunique')))
print('Verificado: nenhum cenário compartilhado entre treino, validação e teste.')
''')
md('''
## 3. Análise exploratória do treino

As estatísticas por cultura evitam tratar a escala elevada do dendê como uma anomalia. As
correlações gerais também podem esconder diferenças entre culturas; por isso apresento
correlações com Yield separadamente. Correlação mede associação, sem provar efeito causal.
''')
code('''
eda = df.iloc[train_idx].copy()
display(eda.groupby('cultura')['rendimento'].describe())
display(eda[clima].drop_duplicates().describe())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=eda, x='cultura', y='rendimento', ax=axes[0])
sns.stripplot(data=eda, x='cultura', y='rendimento', ax=axes[0], color='#25334a', size=3, alpha=.5)
axes[0].set(title='Escalas de rendimento por cultura · treino', ylabel='Yield (u.o.)', xlabel='')
axes[0].tick_params(axis='x', rotation=20)
sns.heatmap(eda[clima].drop_duplicates().corr(), annot=True, fmt='.2f', cmap='vlag',
            center=0, vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title('Correlação climática · cenários de treino')
figura('exploracao')
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for coluna, ax in zip(clima, axes.flat):
    sns.histplot(eda[clima].drop_duplicates()[coluna], bins=8, ax=ax, color='#157a6e')
    ax.set(title=coluna.replace('_', ' ').capitalize(), ylabel='Cenários')
figura('distribuicoes_climaticas')
corr_cultura = pd.DataFrame({c: parte[clima + ['rendimento']].corr()['rendimento'].drop('rendimento')
                             for c, parte in eda.groupby('cultura')})
sns.heatmap(corr_cultura, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, center=0)
plt.title('Associação entre clima e Yield dentro de cada cultura · treino')
figura('correlacoes_por_cultura')
# Relações visuais por cultura, sem misturar escalas no eixo do rendimento.
g = sns.relplot(data=eda, x='precipitacao', y='rendimento', col='cultura', col_wrap=2,
                hue='cultura', facet_kws={'sharey': False}, height=3, legend=False)
g.set_axis_labels('Precipitação (valores originais)', 'Yield (u.o.)')
figura('precipitacao_rendimento')
''')
md('''
### Leitura da exploração

Os registros são balanceados por cultura. Dendê (`Oil palm fruit`) ocupa uma escala de rendimento
muito superior às demais; cacau e borracha têm escalas próximas. Assim, parte importante de um
R² global alto pode vir apenas da identificação da cultura. A base inclui uma faixa estreita de
temperatura e umidade, limitando qualquer extrapolação para clima frio ou seco.

Não existem valores ausentes ou duplicatas exatas nesta versão, portanto não é necessário
imputar nem remover registros. A repetição do clima entre culturas é uma estrutura da amostra,
não uma duplicação de linhas. Os gráficos e a tabela acima permitem observar a direção das
associações por cultura sem confundi-las com evidência causal.
''')
md('''
## 4. Cinco algoritmos e referências simples

- **Regressão Linear:** referência interpretável e aditiva; não captura interações não lineares por si só.
- **Árvore de Decisão:** regras não lineares; profundidade limitada para reduzir sobreajuste.
- **Random Forest:** média de árvores, diminuindo variância; pode continuar limitada fora do domínio observado.
- **Gradient Boosting:** árvores sequenciais que corrigem erros; usa regularização por profundidade e taxa de aprendizado.
- **SVR com kernel RBF:** relações não lineares; entradas e alvo padronizados, com inversão automática para as unidades originais.

As referências são **média global** e **média por cultura**. A segunda responde se o clima agrega
valor além de conhecer a plantação. A média por cultura é implementada por regressão linear
sobre one-hot sem intercepto, equivalente à média de treino de cada categoria.

O critério de seleção é **menor RMSE médio na validação**. MAE expressa o erro absoluto típico;
RMSE penaliza mais os erros grandes; R² compara com uma previsão constante e pode ser negativo.
Todos os erros são calculados na escala original. Os folds não são amostras independentes de
uma população ampla; o desvio entre folds é apenas uma medida descritiva de estabilidade.
''')
code('''
def pipeline(modelo):
    prep = ColumnTransformer([
        ('clima', StandardScaler(), clima),
        ('cultura', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['cultura'])
    ])
    return Pipeline([('preparo', prep),
                     ('modelo', TransformedTargetRegressor(regressor=modelo,
                                                          transformer=StandardScaler()))])

candidatos = {
    'Regressão Linear': (LinearRegression(), {}),
    'Árvore de Decisão': (DecisionTreeRegressor(random_state=SEED),
                         {'max_depth': [2, 3, 5], 'min_samples_leaf': [2, 4]}),
    'Random Forest': (RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
                      {'max_depth': [3, None], 'min_samples_leaf': [2, 4]}),
    'Gradient Boosting': (GradientBoostingRegressor(random_state=SEED, n_estimators=100),
                          {'max_depth': [1, 2], 'learning_rate': [0.03, 0.1]}),
    'SVR (RBF)': (SVR(kernel='rbf'), {'C': [1, 10, 100], 'epsilon': [0.01, 0.1],
                                    'gamma': ['scale', 0.1]})
}
scoring = {'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}
modelos, linhas_cv, buscas = {}, [], {}
for nome, (estimador, grade) in candidatos.items():
    inicio = time.perf_counter()
    busca = GridSearchCV(pipeline(estimador),
                        {f'modelo__regressor__{k}': v for k, v in grade.items()},
                        scoring=scoring, refit='rmse', cv=cv, n_jobs=1, error_score='raise')
    busca.fit(X_train, y_train)
    j = busca.best_index_
    modelos[nome], buscas[nome] = busca.best_estimator_, busca
    linhas_cv.append({'modelo': nome, 'RMSE_CV': -busca.cv_results_['mean_test_rmse'][j],
                     'desvio_RMSE_CV': busca.cv_results_['std_test_rmse'][j],
                     'MAE_CV': -busca.cv_results_['mean_test_mae'][j],
                     'R2_CV': busca.cv_results_['mean_test_r2'][j],
                     'parametros': json.dumps(busca.best_params_, ensure_ascii=False)})
    print(f'{nome}: concluído em {time.perf_counter()-inicio:.1f}s')

baselines = {
    'Média global': DummyRegressor(strategy='mean'),
    'Média por cultura': Pipeline([
        ('cultura', ColumnTransformer([('onehot', OneHotEncoder(handle_unknown='ignore',
                                                               sparse_output=False), ['cultura'])])),
        ('media', LinearRegression(fit_intercept=False))])
}
for nome, base in baselines.items():
    scores = cross_validate(base, X_train, y_train, cv=cv, scoring=scoring, error_score='raise')
    modelos[nome] = clone(base).fit(X_train, y_train)
    linhas_cv.append({'modelo': nome, 'RMSE_CV': -scores['test_rmse'].mean(),
                     'desvio_RMSE_CV': scores['test_rmse'].std(),
                     'MAE_CV': -scores['test_mae'].mean(), 'R2_CV': scores['test_r2'].mean(),
                     'parametros': 'sem busca'})
cv_resultados = pd.DataFrame(linhas_cv).sort_values('RMSE_CV').reset_index(drop=True)
# Escolher e congelar o vencedor entre os cinco algoritmos ANTES de consultar o teste.
vencedor = cv_resultados[cv_resultados.modelo.isin(candidatos)].iloc[0]['modelo']
cv_resultados.to_csv(OUT / 'metricas_validacao.csv', index=False)
display(cv_resultados)
print('Vencedor por validação (congelado):', vencedor)
''')
md('''
## 5. Avaliação única no teste reservado

Os cinco algoritmos e as referências são avaliados nas mesmas 32 observações, de oito cenários
inteiramente reservados. A ordem do teste não altera o vencedor selecionado. A busca de
hiperparâmetros torna a validação otimista em algum grau; o teste independente permite verificar
esse efeito, mas oito cenários ainda geram alta incerteza. Não prometo desempenho em outras
fazendas, anos ou culturas não observadas.
''')
code('''
def metricas(real, previsto):
    return {'MAE': mean_absolute_error(real, previsto),
            'RMSE': root_mean_squared_error(real, previsto), 'R2': r2_score(real, previsto)}

predicoes = {nome: estimador.predict(X_test) for nome, estimador in modelos.items()}
teste_resultados = pd.DataFrame([{'modelo': nome, **metricas(y_test, pred)}
                                  for nome, pred in predicoes.items()]).sort_values('RMSE')
por_cultura = []
for nome, pred in predicoes.items():
    for cultura in sorted(X_test.cultura.unique()):
        mask = X_test.cultura.eq(cultura).to_numpy()
        por_cultura.append({'modelo': nome, 'cultura': cultura, 'n': int(mask.sum()),
                           **metricas(y_test.to_numpy()[mask], pred[mask])})
por_cultura = pd.DataFrame(por_cultura)
teste_resultados.to_csv(OUT / 'metricas_teste.csv', index=False)
por_cultura.to_csv(OUT / 'metricas_por_cultura.csv', index=False)
pred_df = df.iloc[test_idx][['cenario', 'cultura', 'rendimento']].copy()
pred_df['previsao'] = predicoes[vencedor]
pred_df['residuo'] = pred_df.rendimento - pred_df.previsao
pred_df.to_csv(OUT / 'predicoes_teste.csv', index_label='linha_csv_zero_based')
display(teste_resultados)
display(por_cultura[por_cultura.modelo.isin([vencedor, 'Média por cultura'])])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
a = cv_resultados.set_index('modelo').loc[list(candidatos) + list(baselines)]
axes[0].barh(a.index, a.RMSE_CV, xerr=a.desvio_RMSE_CV, color='#157a6e', capsize=3)
axes[0].set(title='Seleção pela validação', xlabel='RMSE médio ± desvio entre folds (u.o.)')
b = teste_resultados.set_index('modelo').loc[a.index]
axes[1].barh(b.index, b.RMSE, color='#314764')
axes[1].set(title='Auditoria no teste reservado', xlabel='RMSE de teste (u.o.)')
figura('comparacao_modelos')
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for j, (cultura, parte) in enumerate(pred_df.groupby('cultura')):
    ax = axes[0, j]
    ax.scatter(parte.rendimento, parte.previsao, color='#157a6e')
    limites = [min(parte.rendimento.min(), parte.previsao.min()),
               max(parte.rendimento.max(), parte.previsao.max())]
    ax.plot(limites, limites, '--', color='#68778c')
    ax.set(title=cultura, xlabel='Real (u.o.)', ylabel='Previsto (u.o.)')
    axes[1, j].scatter(parte.previsao, parte.residuo, color='#b66a2c')
    axes[1, j].axhline(0, linestyle='--', color='#68778c')
    axes[1, j].set(xlabel='Previsto (u.o.)', ylabel='Real − previsto (u.o.)')
fig.suptitle(f'{vencedor} · teste por cultura (escalas próprias)', y=1.02)
figura('diagnostico_teste')
''')
md('''
## 6. Clusterização: regimes climáticos e produtividade

Esta etapa descritiva usa todos os **39 cenários únicos**, sem repetir cada cenário quatro vezes.
As quatro entradas climáticas são padronizadas e agrupadas por K-Means. Testo k de 2 a 6 e
seleciono o maior silhouette (empate: menor k). O rendimento **não entra na distância**; é usado
somente depois para caracterizar a produtividade dentro de cada cultura e grupo.

Uma PCA de duas dimensões serve apenas para visualizar os grupos calculados nas quatro
dimensões. Não é uma regressão nem prova de separação perfeita. A estabilidade é inspecionada
com cinco sementes adicionais. Silhouette e estabilidade não validam causalidade ou utilidade
agronômica por si sós.
''')
code('''
cenarios = df[['cenario'] + clima].drop_duplicates().sort_values('cenario').set_index('cenario')
scaler_cluster = StandardScaler()
Z = scaler_cluster.fit_transform(cenarios)
ks = pd.DataFrame([{'k': k, 'silhouette': silhouette_score(Z, KMeans(n_clusters=k,
                      n_init=30, random_state=SEED).fit_predict(Z))} for k in range(2, 7)])
best_k = int(ks.sort_values(['silhouette', 'k'], ascending=[False, True]).iloc[0]['k'])
kmeans = KMeans(n_clusters=best_k, n_init=30, random_state=SEED).fit(Z)
cenarios['cluster'] = kmeans.labels_ + 1
analise = df.merge(cenarios[['cluster']], left_on='cenario', right_index=True, validate='many_to_one')
perfil = cenarios.groupby('cluster')[clima].mean()
perfil.insert(0, 'n_cenarios', cenarios.groupby('cluster').size())
produtividade = analise.groupby(['cultura', 'cluster']).rendimento.agg(['count', 'mean', 'median', 'min', 'max'])
from sklearn.metrics import adjusted_rand_score
estabilidade = [adjusted_rand_score(kmeans.labels_, KMeans(n_clusters=best_k, n_init=30,
                random_state=s).fit_predict(Z)) for s in [0, 1, 7, 21, 99]]
display(ks)
display(perfil)
display(produtividade)
print('ARI entre sementes (1 = agrupamento equivalente):', estabilidade)
ks.to_csv(OUT / 'selecao_clusters.csv', index=False)
perfil.to_csv(OUT / 'perfil_clusters.csv')
produtividade.to_csv(OUT / 'produtividade_clusters.csv')
analise.to_csv(OUT / 'dados_com_clusters.csv', index=False)
pca = PCA(n_components=2).fit(Z)
proj = pca.transform(Z)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(ks.k, ks.silhouette, marker='o', color='#157a6e')
axes[0].axvline(best_k, linestyle='--', color='#b66a2c')
axes[0].set(title=f'Seleção de k: {best_k} grupos', xlabel='k', ylabel='Silhouette')
sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=cenarios.cluster.astype(str), s=85, ax=axes[1])
axes[1].set(title=f'PCA: {pca.explained_variance_ratio_.sum():.1%} da variância',
            xlabel='Componente 1', ylabel='Componente 2')
axes[1].legend(title='Cluster')
figura('clusters')
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (cultura, parte) in zip(axes.flat, analise.groupby('cultura')):
    sns.boxplot(data=parte, x='cluster', y='rendimento', ax=ax, color='#91c7b1')
    sns.stripplot(data=parte, x='cluster', y='rendimento', ax=ax, color='#25334a', size=4)
    ax.set(title=cultura, ylabel='Yield (u.o.)', xlabel='Cluster climático')
figura('produtividade_clusters')
''')
md('''
## 7. Cenários discrepantes: outliers de rendimento

Um rendimento de dendê não é comparado diretamente ao de cacau para definir anomalias. Em cada
cultura, sinalizo valores abaixo de Q1 − 1,5×IQR ou acima de Q3 + 1,5×IQR. São **candidatos a
investigação**, não erros confirmados. A ausência de sinais também é um resultado válido desse
critério, sem significar que todos os registros estejam corretos. Nenhuma linha é excluída e
nenhum modelo é treinado novamente após consultar estes resultados.

O gráfico desta seção usa os 39 registros de cada cultura. Os quartis podem diferir dos
boxplots exploratórios calculados apenas no treino; isso explica eventuais marcações diferentes.
''')
code('''
limites_iqr = df.groupby('cultura').rendimento.quantile([.25, .75]).unstack()
limites_iqr.columns = ['Q1', 'Q3']
limites_iqr['IQR'] = limites_iqr.Q3 - limites_iqr.Q1
limites_iqr['inferior'] = limites_iqr.Q1 - 1.5 * limites_iqr.IQR
limites_iqr['superior'] = limites_iqr.Q3 + 1.5 * limites_iqr.IQR
out = analise.join(limites_iqr[['inferior', 'superior']], on='cultura')
out['outlier'] = (out.rendimento < out.inferior) | (out.rendimento > out.superior)
display(limites_iqr)
display(out.groupby('cultura').outlier.sum().rename('sinalizados').to_frame())
display(out.loc[out.outlier])
out.loc[out.outlier].to_csv(OUT / 'outliers.csv', index=False)
print(f'{out.outlier.sum()} registros sinalizados; 0 registros removidos.')
# Visualizar os limites calculados na base completa, separados por cultura.
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (cultura, parte) in zip(axes.flat, out.groupby('cultura')):
    sns.boxplot(data=parte, y='rendimento', ax=ax, color='#91c7b1', width=.4)
    sns.stripplot(data=parte, y='rendimento', ax=ax, color='#25334a', size=4, jitter=False)
    ax.axhline(limites_iqr.loc[cultura, 'inferior'], color='#b66a2c', linestyle='--', label='Limites 1,5 × IQR')
    ax.axhline(limites_iqr.loc[cultura, 'superior'], color='#b66a2c', linestyle='--')
    ax.set(title=f'{cultura}: {int(parte.outlier.sum())} sinalizados', ylabel='Yield (u.o.)', xlabel='')
    ax.legend(fontsize=8)
figura('outliers_por_cultura')
''')
md('''
## 8. Exemplo de previsão e uso responsável

A entrada é um DataFrame com a cultura conhecida e as quatro condições na **mesma escala do CSV**.
A saída é um vetor de rendimentos nas unidades originais. O exemplo usa uma condição do teste
já avaliado, não cria uma nova evidência de generalização. Culturas desconhecidas, números não
finitos e extrapolações além do intervalo observado no treino são recusados neste demonstrador.
O modelo permanece ajustado apenas ao treino para preservar a rastreabilidade da avaliação.
''')
code('''
def prever(entrada):
    if not isinstance(entrada, pd.DataFrame) or entrada.empty:
        raise ValueError('Forneça um DataFrame não vazio.')
    if set(entrada.columns) != set(features):
        raise ValueError(f'Colunas necessárias: {features}')
    if not entrada.cultura.isin(X_train.cultura.unique()).all():
        raise ValueError('Cultura desconhecida: use uma das quatro culturas da base.')
    try:
        numeros = entrada[clima].astype(float)
    except (TypeError, ValueError) as erro:
        raise ValueError('As condições climáticas devem ser numéricas.') from erro
    if not np.isfinite(numeros.to_numpy()).all():
        raise ValueError('Condições ausentes ou não finitas.')
    if ((numeros < X_train[clima].min()) | (numeros > X_train[clima].max())).any().any():
        raise ValueError('Entrada fora dos limites de treino; extrapolação não autorizada.')
    valida = entrada.copy()
    valida[clima] = numeros
    return modelos[vencedor].predict(valida[features])

# Encontrar um exemplo de teste dentro dos limites de entrada observados no treino.
dentro = ((X_test[clima] >= X_train[clima].min()) &
          (X_test[clima] <= X_train[clima].max())).all(axis=1)
exemplo = X_test.loc[dentro].iloc[[0]].copy()
display(exemplo)
print('Previsão (u.o.):', prever(exemplo).round(2))
print('Valor observado (u.o.):', y_test.loc[exemplo.index].to_numpy())
# Guardas significativas da interface de demonstração.
for invalida in [exemplo.assign(cultura='cultura desconhecida'),
                 exemplo.assign(temperatura=np.nan), exemplo.assign(temperatura=1000),
                 exemplo.iloc[:0]]:
    try:
        prever(invalida)
    except ValueError:
        pass
    else:
        raise AssertionError('Uma entrada inválida foi aceita.')

resumo = {'autor': 'Eduardo Rick', 'rm': '573745', 'sha256_csv': sha,
          'n_registros': len(df), 'n_cenarios': df.cenario.nunique(),
          'n_treino': len(train_idx), 'n_teste': len(test_idx), 'vencedor_cv': vencedor,
          'teste_vencedor': metricas(y_test, predicoes[vencedor]),
          'teste_media_cultura': metricas(y_test, predicoes['Média por cultura']),
          'k': best_k, 'silhouette': float(ks.loc[ks.k.eq(best_k), 'silhouette'].iloc[0]),
          'outliers': int(out.outlier.sum()), 'ari_sementes': estabilidade}
(OUT / 'resumo.json').write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
print('Verificações de dados, separação por grupos e entradas inválidas: OK.')
''')
md('''
## 9. Conclusões, pontos fortes e limitações

<!-- RESULTADOS_FINAIS -->

**Pontos fortes:** dados originais preservados; auditoria executável; comparação de cinco famílias
de modelos; referências global e por cultura; separação por cenário; pré-processamento dentro dos
folds; análise de erros por cultura; clusters sem usar o alvo e tratamento transparente de outliers.

**Limitações:** apenas 39 cenários climáticos e oito no teste; unidades não confirmadas; ausência
de datas, localização e variáveis do solo; restrição a quatro culturas; faixa climática estreita.
Mesmo quando o R² agregado é alto, isso não significa alta precisão em cada cultura. Escolher o
menor RMSE de validação em escala original também atribui maior influência à cultura de maior
escala; a análise por cultura é essencial para interpretar essa escolha.

Os clusters descrevem **associações de produtividade**, não uma evolução ao longo do tempo ou
uma receita de irrigação. A validação em novas safras exigiria dados datados e independentes,
verificação das unidades e avaliação agronômica. Antes de uso real, também seriam necessários
intervalos de previsão calibrados, monitoramento de mudança dos dados e teste de memória/latência
na máquina de nuvem escolhida. Não há implantação em produção nesta atividade.

### Referências metodológicas
- [scikit-learn: validação cruzada e divisões por grupos](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: prevenção de vazamento e pipelines](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: métricas de regressão](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [scikit-learn: K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- Fonte dos dados e requisitos: arquivo e enunciado disponibilizados no portal FIAP, Fase 5.
''')
nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name':'python','version':'3.12.12'}})
nbf.write(nb, ROOT / 'EduardoRick_rm573745_pbl_fase4.ipynb')
print('Notebook gerado:', len(cells), 'células')
