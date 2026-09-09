"""Executa em kernel limpo, verifica saídas e exporta uma cópia HTML para leitura."""
from pathlib import Path
import json
import os
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/farmtech-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/tmp/farmtech-jupyter')
path = ROOT / 'EduardoRick_rm573745_pbl_fase4.ipynb'
nb = nbformat.read(path, as_version=4)
# O kernelspec padrão do ambiente usa o Python que executa este script.
client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
client.execute()
errors = [o for c in nb.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
assert not errors
assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
r = json.loads((ROOT / 'resultados/resumo.json').read_text())
import pandas as pd
pc = pd.read_csv(ROOT / 'resultados/metricas_por_cultura.csv')
cv = pd.read_csv(ROOT / 'resultados/metricas_validacao.csv')
perf = pd.read_csv(ROOT / 'resultados/perfil_clusters.csv')
prod = pd.read_csv(ROOT / 'resultados/produtividade_clusters.csv')
texto = f"""**Resultados executados:** {r['vencedor_cv']} foi selecionado pela validação. No teste,
obteve MAE **{r['teste_vencedor']['MAE']:,.2f}**, RMSE **{r['teste_vencedor']['RMSE']:,.2f}** (u.o.)
e R² **{r['teste_vencedor']['R2']:.4f}**. A referência pela média de cada cultura obteve
RMSE **{r['teste_media_cultura']['RMSE']:,.2f}** e R² **{r['teste_media_cultura']['R2']:.4f}**.
A diferença de RMSE contra essa referência é
**{(1-r['teste_vencedor']['RMSE']/r['teste_media_cultura']['RMSE'])*100:.2f}%**
(valores negativos significam piora). O resultado evidencia por que a média por cultura é um
controle mais exigente que a média global.

**Desempenho dentro de cada cultura:**\n\n"""
texto += pc[pc.modelo.eq(r['vencedor_cv'])][['cultura','n','MAE','RMSE','R2']].to_markdown(index=False, floatfmt='.3f')
texto += '\n\nR² negativo significa desempenho inferior à média observada daquele subconjunto de teste; '
texto += 'não deve ser ocultado pelo R² agregado. A comparação operacional com uma média disponível '
texto += 'no treino é fornecida pela referência por cultura na seção 5.\n\n'
texto += f"**Grupos:** foram selecionados **{r['k']} clusters**, silhouette **{r['silhouette']:.3f}**. "
texto += f"A estabilidade entre sementes apresentou ARI mínimo **{min(r['ari_sementes']):.3f}**. "
texto += 'Os perfis abaixo caracterizam os agrupamentos nas unidades originais:\n\n'
texto += perf.to_markdown(index=False, floatfmt='.2f')
texto += '\n\n**Associações de produtividade:**\n\n'
for crop, part in prod.groupby('cultura'):
    hi = part.loc[part['mean'].idxmax()]; lo = part.loc[part['mean'].idxmin()]
    texto += f"- {crop}: maior média no cluster {int(hi['cluster'])} ({hi['mean']:,.2f} u.o.); menor no cluster {int(lo['cluster'])} ({lo['mean']:,.2f} u.o.).\n"
texto += '\nEssas diferenças descrevem a amostra e não estimam o efeito de mudar o clima. '
texto += 'O tamanho de cada grupo e a sobreposição nos boxplots devem acompanhar a leitura das médias.\n\n'
texto += f"**Outliers:** {r['outliers']} registro(s) sinalizado(s) pelo IQR por cultura; nenhum removido. "
if r['outliers']:
    out = pd.read_csv(ROOT / 'resultados/outliers.csv')
    texto += 'Os registros devem ser investigados com a fonte; a base não permite classificá-los como erros.\n\n'
    texto += out[['cultura','cenario','rendimento','cluster']].to_markdown(index=False)
else:
    texto += 'Isso indica apenas ausência de sinais pelo critério adotado, sem garantir ausência de anomalias.'
for cell in nb.cells:
    if cell.cell_type == 'markdown' and '<!-- RESULTADOS_FINAIS -->' in cell.source:
        prefix, suffix = cell.source.split('<!-- RESULTADOS_FINAIS -->', 1)
        if '<!-- FIM_RESULTADOS -->' in suffix:
            suffix = suffix.split('<!-- FIM_RESULTADOS -->', 1)[1]
        cell.source = prefix + '<!-- RESULTADOS_FINAIS -->\n\n' + texto + '\n\n<!-- FIM_RESULTADOS -->' + suffix
nbformat.validate(nb)
nbformat.write(nb, path)
html, _ = HTMLExporter(template_name='lab').from_notebook_node(nb)
(ROOT / 'docs/notebook.html').write_text(html, encoding='utf-8')
relatorio = {'celulas_codigo': sum(c.cell_type=='code' for c in nb.cells),
             'celulas_executadas': sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),
             'erros': len(errors), 'validacao_nbformat': True, 'python': sys.version,
             'verificacoes_embutidas': 'dados, grupos disjuntos, previsão e entradas inválidas'}
(ROOT / 'resultados/verificacao.json').write_text(json.dumps(relatorio, ensure_ascii=False, indent=2))
print(json.dumps(r, ensure_ascii=False, indent=2))
print('Notebook executado e HTML exportado com sucesso.')
