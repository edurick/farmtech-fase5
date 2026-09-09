"""Confere integridade local. --final exige links de YouTube já preenchidos."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import nbformat
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
args = argparse.ArgumentParser()
args.add_argument('--final', action='store_true')
final = args.parse_args().final
nbpath=ROOT/'EduardoRick_rm573745_pbl_fase4.ipynb'
nb=nbformat.read(nbpath,as_version=4); nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=='code']
assert code and all(c.execution_count is not None for c in code)
assert not [o for c in code for o in c.outputs if o.output_type=='error']
resumo=json.loads((ROOT/'resultados/resumo.json').read_text())
assert hashlib.sha256((ROOT/'dados/crop_yield.csv').read_bytes()).hexdigest()==resumo['sha256_csv']
split=pd.read_csv(ROOT/'resultados/divisao_dados.csv')
assert split.groupby('cenario').conjunto.nunique().eq(1).all()
assert split.groupby('conjunto').size().to_dict()=={'teste':32,'treino':124}
cv=pd.read_csv(ROOT/'resultados/metricas_validacao.csv')
test=pd.read_csv(ROOT/'resultados/metricas_teste.csv')
algoritmos={'Regressão Linear','Árvore de Decisão','Random Forest','Gradient Boosting','SVR (RBF)'}
assert algoritmos.issubset(set(cv.modelo)) and algoritmos.issubset(set(test.modelo))
assert cv[cv.modelo.isin(algoritmos)].sort_values('RMSE_CV').iloc[0].modelo==resumo['vencedor_cv']
assert (ROOT/'resultados/predicoes_teste.csv').exists()
custos=pd.read_csv(ROOT/'docs/aws/comparacao_custos.csv')
assert len(custos)==12
for _,row in custos.iterrows():
    assert abs(row.total_mensal_exato-(730*row.usd_hora+row.ebs_mensal))<1e-8
    assert abs(row.total_anual_calculadora-12*row.total_mensal_calculadora)<1e-8
for reg,arquivo,total in [('Virgínia do Norte','virginia-magnetic-calculo.txt',8.63),
                          ('São Paulo','sao-paulo-magnetic-calculo.txt',15.78),
                          ('Virgínia do Norte','virginia-gp3-calculo.txt',10.13),
                          ('São Paulo','sao-paulo-gp3-calculo.txt',17.38)]:
    conteudo=(ROOT/'docs/aws'/arquivo).read_text()
    assert f'{total:.2f} USD' in conteudo
    assert '100' in conteudo and '50' in conteudo and 'On-Demand' in conteudo
for doc in [ROOT/'README.md']:
    for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',doc.read_text()):
        if '://' not in link and not link.startswith('#'):
            assert (doc.parent/link.split('#')[0]).exists(),f'Link local quebrado: {doc.name}: {link}'
readme=(ROOT/'README.md').read_text()
pendente='PENDENTE: inserir link' in readme
if final:
    assert not pendente, 'Faltam os links dos dois vídeos no README.'
    videos=set(re.findall(r'https://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]+|youtu\.be/[\w-]+)',readme))
    assert len(videos)>=2, 'São necessários dois links distintos do YouTube no README.'
print(f'OK: {len(code)} células executadas, base íntegra, divisão por grupos, cinco modelos, 12 cenários de custo e links locais.')
print('Vídeos: '+('PENDENTES — gravar, publicar e preencher README.' if pendente else 'links preenchidos; confirmar reprodução em janela anônima.'))
print('Publicação GitHub e envio no portal devem ser confirmados externamente.')
