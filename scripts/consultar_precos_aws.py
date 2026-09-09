"""Consulta pública complementar ao uso da calculadora; não requer conta AWS."""
import csv
import io
import json
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[1]
def consultar(regiao):
    url = f'https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/{regiao}/index.csv'
    selecionadas = []
    with urllib.request.urlopen(url, timeout=90) as response:
        texto = io.TextIOWrapper(response, encoding='utf-8')
        for linha in texto:
            if linha.startswith('"SKU"'):
                cabecalho = next(csv.reader([linha])); break
        for r in csv.DictReader(texto, fieldnames=cabecalho):
            if r.get('TermType') != 'OnDemand': continue
            uso = r.get('usageType', '')
            if 'EBS:VolumeIOUsage' in uso or uso.endswith('EBS:VolumeUsage') or uso.endswith('EBS:VolumeUsage.gp3'):
                selecionadas.append(r)
    resultado = {'fonte':url, 'regiao':regiao, 'linhas':selecionadas}
    (ROOT / f'docs/aws/precos-oficiais-{regiao}.json').write_text(json.dumps(resultado,ensure_ascii=False,indent=2))
    print(regiao,[(r.get('usageType'),r.get('PricePerUnit'),r.get('Unit')) for r in selecionadas],flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(consultar,['us-east-1','sa-east-1']))
