"""Export editable LaTeX source to PDF and previews with draw.io + Poppler.
Each page is rendered in isolation to avoid draw.io's multi-page MathJax bounds bug.
No dependencies are installed. roadmap.drawio remains the editable source.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
import subprocess
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parent
pages=list(ET.parse(R/'roadmap.drawio').getroot())
names=['01-baselines','02-components','03-paper-map','04-length','05-chunk-stopd']+[f'{n:02d}-paper-formulas' for n in range(6,len(pages))]+[f'{len(pages):02d}-signal-comparison']
def run(cmd):
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr[-1500:])
with TemporaryDirectory(prefix='opd-math-export-') as temp:
    temp=Path(temp)
    root=ET.parse(R/'roadmap.drawio').getroot()
    def export(pair):
        n,diagram=pair;name=names[n]
        single=ET.Element('mxfile',host='drawio');single.append(diagram)
        source=temp/(name+'.drawio');pdf=temp/(name+'.pdf')
        ET.ElementTree(single).write(source,encoding='utf-8',xml_declaration=True)
        run(['drawio','-x','-f','pdf','-o',str(pdf),str(source)])
        run(['pdftoppm','-scale-to','1800','-png','-singlefile',str(pdf),str(R/name)])
        print(f'Exported {n+1}/{len(names)}: {name}',flush=True)
        return pdf
    with ThreadPoolExecutor(max_workers=1) as pool:
        pdfs=list(pool.map(export,enumerate(list(root))))
    run(['pdfunite',*[str(p) for p in pdfs],str(R/'roadmap.pdf')])
print(f'Exported {len(pages)}-page roadmap.pdf and all previews; editable source: roadmap.drawio',flush=True)
