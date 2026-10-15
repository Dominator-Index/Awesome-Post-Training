"""Regenerate editable diagrams and offline reading cards from the CSV catalog.
No network, dependency installation, or training is performed.
"""
from pathlib import Path
import json, csv, html, xml.etree.ElementTree as ET
R=Path(__file__).resolve().parent
# CSV is published; optional JSON metadata stays local.
COM = {'D': '题目与训练数据', 'H': 'rollout 与前缀分布', 'T': 'teacher 来源与额外信息', 'I': '监督接口与动作单位', 'F': '散度、估计器与梯度路径', 'A': '奖励、return、baseline 与 critic', 'W': 'token / trajectory 权重与门控', 'N': 'advantage / loss 归一化', 'C': '约束、更新幅度与训练调度', 'L': 'token length、截断与预算'}
FIELDS = {'name': '论文', 'high_level': '一句话认知', 'method': '方法',
          'formula': '关键公式', 'formula_kind': '公式性质', 'problem': '问题',
          'boundary': '适用边界', 'evidence': '证据', 'evidence_location': '定位',
          'path': '目录', 'source_url': '来源'}
with (R/'论文组件矩阵.csv').open(encoding='utf-8', newline='') as catalog:
 P = [{**{key: row[column] for key, column in FIELDS.items()},
       'components': row['修改组件'].split()} for row in csv.DictReader(catalog)]
DATA = {'components': COM, 'papers': P}
COL={'D':('#dae8fc','#6c8ebf'),'H':('#dae8fc','#6c8ebf'),'T':('#e1d5e7','#9673a6'),'I':('#e1d5e7','#9673a6'),'F':('#d5e8d4','#82b366'),'A':('#d5e8d4','#82b366'),'W':('#fff2cc','#d6b656'),'N':('#fff2cc','#d6b656'),'C':('#ffe6cc','#d79b00'),'L':('#ffe6cc','#d79b00')}
mx=ET.Element('mxfile',host='drawio',version='30.3.14')
class Page:
 def __init__(self,name,w=1720,h=1180):
  d=ET.SubElement(mx,'diagram',name=name,id='page'+str(len(mx)))
  self.model=ET.SubElement(d,'mxGraphModel',dx=str(w),dy=str(h),grid='1',gridSize='10',page='1',pageScale='1',pageWidth=str(w),pageHeight=str(h),math='1',background='#ffffff')
  self.prefix=d.get('id')+'-';self.root=ET.SubElement(self.model,'root');ET.SubElement(self.root,'mxCell',id='0');ET.SubElement(self.root,'mxCell',id='1',parent='0');self.n=1
 def box(self,label,x,y,w,h,c='D',size=20,style='',link=None,parent='1'):
  self.n+=1;ident=self.prefix+str(self.n);fill,stroke=COL.get(c,('#f5f5f5','#666666'))
  has_math='$$' in label or r'\(' in label
  if has_math:label='<div style="padding:12px;box-sizing:border-box;">'+label+'</div>'
  overflow='overflow=block;' if has_math else ''
  attrs=dict(id=ident,value=label,style=f'rounded=1;whiteSpace=wrap;html=1;{overflow}fillColor={fill};strokeColor={stroke};fontColor=#172f36;fontFamily=Noto Sans CJK SC;fontSize={size};spacing=14;align=left;verticalAlign=middle;'+style,vertex='1',parent=str(parent))
  if link:attrs['link']=link
  e=ET.SubElement(self.root,'mxCell',attrs);ET.SubElement(e,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'});return ident
 def edge(self,a,b,label='',vertical=False,up=False):
  self.n+=1;coords=('exitX=0.5;exitY=0;entryX=0.5;entryY=1;' if up else 'exitX=0.5;exitY=1;entryX=0.5;entryY=0;') if vertical else 'exitX=1;exitY=0.1;entryX=0;entryY=0.1;'
  e=ET.SubElement(self.root,'mxCell',id=self.prefix+str(self.n),value=label,style='edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#536a72;fontSize=16;labelBackgroundColor=#ffffff;'+coords,edge='1',parent='1',source=a,target=b);ET.SubElement(e,'mxGeometry',relative='1',attrib={'as':'geometry'})
 def title(self,title,sub):
  self.box('<b>'+title+'</b><br><font style="font-size:18px">'+sub+'</font>',40,30,1640,100,'gray',28,'fillColor=none;strokeColor=none;')
 def legend(self,y):
  for i,(c,t) in enumerate([('D','数据 / 轨迹'),('T','teacher / 接口'),('F','监督信号 / 估计'),('W','权重 / 归一化'),('C','更新 / 长度')]):self.box(t,40+i*330,y,310,50,c,16)

def math(tex,display=True):
 return ('$$' if display else r'\(')+html.escape(tex)+('$$' if display else r'\)')

p=Page('01｜五种基线');p.title('SFT · PPO · GRPO · OPD · OPSD','先区分轨迹来自谁、监督来自谁，再比较 loss；箭头表示组件变换，不表示唯一训练顺序。')
texts=[('SFT','参考文本 → 明确目标',r'\mathcal L_{\mathrm{SFT}}=-\sum_t\log\pi_\theta(y_t^*\mid x,y_{<t}^*)','强项：稳定、适合冷启动<br>问题：参考前缀与推理前缀不同','D'),('PPO','学生尝试 → reward + critic',r'\widehat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}','强项：任务收益、可 bootstrap<br>问题：critic 偏差与探索成本','A'),('GRPO','同题多条回答互相比较',r'\widehat A_i=\frac{R_i-\bar R}{\sigma_R+\epsilon}','强项：省 learned critic<br>问题：全同奖励为零、信用粗','N'),('OPD','学生前缀 → teacher 分布',r'\mathcal L_{\mathrm{OPD}}=\mathbb E_{y\sim\pi_\theta}\sum_t D(p_t,q_t)','强项：局部反馈密集<br>问题：teacher 偏好未必有益','F'),('OPSD','同模型 + 额外信息作 teacher',r'q_t=\pi_{\bar\theta}(\cdot\mid x,z,y_{<t})','强项：利用自身信息优势<br>问题：z 缺失、风格和探索错配','T')]
ids=[]
for i,(name,idea,f,desc,c) in enumerate(texts):
 x=40+(i%3)*560;y=190+(i//3)*360
 ids.append(p.box(f'<b>{name}</b><br>{idea}<br><br>{math(f)}<br><br>{desc}',x,y,510,290,c,21))
p.edge(ids[1],ids[2]);p.edge(ids[3],ids[4])
p.box('PPO → GRPO：将 value baseline 换成同题组统计',600,495,1070,40,'gray',18,'fillColor=none;strokeColor=none;')
p.box('OPD → OPSD：让同模型 teacher 获得额外参考信息',40,880,1070,55,'gray',18,'fillColor=none;strokeColor=none;')
p.box('<b>一个共同框架</b><br>题目 → 采样状态 → 构造信号 → 权重 / 归一化 → 更新<br><br>改动可以很小，但“目标变了”与“只换估计器”必须区分。',1160,550,510,290,'C',22)
p.legend(1000)

p=Page('02｜十个组件');p.title('训练循环的十个可修改位置','每篇论文可以同时修改多个位置；长度 L 横跨采样 H、信号 A、权重 W 和归一化 N。')
flow=[]
for i,c in enumerate(['D','H','I','F','W','N','C']):
 labels={'D':'\\(x\\sim\\mathcal D\\)<br>题目 / 参考数据','H':'\\(y\\sim\\mu\\)<br>rollout / prefix','I':'token / 文本单位<br>监督对齐接口','F':'KL / MC / sg<br>loss 与梯度路径','W':'\\(m_t w_t\\)<br>筛选 / 加权','N':'\\(\\sigma, Z\\)<br>尺度 / 聚合','C':'clip / anchor<br>控制更新'}
 flow.append(p.box('<b>'+c+'｜'+COM[c]+'</b><br><br>'+labels[c],40+i*240,410,210,190,c,18))
for a,b in zip(flow,flow[1:]):p.edge(a,b)
t=p.box('<b>T｜teacher 与信息</b><br>外部模型 / 自教师 / reference<br>crop / feedback / harness<br><b>\\(q(\\cdot\\mid h,z)\\)</b>',760,180,340,170,'T',21);p.edge(t,flow[3],vertical=True)
a=p.box('<b>A｜reward · baseline · critic</b><br>\\(R\\) / log-ratio / \\(Q-V\\) / return<br>vOPD、AOPD、RLCSD、AC2',760,700,420,180,'A',21);p.edge(a,flow[3],vertical=True,up=True)
l=p.box('<b>L｜长度与预算</b><br>生成多少？监督到哪里？<br>按 token 还是序列平均？<br>完整 / chunk / random stop',280,700,400,180,'L',21);p.edge(flow[1],l,vertical=True)
p.box('示例：DF-OPD 改 D；CorrGRPO 改 N；AC2 改 H/A/L；STOPD 改 H/W/N/L；TOP-D 改 T/F/A/N/C/L。',40,940,1640,70,'gray',21)
p.legend(1060)

p=Page('03｜论文组件地图',1720,2220);p.title('论文在哪里动手：按组件查，而非按新名字记','每张卡是一个可继续展开的组件分支；一篇论文可出现多次。细节与关键公式见认知卡片 / 离线浏览器。')
for i,(c,label) in enumerate(COM.items()):
 names=[r['name'] for r in P if c in r['components'] and r['formula']]
 text='<b>'+c+'｜'+label+'</b><br><br>'+'<br>'.join(' · '.join(names[j:j+2]) for j in range(0,len(names),2))
 p.box(text,40+(i%2)*840,180+(i//2)*350,800,310,c,18)
p.box('<b>正文待核对（只保留题名指向）</b><br>Better Rollouts · CoTrek · Fewer Tokens / Fewer Layers · Early Risk Hindsight<br>REOPOLD 依据同方法 arXiv 版本，原 OpenReview 版细节未逐字核对。',40,1960,1640,150,'gray',20)
p.legend(2140)

p=Page('04｜长度的不同位置',1720,1490);p.title('不要把所有 token length 改动混为一谈','生成截断、loss mask、长度分母、未来 return 和位置权重，作用位置不同。')
rows=[('GRPO outcome','完整回答 → terminal reward','每条序列所有 token 共用 \\(\\widehat A_i\\)；通常需要完整终局结果','A'),('AC2 chunk','replay prefix → 新 chunk → critic V','ready 时不生成后半段；用 \\(V(sc_i)\\) 构造组内优势','A'),('STOPD block','随机停止 → 局部 teacher → \\(1/p_t\\)','不预测未生成未来；IPW 补偿后段 token 的覆盖差异','W'),('Draft-OPD block','draft 提议 → target 验证 → anchor replay','block 用于投机验证与接受长度；不是 value chunk','T')]
for i,(title,flow,desc,c) in enumerate(rows):
 y=180+i*200;p.box('<b>'+title+'</b><br>'+desc,40,y,530,160,c,20)
 prev=None
 for j in range(8):
  fill='D' if j<3 else c
  label='prefix' if j==1 else ('R' if i==0 and j==7 else 'V' if i==1 and j==4 else 'STOP' if i==2 and j==4 else ('✓ / ×' if i==3 else 'token'))
  style='dashed=1;fillColor=#ffffff;strokeColor=#a6b3b7;' if i in [1,2] and j>4 else ''
  p.box(label,620+j*130,y+20,110,70,fill,16,style)
 p.box(flow,620,y+105,1010,50,'gray',19,'fillColor=none;strokeColor=none;')
p.box('<b>loss 长度聚合：三种不同权重</b><br>序列平均：\\(\\frac{1}{B}\\sum_i\\frac{1}{T_i}\\sum_t\\ell_{it}\\)<br>token 平均：\\(\\frac{\\sum_{i,t}\\ell_{it}}{\\sum_i T_i}\\)<br>固定分母：\\(\\frac{\\sum_{i,t}\\ell_{it}}{BL}\\)',40,1010,760,380,'N',21)
p.box('<b>还要分开记录</b><br>训练预算 ≠ 测试预算；已生成后 mask ≠ 提前停止<br>TOP-D：即时信号 + 平均未来 return<br>PW-OPSD：相对位置权重；VLM：输入 vision token 数',850,1010,820,380,'L',22)

p=Page('05｜AC2与STOPD',1720,1570);p.title('都能少做完整 rollout，但学习信号来自不同地方','AC2 用 learned critic 估计未生成未来；STOPD 对已定义的局部监督做随机抽样与 IPW 补偿。')
left=[];right=[]
L=[('恢复同一个 replay prefix s','复用起点，之后的新 chunk 由当前 actor 生成'),('分叉生成多个新 chunk cᵢ','不是从历史轨迹随意截几段直接拟合'),('检查 critic 是否 ready','未 ready / audit → 完整终局 \\(R_i\\)<br>ready 且截断 → \\(V(sc_i)\\)'),('构造 chunk advantage，再 PPO 更新','\\(\\widehat A_i=v_i-\\frac1G\\sum_j v_j\\)<br>主实验不除 σ；chunk 内共享 credit')]
S=[('生成前随机抽停止 block','\\(A\\sim\\operatorname{Unif}\\{0,\\ldots,L-1\\}\\)<br>停止点与轨迹独立'),('仅生成到 stop 或自然 EOS','实际生成 \\(M=\\min(T,S)\\)<br>teacher 对保留前缀提供局部反馈'),('按保留概率的倒数加权','\\(p_t=\\frac{L-K\\lfloor(t-1)/K\\rfloor}{L}\\)<br>晚位置少保留，但权重 \\(1/p_t\\) 更大'),('以 lagged token count 归一化','\\(\\mathbb E_S[\\widehat G]=G,\\quad\\mathbb E_S[\\widehat g]=\\frac NC g^{\\mathrm v}\\)<br>梯度和无偏，归一化仍有尺度差异')]
for i,((t,d),(t2,d2)) in enumerate(zip(L,S)):
 y=190+i*280
 left.append(p.box('<b>AC2 · '+t+'</b><br><br>'+d,40,y,760,240,'A',20))
 right.append(p.box('<b>STOPD · '+t2+'</b><br><br>'+d2,920,y,760,240,'W',20))
for group in [left,right]:
 for a,b in zip(group,group[1:]):p.edge(a,b,vertical=True)
p.box('<b>重要条件</b><br>AC2：critic 要校验，仍保留真实评分通路。<br>STOPD：局部 loss 不能依赖被删掉的未来；IPW 不能直接恢复需要终局 reward 的 GRPO。',40,1350,1640,150,'C',22)

# Formula cards are separate cells to make TeX easy to edit in draw.io.
def card_formula(row):
 tex=row['formula']
 if not tex:return ''
 if row['name'].startswith('LGR /'):
  tex=r'r_t(a)\propto\operatorname{Conf}_T(h_ta)'
 elif row['name']=='AOPD':
  tex=r'\begin{aligned}\widehat J&=\frac{R-b}{H}\sum_t\bigg[\\&\frac{T(a_t\mid h_t)}{S(a_t\mid h_t)}\log S_\theta(a_t\mid h_t)\\&-\mathbb E_{a\sim T}\log S_\theta(a\mid h_t)\bigg]\end{aligned}'
 elif row['name']=='Draft-OPD':
  tex=r'\begin{aligned}\mathcal L=\tfrac12\big[&\operatorname{mean}_{\rm acc}D_{\rm KL}(T\|S)\\+&\operatorname{wmean}_{\rm rej,\gamma^{k-1}}D_{\rm KL}(S\|T)\big]\end{aligned}'
 else:
  parts=tex.split(r';\quad')
  if len(parts)==1:parts=tex.split(r';\qquad')
  if len(parts)==1 and len(tex)>125:parts=tex.split(r',\quad')
  if len(parts)>1:tex=r'\begin{gathered}'+r'\\'.join(parts)+r'\end{gathered}'
 return math(tex)
for start in range(0,len(P),4):
 p=Page(f'{6+start//4:02d}｜论文公式 {start+1}–{min(start+4,len(P))}',1720,1570)
 p.title('一眼抓住方法：认知 → 改动位置 → 关键公式','所有公式均为 draw.io 原生 LaTeX / MathJax 标签；结构示意与原文公式分别标注。')
 for i,row in enumerate(P[start:start+4]):
  x=40+(i%2)*840;y=170+(i//2)*680;c=(row['components'] or ['D'])[0]
  container=p.box('',x,y,800,640,c,20,'container=1;collapsible=0;')
  p.box('<b>'+html.escape(row['name'])+'</b><br><br><b>认知：</b>'+html.escape(row['high_level'])+'<br><br><b>方法：</b>'+html.escape(row['method']),10,10,780,245,c,20,'fillColor=none;strokeColor=none;',parent=container)
  p.box(card_formula(row) or '关键公式待正文核对',15,265,770,205,c,20,'fillColor=none;strokeColor=none;align=left;',parent=container)
  p.box('<b>改动：</b>'+' · '.join(row['components'])+'<br><b>公式：</b>'+html.escape(row['formula_kind'])+'<br><b>边界：</b>'+html.escape(row['boundary']),10,490,780,140,c,16,'fillColor=none;strokeColor=none;',parent=container)
p=Page(f'{6+(len(P)+3)//4:02d}｜OPD信号变换对比',1720,1690)
p.title('同一研究分支：如何重新决定 teacher–student 分歧的教学强度','共享动机不意味着算法等价：分布插值、系数变换、token 选择、领域尺度，是不同操作。')
p.box('<b>共同更新框架（局部信号）</b><br>'+math(r'g=Z^{-1}\sum_t\operatorname{sg}[m_t w_t f(p_t,q_t)]\nabla\log p_{\theta,t}')+'<br>p：student token 概率；q：teacher token 概率；m：是否保留；w：额外权重',40,170,1640,190,'F',22)
comparison=[('TOP-D｜混合分布',r'q_\lambda=(1-\lambda)p+\lambda q',r'f=\log(1-\lambda+\lambda q/p)','负向有下界；只依赖概率比<br>完整算法还有 return、标准化与内部迭代','T'),('PowerOPD｜变换系数',r'f=q^\alpha-p^\alpha\in[-1,1]',r'f(\epsilon p,\epsilon q)=\epsilon^\alpha f(p,q)','直接考虑绝对概率尺度<br>没有构造归一化的插值 teacher','A'),('DIAL-OPD｜选择 token',r's_\beta=|\log(q/p)|^{1-\beta}|q-p|^\beta',r'f_{\rm effective}=m_t\log(q/p)','插值的是选词分数；选中后仍用 log-ratio<br>完整 rollout 仍需生成和评分','W'),('DN-MOPD｜校准领域尺度',r'w_d=\operatorname{clip}(\sigma_{\rm all}/\sigma_d,0.25,4)',r'f_{\rm effective}=w_d\log(q/p)','领域内同乘正数，保持正负号与排序<br>不是 token 重要性判断','N')]
for i,(title,f1,f2,desc,c) in enumerate(comparison):
 x=40+(i%2)*840;y=410+(i//2)*400
 p.box('<b>'+title+'</b><br>'+math(f1)+'<br>'+math(f2)+'<br>'+desc,x,y,800,350,c,21)
p.box('<b>最直接的重合：DIAL 的选词分数 = Power 的系数幅度</b><br>'+math(r's_{\rm DIAL,\beta=1}=|q-p|=|r_{\rm Power,\alpha=1}|')+'<br>一个据此选择位置，一个直接据此加权；即便 DIAL 保留全部位置，也回到 vanilla OPD。',40,1240,1640,190,'W',22)
p.box('<b>共同边界：概率分歧 / 支持度 ≠ 对最终正确率的因果贡献</b><br>已做：插值、幂变换、稀疏监督、领域缩放。<br>这几篇尚未直接解决：高概率风格偏好是否值得学？低概率修复动作是否关键？<br>新 idea 若围绕任务收益，需要继续对照 AOPD、PIVOT、CC-OPD 等工作。',40,1470,1640,180,'C',21)
ET.indent(mx);ET.ElementTree(mx).write(R/'roadmap.drawio',encoding='utf-8',xml_declaration=True)
# CSV and concise reading cards, using the shared catalog.
with (R/'论文组件矩阵.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator="\n");w.writerow(['论文','修改组件','一句话认知','方法','关键公式','公式性质','问题','适用边界','证据','定位','目录','来源'])
 for p in P:w.writerow([p['name'],' '.join(p['components']),p['high_level'],p['method'],p['formula'],p['formula_kind'],p['problem'],p['boundary'],p['evidence'],p['evidence_location'],p['path'],p['source_url']])
lines=['# 论文一句话认知与关键公式','', '每篇先看认知，再看方法和修改位置。**结构示意**用于统一比较，不表示原文逐字使用该式。待核对条目不猜算法。','']
for p in P:
 lines += [f'## {p["name"]}','',f'**认知：**{p["high_level"]}','',f'**方法：**{p["method"]}','', '**改哪里：**'+'；'.join(c+'｜'+COM[c] for c in p['components'])+'。','']
 if p['formula']:lines += [f'**关键公式（{p["formula_kind"]}）：**','','$$',p['formula'],'$$','']
 else:lines += ['**关键公式：**待取得正文后补充。','']
 lines += [f'**边界：**{p["boundary"]}','',f'**证据：**{p["evidence"]}；{p["evidence_location"]}。[论文目录](../{p["path"]}/)','']
(R/'论文一句话认知.md').write_text('\n'.join(lines))
# Offline, no CDN. TeX source is shown verbatim; rendered math is in Markdown.
payload=json.dumps(DATA,ensure_ascii=False).replace('</','<\\/')
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OPD RoadMap</title>
<style>body{margin:0;background:#f5f7fa;color:#18323a;font:16px/1.65 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1400px;margin:auto;padding:32px}h1{font-size:34px;margin-bottom:5px}a{color:#245ab6}header{background:white;border-radius:18px;padding:24px;margin-bottom:25px}input,select,button{font:inherit;padding:9px 14px;border:1px solid #bfced4;border-radius:9px;margin:4px;background:white}nav{position:sticky;top:0;background:#f5f7faee;padding:12px 0;z-index:2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(370px,1fr));gap:18px}.card{background:white;border:1px solid #d9e4e8;border-radius:16px;padding:23px}.card h2{margin:0 0 12px;font-size:22px}.idea{font-size:19px;font-weight:650}.tag{display:inline-block;padding:3px 9px;border-radius:7px;margin:3px;background:#e7eff7;font-size:13px}.formula{font:16px/1.7 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;padding:15px;background:#ecf5ec;border-left:4px solid #63a773}.boundary{font-size:14px;color:#526a74}details{background:white;border-radius:12px;padding:16px;margin:12px 0}details img{width:100%;height:auto;margin-top:16px}.pending{border-top:4px solid #9ba9af}small{color:#627b85}.tabs button.active{background:#18323a;color:white}#count{font-weight:700;margin:12px 0}@media(max-width:600px){main{padding:14px}.grid{grid-template-columns:1fr}}</style>
<main><header><h1>OPD RoadMap：按组件串联论文</h1><p>SFT · PPO · GRPO · OPD · OPSD ｜ 完整 rollout · critic chunk · 随机截断</p><p><a href="README.md">完整数学说明</a> · <a href="roadmap.drawio">可扩展 draw.io 源图</a> · <a href="roadmap.pdf">PDF 图册</a> · <a href="论文一句话认知.md">全部认知卡片</a></p><small>本页无网络依赖。关键公式显示 LaTeX 源码；Markdown 版本可渲染数学。结构示意与原文公式分别标注，未核对正文的条目单独显示。</small></header>
<section id="figs"></section><nav><input id="search" placeholder="搜索论文、问题、认知…" aria-label="搜索"><select id="component" aria-label="组件"><option value="">全部组件</option></select><select id="evidence" aria-label="证据"><option value="">全部条目</option><option value="read">已据正文 / 同方法版本整理</option><option value="pending">正文待核对</option></select><div id="count"></div></nav><section class="grid" id="cards"></section></main>
<script>const data=PAYLOAD;const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const imgs=['01-baselines.png','02-components.png','03-paper-map.png','04-length.png','05-chunk-stopd.png',...Array.from({length:Math.ceil(data.papers.length/4)},(_,i)=>String(i+6).padStart(2,'0')+'-paper-formulas.png'),String(6+Math.ceil(data.papers.length/4)).padStart(2,'0')+'-signal-comparison.png'];const names=['五种基线方法','十个组件','论文改动地图','token length / chunk','AC2 与 STOPD',...Array.from({length:Math.ceil(data.papers.length/4)},(_,i)=>'论文认知与 LaTeX 公式 '+(4*i+1)+'–'+Math.min(4*i+4,data.papers.length)),'OPD 信号变换对比'];document.querySelector('#figs').innerHTML=imgs.map((f,i)=>`<details ${i===4?'open':''}><summary>${i+1}｜${names[i]}</summary><a href="${f}"><img src="${f}" alt="${names[i]}" loading="lazy"></a></details>`).join('');for(const [k,v] of Object.entries(data.components)){let o=document.createElement('option');o.value=k;o.textContent=k+'｜'+v;document.querySelector('#component').append(o)}function render(){const q=document.querySelector('#search').value.toLowerCase(),c=document.querySelector('#component').value,e=document.querySelector('#evidence').value;const rows=data.papers.filter(p=>(!q||JSON.stringify(p).toLowerCase().includes(q))&&(!c||p.components.includes(c))&&(!e||(e==='pending'?!p.formula:!!p.formula)));document.querySelector('#count').textContent=`显示 ${rows.length} / ${data.papers.length} 篇`;document.querySelector('#cards').innerHTML=rows.map(p=>`<article class="card ${p.formula?'':'pending'}"><h2>${esc(p.name)}</h2><p class="idea">${esc(p.high_level)}</p><p><b>方法：</b>${esc(p.method)}</p><p>${p.components.map(c=>`<span class="tag">${c}｜${esc(data.components[c])}</span>`).join('')}</p><small>${esc(p.formula_kind)}</small><div class="formula">${esc(p.formula||'关键公式待正文核对')}</div><p class="boundary"><b>边界：</b>${esc(p.boundary)}</p><p><small>${esc(p.evidence)} · ${esc(p.evidence_location)}</small></p><a href="${encodeURI('../'+p.path+'/')}">本地论文目录</a> ${p.source_url?` · <a href="${esc(p.source_url)}">来源</a>`:''}</article>`).join('')}for(const id of ['search','component','evidence'])document.getElementById(id).addEventListener('input',render);render();</script></html>'''
(R/'index.html').write_text(page.replace('PAYLOAD',payload))
print(f'Generated LaTeX draw.io pages, interactive index and {len(P)} cards.')
