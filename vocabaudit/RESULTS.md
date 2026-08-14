# stage4 主实验记录（vocabaudit）

协议冻结：OpenCLIP ViT-B/16 (openai, quickgelu)，openai 80 模板，short=336，window=224，stride=112，
logit_scale=40，softmax 后子查询 max 池化（SCLIP 惯例），mIoU 只统计 GT 出现类。
whitening 统计量默认取自当前推理词表 query 集合；shrink=0.5（初始）。

## 实现校准
- CSA 修正为 softmax(qq)+softmax(kk)（对齐 SCLIP 官方）后，VOC-21（300 图，SCLIP 词表）：sclip=55.4。
  论文全集报告 59.1（含 PAMR/全集），复现水平可接受。
- 注意：07-30 白天的前 6 个 json 是 CPU 上跑的（torch 被 open_clip 安装拉到 cu130 不可用），数值有效。
  之后 torch 2.4.1+cu121 修复，GPU ~0.1s/img。

## KC1/KC2 裁决实验：PC-459（500 图，plain 459 词表）

| variant | none | center | zca(vocab,s=0.5) | zca(global=ade847+coco171) |
|---|---|---|---|---|
| sclip     | 12.78 | **13.70** | 11.63 | 11.88 |
| clearclip | **15.28** | 15.16 | 13.24 | 13.71 |
| maskclip  | **9.62** | 9.27 | 8.21 | 8.37 |

**KC 裁决（2026-07-30，shrink=0.5，待 shrink 扫描复核）：KC1 失败，KC2 失败。**
按 card2 预注册规则执行：砍掉方法主贡献，pivot 到 audit-only 叙事（目标 D&B / workshop / findings 类论文）。

判读：
- KC1（whitening 增益 > +0.5）：ZCA whitening 在大词表上**降低** mIoU，KC1 趋向失败。
- KC2（词表条件化 > 全局统计量）：vocab-zca ≈ global-zca（甚至略差），KC2 趋向失败。
- 例外信号：仅去均值 (center) 对 SCLIP +0.9，对 clearclip 中性——文本端修复空间存在但很窄。
- 嵌入诊断（margin/eff-rank 大幅改善）与分割 mIoU 脱钩：whitening 扩大类间 margin 的同时
  破坏了与视觉 patch 特征的对齐几何。这本身是审计论文的重要发现。

> 注（v2）：发现同义词轴曾误替换 background（"background knowledge"），修复后加入
> EXCLUDE={background,unknown,other} 规则并重跑全部 syn/dis 条件；v1 结果归档于
> runs/archive_v1_bgleak/。下方 VOC 矩阵为 v1 初值，**论文表格以 v2 json 为准**
> （种子间波动修正为最大 3.7 pts；干扰词 near+200 = 44.5/40.9/37.1）。

## VOC-21 审计矩阵（300 图，plain 词表基准 = 100% 一致协议）

mIoU（sclip / clearclip / maskclip）：
- SCLIP 官方词表（bg 展开26子类）：55.4 / 53.6 / 43.6
- plain 词表：34.8 / 34.9 / 31.8   ← **命名惯例差 ≈ +20 mIoU，最大单一敏感源**
- 同义词 25%：33.3 / 33.0 / 29.1
- 同义词 50%（seed 0/1/2）：27.6,33.2,29.1 / 28.4,32.8,27.7 / 25.4,28.9,26.0  ← 种子间波动最高 5.6 pts
- 同义词 100%：28.4 / 27.9 / 24.6
- 干扰词 near+50：41.6 / 38.2 / 36.6   ← mIoU **上升**
- 干扰词 near+200：45.4 / 41.4 / 37.8  ← 继续上升
- 干扰词 mid+50：40.4 / 37.7 / 35.3

干扰词提升 mIoU 是背景类建模缺陷的暴露：plain 'background' 文本嵌入吸附不了背景像素，
干扰类反而替它吸收了假阳性 → GT 类 precision 提升。审计层面的核心发现之一。

zca 在 VOC plain 上 +0.9~+1.5、在 PC-459 上 −1.4~−2.0：whitening 效果随词表规模反号，
与嵌入诊断（margin 单调改善）解耦 → 文本端几何指标不是分割性能的可靠代理。

## 补充实验（全部完成 2026-07-30）

### shrink 扫描（clearclip / PC-459 / vocab-zca）
s=0.1: 11.60, s=0.3: 12.57, s=0.5: 13.24, s=0.7: 13.93, s=0.9: 14.71, none: 15.28
→ 单调：whitening 强度越小越好，KC1 在整个 shrink 谱上失败（非调参问题）。

### 标准词表 regime（500 图，none/center/zca）
- ADE-150：sclip 13.08/12.35/10.76；clearclip 14.72/13.85/12.70；maskclip 10.21/9.24/8.06
- COCO-171：sclip 22.90/21.23/19.85；clearclip 24.21/23.11/21.70；maskclip 17.43/15.50/15.15
→ center 也普遍负效（PC-459 sclip 的 +0.9 是孤例）。文本端后处理无免费午餐。

### 粒度轴（冻结 WordNet 首义上位词规则，background/unknown 豁免）
- VOC-21：34.8 → 17.0 / 34.9 → 17.0 / 31.8 → 16.9（腰斩）
- ADE-150：13.1 → 2.5 / 14.7 → 3.8 / 10.2 → 1.9（几乎归零）
→ 上一层粒度的自动名称替换对 training-free OVSS 是灾难性的；也暴露自动层级工具的首义失配风险。

### 分类 vs 分割对照（VOC-21，GT bbox crop，同词表同模板同池化）
- plain：cls acc 72.8 / seg mIoU 34.8
- syn50 (s0/1/2)：cls 71.0/65.4/65.8；seg 34.4/33.2（旧版种子波动更大）
- syn100：cls 63.0；seg 29.0
- **dis_near200：cls 54.3（−18.5）；seg mIoU 反而 +10**
→ 分类准确率与分割 mIoU 在干扰词轴上**方向相反**：词表敏感性不是分类结论的平移，
  dense 预测 + mIoU 协议有自己的失真机制（背景吸收效应）。dense-specific 贡献成立。

### 排名稳定性
- 官方词表：sclip(55.4) > clearclip(53.6) > maskclip(43.6)
- plain：clearclip(34.9) ≈ sclip(34.8) > maskclip(31.8) —— 排名翻转
- PC-459：clearclip(15.3) > sclip(12.8) > maskclip(9.6) —— 再翻转
→ 方法间排序依赖词表 regime，现行单词表比较不足以支撑 SOTA 声明。

## 尚未做（写作阶段可选补）
- A-847（需 ADE20K 2021 注册下载，受阻）；已用 PC-459 替代大词表 regime
- ProxyCLIP/NACLIP 变体（需外部 DINO backbone/官方仓库对齐，当前三变体已满足 3 方法要求）
- Kendall tau 数值化（方法数=3，直接报排名翻转更诚实）

## Stage6 rebuttal experiments (2026-07-30, runs/reb_*.json)

针对模拟审稿意见补做（全部 GPU、v2 协议）：

1. Engineered-background 对照（R2-Q4）：official vocab + 200 near distractors（300 图）
   - SCLIP 55.4 → 52.5（−2.9）；ClearCLIP 53.6 → 51.6（−2.0）；MaskCLIP 43.6 → 41.9（−1.7）
   - 结论：plain 词表下的 distractor 增益在背景充分建模时反转为下降 → 背景吸收机制被隔离。
2. 标准 all-class mIoU（R1-W9）：evaluator 新增 miou_all
   - plain+200near：SCLIP/ClearCLIP/MaskCLIP all-class mIoU = 4.2/3.9/3.5（GT-present 44.5/40.9/37.1）
   - 词表扩张效应的方向本身是 metric 约定的函数。
3. VOC-21 全量 split 校准（R1-W2/R2-W6，1449 图）：
   - plain：SCLIP 35.2 / ClearCLIP 34.5 / MaskCLIP 32.5
   - official：SCLIP 56.9 / ClearCLIP 55.4 / MaskCLIP 44.4（Δ 21.7/21.1/11.9，命名效应全量成立）
4. 子集重采样方差（R1-W2/R2-W2，plain，4 个不相交 300 图子集）：
   - SCLIP 34.75/33.88/34.86/37.00（range 3.1）；ClearCLIP 34.89/33.22/33.61/35.14（range 1.9）
   - VOC plain 的 0.1 "ranking flip" 在噪声内，论文已降级该主张。
5. dis_near200 v2 重跑带 miou_all：与原 v2 数字一致（44.5/40.9/37.1）。

## Stage6 round-2 补跑 (runs/reb2_*.json)：global-ZCA 补齐 ADE-150/COCO-171

- ADE-150 global-ZCA：SCLIP 10.9 / ClearCLIP 12.9 / MaskCLIP 8.3（vocab-ZCA 10.8/12.7/8.1，none 13.1/14.7/10.2）
- COCO-171 global-ZCA：SCLIP 18.9 / ClearCLIP 21.4 / MaskCLIP 14.4（vocab-ZCA 19.9/21.7/15.2，none 22.9/24.2/17.4）
- KC2 结论从 PC-459-only 扩展到 3 个数据集 9 个 cell：词表条件化统计量无一处优于全局统计量。

## NACLIP 复制检验 (runs/nac_*.json, 2026-07-30)

第四个方法（k-k attention + Gaussian 邻域先验，同一冻结协议重实现）复制全部效应：
- 命名：official 55.0 vs plain 36.5（Δ18.5）
- 同义词：syn50 s0/s1/s2 = 33.9/34.5/32.2；syn100 29.7
- 粒度：voc21 36.5→18.1；ade150 15.4→4.1
- 干扰词：plain+200near 42.5（+5.9）；official+200near 52.3（−2.7 反转）
- 修复（PC-459）：none 14.3 / zca 12.8（−1.6）/ center 14.5（+0.2，第二个 centering 正例）
