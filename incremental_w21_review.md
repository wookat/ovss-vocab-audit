# 增量复审意见 — CorrCLIP VABS 验证 + tab:dosecurve + Limitations（对抗性，A 会标准）

**复审范围**：main.tex L457–489（CorrCLIP 段 + dose-curve 表）、L695–699（four official releases 句）、L700–709（applicability warning）。不重审全文。

---

## ① 数字核对（正文/表格 vs run 值）

| 项 | run 值 | 正文/表格 | 判定 |
|---|---|---|---|
| VOC plain | 54.30 | 54.3 | ✓ |
| VOC +VABS64 | 69.54 | 69.5 | ✓ |
| VOC 增益 | +15.24 | +15.2 | ✓ |
| 收回比例 | 15.24/20.5 = 74.3% | 74% | ✓ |
| VOC rand64 | 65.63 | 65.6 | ✓ |
| 选择优势 | 69.54−65.63 = +3.91 | +3.9 | ✓ |
| COCO plain | 42.26 | 42.3 | ✓ |
| COCO VABS | 38.18（−4.08） | −4.1 | ✓ |
| COCO rand | 38.95（−3.31） | −3.3 | ✓ |
| VABS−rand | −0.77 | (−0.8) | ✓ |
| 冻结判据 | 增益≥+3 且选择>0：+15.2 / +3.9 双通过；COCO VABS−plain<+3 成立 | 如实 | ✓ |

数字全部一致，无造假/搬运错误。

## ② 主张强度

**M1（必改）"the first author-code harm point at a NEG as small as $+1.4$"（L465）— 表述有两处错误。**
(a) "first" 不成立：Trident/Context-60（NEG +0.2，L452–456）已是 author-code harm 点，且写在同一段前文。
(b) "as small as" 方向反了：+1.4 比已有 harm 点（+0.2）**大**，新信息恰恰是 harm 出现在**更大**的 NEG 上。建议改为 "extending the author-code harmful regime up to a NEG of $+1.4$, the largest headroom at which harm has been observed" 之类。

**M2（必改）"replicates both ends of the curve"（L459）— 超证据。**
CorrCLIP/COCO（NEG +1.4）落在 harmful 档，而 NEG 相近且更大的 Trident/COCO（+2.0）是 parity 档——低端并非"复现"同一端点，而是把 harmful 端**外推**到了更高 NEG，且揭示了跨宿主的非单调性（+1.4 harm < +2.0 parity）。冻结预测本身只是 VABS−plain<+3（null 预测），预测的是"低于门槛"而非"有害"；"the frozen boundary prediction held" 这句可以保留，但 "replicates both ends" 应改为 "reproduces the recovery end and extends the harmful end"，并加半句承认 NEG 不能跨宿主单调排序 regime（CorrCLIP 在 +1.4 有害而 Trident 在 +2.0 仅持平），这与 Limitations 中 "consistent with every measured point but ... not a validated predictor" 的谨慎立场一致——现在这两个点之间实际上已经出现了排序张力，正文不点破会被审稿人抓。

**M3（建议→接近必改）+3.9 选择优势缺少 seed 披露。** SC-CLIP 锚点处明确写了 "(single run, single negative seed)"（L409），CorrCLIP 段没有对应披露。+3.9 是全文最大的 author-code 选择优势，单 seed 单 run 更需注明，避免读者拿它与 in-stack 多 seed 均值直接比大小。

其余表述（"官方前向零改动、只改 name file"、VABS-only、−0.8 只作描述）与预注册 prereg_w21_corrclip_vabs.md 声明一致，未见仲裁主张越界。✓

## ③ tab:dosecurve 表

**M4（必改）括号选择值未定义。** 表中 (−0.8)、(+0.7) 加括号但 caption 只说 "selection is VABS − matched random"，未说明括号含义。加一句 "parenthesised selection values are differences between two harmful arms and are reported descriptively only, with no benefit claim"。

**M5（必改）Trident/COCO 行 selection "−0.1" 与正文 "ties (40.4 vs 40.4)"（L448）表面冲突。** 若未取整值为 −0.05 级，请在正文改为与表一致的 −0.1，或表中用 −0.0，二选一统一；同时 L439 另有 "$-0.0$"（SC-CLIP 宿主 COCO），三处口径需自洽。

**建议 S1**：caption "only the class-name file changed" 对 CorrCLIP 而言技术上成立（负词经 repo 自带 '; ' 别名机制注入 background 行），但 "区域掩码预生成、各臂相同" 是保证公平比较的关键细节，建议在 caption 或正文加半句/脚注。

双宿主格式本身清晰：五行按 NEG 递减排列直观展示 dose curve，regime 列命名合理。✓

## ④ Limitations 一致性

- L695–699 "four official releases ... CorrCLIP" + "VABS only"：与正文 L466 一致，仲裁未验证的划界保持。✓
- **M6（必改）L700–703 applicability warning 的 dose-curve 枚举只列 Trident 三点**（VOC +19.7 / COCO +2.0 tie / Context-60 +0.2 harm），未纳入 CorrCLIP/COCO 的 **NEG +1.4 → harm**。这个新点把"危险区"从近零 NEG 上移到 +1.4，且与 Trident +2.0 tie 形成非单调，直接影响部署建议的强度——warning 必须更新，否则读者会低估风险边界。
- **建议 S2**：Abstract L64 仍只说 "VABS transfers to official ProxyCLIP and SC-CLIP code"，落后于正文的四宿主覆盖；虽在复审范围之外，建议同步更新（一处即可）。

## ⑤ 清单与裁决

**必改（Accept 前提）**
1. M1：改写 "first ... as small as +1.4"（方向与 "first" 均错）。
2. M2："replicates both ends" 降格为 recovery 端复现 + harmful 端外推，并点明 +1.4/+2.0 跨宿主非单调。
3. M4：caption 定义括号选择值。
4. M5：Trident/COCO selection −0.1 vs 正文 tie/−0.0 口径统一。
5. M6：Limitations applicability warning 纳入 CorrCLIP COCO +1.4 harm 点。

**建议**
- M3/S1：CorrCLIP 段补 "(single run, single negative seed)" 与掩码预生成披露。
- S2：Abstract 官方宿主覆盖同步。

**分数裁决**：数字与预注册对齐无瑕疵，证据本身扎实且新增了曲线两端的高价值数据点；问题集中在 3–4 处措辞超证据/口径不一，均为局部可修。**维持 Accept 档**（修必改项后无保留；不修 M1/M2/M6 则降为 weak accept，因当前措辞在 harm 边界上既夸大新颖性又低报部署风险）。
