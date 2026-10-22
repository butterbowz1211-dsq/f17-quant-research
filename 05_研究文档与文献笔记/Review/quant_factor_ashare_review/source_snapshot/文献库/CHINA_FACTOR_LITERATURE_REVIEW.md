<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# China A-share Factor Literature: Methodology-Level Extraction

**Prepared for:** A-share daily-bar multi-factor strategy (2010–2026, ~5,200 stocks, OHLCV + turnover + float mktcap + 83–84 baostock industries)
**Author:** Gilmour Jia
**Scope:** full-text reads of the methodology / data / results / appendix sections of the papers below. No numbers below are invented; every figure is quoted from a section I actually read. Papers I could not access are listed explicitly in §7.

---

## 0. Corpus actually read (full text, not abstracts)

| # | Paper | Source file read | Access route |
|---|-------|------------------|--------------|
| P1 | Liu, Stambaugh, Yuan (2019), "Size and value in China", JFE 134(1), 48–69 | NBER WP w24458, 57 pp. | https://www.nber.org/system/files/working_papers/w24458/w24458.pdf |
| P2a | Jansen, Swinkels, Zhou (2021), "Anomalies in the China A-share market", Pacific-Basin Finance Journal 68, 101607 | 28 pp. CC-BY published version | https://pure.eur.nl/ws/portalfiles/portal/58642799/Anomalies_in_the_China_A_share_market.pdf |
| P2b | Hou, Qiao, Zhang (2023), "Finding Anomalies in China" (working paper; authors are Kewei Hou, Fang Qiao, **Xiaoyan Zhang** — *not* Lu Zhang) | 116 pp. | https://cfrc.pbcsf.tsinghua.edu.cn/__local/1/28/BE/ECB6EDE9DA787BEA738415A420D_9BF29B07_1BA025.pdf |
| P3a | Chui, Titman, Wei (2010), "Individualism and Momentum around the World", JF 65(1), 361–392 | 40 pp. | http://www.fin.ntu.edu.tw/~conference/conference2004/proceedings/proceeding/6/6-4(A63).pdf |
| P3b | Kang, Liu, Ni (2002), "Contrarian and momentum strategies in the China stock market: 1993–2000", PBFJ 10(3), 243–265 | 23 pp. | http://www.technicalanalysis.org.uk/momentum/KaLN02.pdf |
| P3c | Cheema, Nartea (2017), "Momentum, idiosyncratic volatility and market dynamics: Evidence from China", PBFJ, doi:10.1016/j.pacfin.2017.09.001 | 42 pp. accepted manuscript | https://researchcommons.waikato.ac.nz/bitstreams/a9e00dd8-fa27-4dc4-89ef-8ca85438a95a/download |
| P3d | Cheema, Nartea, Man (2020), "Maxing Out in China: Optimism or Attention?", International Review of Finance 20(4), doi:10.1111/irfi.12241 | 14 pp. | https://researchcommons.waikato.ac.nz/bitstreams/0061c5f2-31b5-461c-ace4-b9fd75bbc060/download |
| P3e | Liu, Tan, Xu, Yuan, Zhu (2025), "Dissecting Momentum in China", working paper, Feb 2025 | 58 pp. | http://yuchenxu.com/paper/Momentum.pdf |
| P4 | Daniel, Moskowitz (2016), "Momentum crashes", JFE 122(2), 221–247 | 27 pp. | https://www.kentdaniel.net/papers/published/jfe_16.pdf |
| P5a | Bali, Cakici, Whitelaw (2011), "Maxing Out…", JFE 99(2), 427–446 | NBER w14804, 50 pp. | https://www.nber.org/system/files/working_papers/w14804/w14804.pdf |
| P5b | Nartea, Kong, Wu (2017), "Do extreme returns matter in emerging markets? Evidence from the Chinese stock market", JBF 76, 189–197 | full text | https://silo.tips/download/do-extreme-returns-matter-in-emerging-markets-evidence-from-the-chinese-stock-ma (via r.jina.ai) |
| P5c | "The MAX Effect in China's A-Share Market from the Perspective of Investor Behavior", SCIRP 2023 | 23 pp. | https://www.scirp.org/pdf/tel_2023081014211370.pdf |
| P5d | (sentiment) "Investor Sentiment and Chinese A-Share Stock Markets Anomalies" | 20 pp. | https://pdfs.semanticscholar.org/be8a/f711439d13920c13083e20cfeffda9df69f7.pdf |

---

## 1. Liu–Stambaugh–Yuan (2019) — full construction detail

### 1.1 Data, sample, filters (Appendix A.1)
- **Source:** WIND (trading + financials); risk-free = 1-year deposit rate from CSMAR/WRDS.
- **Sample:** Jan 2000 – Dec 2016 (**204 months**). A-shares from Shanghai main board, Shenzhen main board and GEM; tickers whose first two digits are **60, 30, 00**.
- **Four filters:**
  1. Exclude stocks that became public within the past **6 months**.
  2. Exclude stocks with **< 120 trading days in the past 12 months**.
  3. Exclude stocks with **< 15 trading days in the most recent month**. (2 and 3 exist to kill long-suspension return artifacts.)
  4. **Eliminate the bottom 30% of stocks ranked by market capitalisation at the end of the previous month.** Market cap = close price × **total shares outstanding, including non-tradable shares**.
- **Minimum portfolio size:** "at least 50 stocks in all portfolios after imposing our filters". Stock count grows from **610 (2000) → 1,872 (2016)**, so each decile holds ≥60 stocks even early.
- **ST stocks:** LSY **do not** mention excluding ST/PT stocks. (JSZ do — see §2.1.)
- **Missing data:** accounting sorts use the financial report with the **most recent public release date prior to that month's end** (WIND release dates). Quarterly statements from Jan 2002; semi-annual before that.

### 1.2 Why the size screen exists (Section 3)
- CSRC caps IPO throughput (7 administrative steps, 25-member committee, ~3 years, uncertain). Private firms use **reverse mergers** into shells.
- Of **133 reverse mergers (2007–2016), 83% come from the bottom 30%, and more than half from the bottom 10%**.
- Shell value estimate: `S = pG / (r + p)`, `p` = annual rate at which bottom-30% stocks become shells, `G` = average value gain on the event, both estimated on a **2-year rolling window**; `r = 3%` (average 1-year deposit rate 2007–2016). Panel A of Fig. 2: average **S/V = 29.5%** for the bottom 30% (range 10%–60%).
- The bottom 30% are only **7% of total market capitalisation**, yet including them *raises* the size premium and *lowers* the value premium and breaks the model (see 1.5).
- Smallest 30% returns are **less** related to fundamentals: regression of earnings-window returns on SUE, slope `b` (×100) for China = **0.14 (t=6.34) smallest, 0.17 (12.42) middle, 0.24 (17.28) largest** — i.e. *increasing* in size, the **opposite of the US** (0.19 / 0.07 / 0.05). (k=3 window: 0.43 / 0.58 / 0.59 China vs 0.52 / 0.20 / 0.13 US.)

### 1.3 Value: the EP vs BM horse race (Section 4, Table 2)
Fama–MacBeth regressions, 2000–2016, Newey–West 4 lags, β from past-year daily returns with 5-lag Dimson correction. Smallest 30% excluded.

| Regressor | Alone (t) | Joint, col (8) (t) | EP+BM only, col (9) (t) |
|---|---|---|---|
| β | −0.0002 (−0.09) | −0.0004 (−0.15) | — |
| logME | −0.0049 (−2.91) | −0.0064 (−4.40) | −0.0066 (−4.49) |
| logBM | 0.0057 (3.21) | 0.0022 (1.31) | 0.0035 (1.76) |
| logAM | 0.0045 (3.03) | 0.0014 (0.99) | — |
| **EP+** | **0.9503 (4.88)** | **0.7825 (4.38)** | **0.7960 (5.06)** |
| D(EP<0) | — | −0.0005 (−0.29) | −0.0001 (−0.04) |
| CP+ | 0.0546 (3.41) | 0.0181 (1.35) | — |
| D(CP<0) | — | 0.0016 (2.37) | — |
| adj R² | 0.0196–0.0677 | **0.0832** | 0.0776 |

- **EP dominates.** One standard deviation of EP+ ⇒ **0.52% difference in expected monthly return**.
- Financial firms are **kept** (as in FF 1993); dropping them (incl. real estate) leaves results virtually unchanged (Appendix A.6).

### 1.4 CH-3 construction (Section 5.1) — exact recipe
1. Universe = top 70% by market cap (previous month-end, total shares incl. non-tradable).
2. **Independent** monthly sort into 2 size groups at the **median market value of that universe** (S, B).
3. Independent monthly sort into 3 EP groups: **top 30% = value (V), middle 40% (M), bottom 30% = growth (G)**.
4. Six **value-weighted** portfolios (S/V, S/M, S/G, B/V, B/M, B/G); weights = market cap of **all outstanding A shares, including non-tradable shares**.
5. **SMB = ⅓(S/V + S/M + S/G) − ⅓(B/V + B/M + B/G)**
6. **VMG = ½(S/V + B/V) − ½(S/G + B/G)**
7. **MKT** = VW return of the top-70% universe minus the 1-year deposit rate.
8. **Negative-EP stocks are KEPT and classified as growth** (they comove with growth; they load −0.28, t=−3.31 on a positive-EP-only value factor). They are on average **15% of the universe**. Excluding them is a robustness check that holds.

**Factor statistics (Table 3, 204 months):**

| Factor | Mean %/mo | SD | t |
|---|---|---|---|
| MKT | 0.66 | 8.09 | 1.16 |
| SMB | **1.03** | 4.52 | **3.25** |
| VMG | **1.14** | 3.75 | **4.34** |

Correlations: MKT–SMB +0.12, MKT–VMG −0.27, SMB–VMG **−0.62** (small stocks tend to be growth stocks in China). Two-factor alphas: MKT 1.57 (2.30), SMB 1.91 (6.92), VMG 1.71 (7.94).

**Variance explained (Table 4, avg R² over rolling 36-month regressions):** China all stocks: MKT 0.385; MKT+SMB 0.507; MKT+VMG 0.471; all three **0.536**. China top 70%: 0.417 / 0.528 / 0.501 / **0.562**. US (FF-3): 0.177 / 0.231 / 0.226 / **0.273**. China's market factor alone explains **2×** the US. Size+value add **15.1%** in China vs **9.6%** in the US.

### 1.5 CH-3 vs FF-3 (Section 5.3, Table 5)
- FF-3 leaves **SMB α = 0.47%/mo (t=7.03)** and **VMG α = 1.39%/mo (t=7.93)** → 16.68% annualised.
- CH-3 leaves FFSMB α = −0.04 (t=−0.66) and FFHML α = 0.34 (t=0.97).
- GRS: CH-3 pricing FF-3's factors p = **0.41**; FF-3 pricing CH-3's factors p = **2.14 × 10⁻¹³**.
- **Including shells:** SMB premium rises to **1.36%**, VMG falls to **0.87%**; shell-free SMB then has α = **−0.23%/mo (t=−3.30)** and VMG α = **+0.27% (t=3.32)** against the shell-inclusive model.

### 1.6 The 14 anomalies tested and which are significant (Section 6, Table 6)
LSY survey the literature, find **14 anomalies in 9 categories**. Long-short = decile 1 minus decile 10 **value-weighted**, long leg = higher-performing decile. CAPM alphas, monthly %, White t-stats, 2000–2016.

**Panel A — unconditional sorts**

| Category | Anomaly | R̄ | α | β | t(R̄) | t(α) | t(β) |
|---|---|---|---|---|---|---|---|
| Size | Market cap | 1.09 | 0.97 | 0.18 | 1.92 | 1.81 | 1.90 |
| Value | EP | 1.27 | **1.37** | −0.16 | 2.58 | **2.93** | −2.15 |
| Value | BM | 1.14 | **1.14** | 0.01 | 2.08 | **2.13** | 0.06 |
| Value | CP | 0.73 | 0.70 | 0.04 | 1.67 | 1.69 | 0.55 |
| Profitability | ROE | 0.83 | **0.93** | −0.15 | 1.77 | **2.11** | −2.09 |
| Volatility | 1-month vol | 0.81 | **1.03** | −0.34 | 1.64 | **2.31** | −5.55 |
| Volatility | **MAX** | 0.57 | **0.81** | −0.36 | 1.26 | **2.02** | −6.39 |
| Reversal | 1-month return | 1.47 | **1.49** | −0.02 | 2.96 | **3.06** | −0.22 |
| Turnover | 12-month turnover | 0.33 | 0.53 | −0.29 | 0.63 | 1.09 | −3.46 |
| Turnover | 1-month abnormal turnover | 1.14 | **1.27** | −0.19 | 2.44 | **2.92** | −2.66 |

*(Note: in the PDF's Table 6 the "12-Month turn." row is typeset under the "Reversal" category label; the paper's own text — "The monthly CAPM alphas range from 0.53%, for 12-month turnover, to 1.49%, for one-month reversal" — confirms it is the 12-month **turnover** row.)*

**Panel B — size-neutral sorts** (size deciles first, then anomaly deciles within each, then pool + value-weight)

| Anomaly | R̄ | α | t(R̄) | t(α) |
|---|---|---|---|---|
| EP | 1.80 | **1.89** | 4.32 | **4.72** |
| BM | 1.14 | **1.10** | 2.23 | **2.22** |
| CP | 0.78 | **0.76** | 2.27 | **2.25** |
| ROE | 1.45 | **1.50** | 3.90 | **4.11** |
| 1-month vol | 0.66 | **0.90** | 1.41 | **2.19** |
| MAX | 0.39 | 0.60 | 0.93 | 1.61 |
| 1-month return | 1.67 | **1.65** | 3.65 | **3.68** |
| 12-month turnover | 0.51 | 0.74 | 1.06 | 1.74 |
| 1-month abn. turnover | 1.29 | **1.39** | 3.19 | **3.68** |

- Size-neutralisation **increases** several alphas: ROE **+0.57%**, EP **+0.52%**, 12-month turnover **+0.21%**; and cuts standard deviations (so t-stats rise). Average α across the 10: **1.02%/mo (avg t 2.21)** unconditional, **1.17%/mo (avg t 2.91)** size-neutral.
- **Do NOT work in China** (no significant CAPM alpha, either sort): **investment** (α ≤ 0.22%/mo), **accruals** (α ≤ 0.42%/mo), **illiquidity** (not quite significant but economically large, **up to 0.83%/mo**).

### 1.7 Reversal — the strongest single finding in LSY
> "Reversal effects in China are especially strong. Past performance over any length window tends to reverse in the future."

Untabulated LSY results: past returns over **1, 3, 6 and 12 months, as well as 5 years, ALL negatively predict future returns**, in monotonically weakening magnitudes.
- **1-month window**: biggest losers beat biggest winners by a CAPM alpha of **18% annualised (t=2.96)**.
- **1-year window**: alpha drops to **6% and becomes insignificant (t=0.90)**.
- **Not bid–ask bounce**: using WIND best bid/ask mid-price returns (2012–2016), the 1-month reversal long-short average monthly return is **2.21% vs 2.15%** using closing prices.

**LSY's reversal rebalancing recipe (Appendix A.2) — this is unusual and important:**
> "Because one-month return reversal is a short-term anomaly, we sort the stock universe **each day** based on the most recently available 20-day cumulative return. Using this sort, we rebalance a **one-fifth 'slice'** of the total portfolio that is then held for **five trading days**. Each day we average the returns across the five slices. Those resulting daily returns are then compounded across days to compute the reversal anomaly's monthly return."

For **all other anomalies**: monthly sort on most-recent month-end measures, hold 1 month, **value-weighted** deciles 1 and 10.

### 1.8 CH-3 alphas for the ten anomalies (Table 7 Panel A)

| Anomaly | CH-3 α | t(α) | Notable loading |
|---|---|---|---|
| Market cap | 0.21 | 1.71 | t(β_SMB)=40.88, t(β_VMG)=−11.70 |
| EP | 0.04 | 0.16 | t(β_VMG)=14.75 |
| BM | 0.64 | 1.02 | — |
| CP | 0.20 | 0.45 | t(β_VMG)=4.00 |
| ROE | −0.36 | −0.88 | **t(β_VMG)=9.43** |
| 1-month vol | 0.23 | 0.44 | t(β_VMG)=3.86 |
| MAX | 0.27 | 0.65 | t(β_VMG)=2.55 |
| **1-month return (reversal)** | **0.93** | **1.70** | t(β_SMB)=3.15 |
| 12-month turnover | 0.42 | 1.30 | t(β_SMB)=−9.33 |
| **1-month abn. turnover** | **1.28** | **2.86** | small, insignificant loadings |

- **CH-3 explains 8 of 10**; **fails only on reversal and abnormal turnover**. Size-neutral sorts: reversal α **1.13 (t=2.12)**, abnormal turnover α **1.24 (t=3.04)**.
- Illiquidity: CAPM α up to 0.83%/mo → **CH-3 α only 0.23% (t=1.14)**.
- Average |CH-3 α| across the 8 explained anomalies = **0.30%** (uncond) / **0.26%** (size-neutral); FF-3 = **0.84%** / **0.90%**.
- GRS joint test of zero alphas on all 10: **CH-3 p = 0.15** (uncond), **0.05** (size-neutral); FF-3 p < 10⁻⁴.
- **Key interpretation:** ROE, volatility and MAX all **load heavily on VMG**. "Low (high) volatility stocks behave similarly to value (growth) stocks." Profitability in China is *captured by* CH-3, unlike the US.

### 1.9 CH-4 — the turnover (PMO) factor (Section 7)
- **Motivation:** >**101 million** individual trading accounts; individuals own **88% of free-floating shares** (year-end 2015). Shorting is "extremely costly" in China. Baker–Stein (2004) logic ⇒ with short-sale constraints, high turnover accompanies strong **optimism**, so high turnover predicts **low** returns.
- **Fourth factor PMO (pessimistic minus optimistic)** built on **abnormal turnover = past month's share turnover ÷ past year's turnover**.
- Constructed **precisely like VMG**: size-neutralised, same breakpoints (2 size groups × 3 turnover groups at 30/40/30), but **long low turnover, short high turnover**.
- **New SMB = simple average of the EP-neutralised SMB from CH-3 and the turnover-neutralised SMB** (following FF 2015).
- New SMB and PMO have annualised averages of **11% and 12%**.
- **Result:** CH-4 α for abnormal turnover = **0.00% (t=−0.01)**, for reversal = **0.49% (t=0.87)**. Average |α| **0.30%** (vs 0.45% CH-3); average |t| **0.69** (vs 1.12); GRS p = **0.41**.
- Rank correlation between the turnover and reversal sorting variables ≈ **0.3**.

### 1.10 LSY's stated reasons US anomalies fail / are distorted in China
1. **Shell-value contamination of small stocks** (primary and proprietary): tight IPO regulation → reverse mergers → 83% of shells from the bottom 30% → ~30% of their market value is shell optionality. This **inflates the size premium**, **deflates the value premium** (poor-earnings small firms have low EP and get misclassified as growth), and breaks factor pricing of regular stocks.
2. **Retail-dominated investor base** (101m accounts, 88% of float) ⇒ high susceptibility to sentiment ⇒ turnover is a valid sentiment gauge.
3. **Short-sale impediments** ⇒ overpricing is not arbitraged ⇒ anomalous returns survive (and the turnover-sentiment link has the sign it does).
4. **IPO constraints / the reverse-merger channel** distort the smallest firms' returns and their correlation with fundamentals.
- **LSY do NOT invoke the T+1 rule or price limits as reasons for anomaly failure.** (Price limits appear in *other* papers, and only as a *measurement* problem — see §2.1 and §2.4.)

---

## 2. Other China anomaly papers

### 2.1 Jansen, Swinkels, Zhou (2021) — 32 anomalies, 2000–2019

**Data & filters (Section 2):** CSMAR, SHSE + SZSE, main board + STAR + ChiNext, Jan 2000 – Dec 2019 (**data from 1990 used to build lags**). Firm counts 1,060 (2000) → 3,750 (2019). Market index = internally computed VW portfolio of all investable A-shares (CSMAR's own index has look-ahead bias w.r.t. suspensions). Risk-free = 1-year RMB lump-sum deposit rate.

Filters, following LSY:
1. Listed **≥ 6 months**.
2. Suspended **< 5 trading days in the past month**.
3. Suspended **< 120 trading days in the past year**.
4. Exclude stocks **suspended on the portfolio construction / rebalancing day**.
5. **Exclude ST ("special treatment") and PT ("particular transfer") stocks** — these have a **5% daily price limit** vs the normal 10%.
6. **Eliminate the bottom 30% by market value (microcaps).**

Accounting: release dates used; if no release date, discard interim reports but retain annual reports assuming a **1 May** release date. Quarterly statements from 2002; semi-annual before.

**Portfolio construction (Section 3):** monthly deciles on each characteristic; long-short = extreme deciles, long leg = the one with higher expected return; **both equal- and value-weighted**; **overlapping (Jegadeesh–Titman) holding periods** for K = 1, 6, 12 months (rebalancing 1/K each month).

**Table 4 — anomaly returns, % per month, Jan 2000 – Dec 2019, Newey–West t-stats, holding period 1 month**

| Category | Anomaly | EW ret | EW t | VW ret | VW t |
|---|---|---|---|---|---|
| Size | SIZE | 0.59 | 1.36 | 0.64 | 1.21 |
| Value | BM | 1.12 | 3.02 | 1.06 | **2.38** |
| Value | EP | 1.10 | 3.41 | 1.09 | **2.53** |
| Value | SP | 0.93 | 3.34 | 0.80 | **2.21** |
| Value | DP | 0.76 | 2.71 | 0.75 | 1.87 |
| Value | CP | 0.51 | 2.74 | 0.58 | 1.76 |
| Value | OCP | 0.78 | 2.78 | 0.74 | 1.88 |
| Quality | ROE | 0.54 | 1.82 | 0.62 | 1.61 |
| Quality | GP | 0.63 | 2.18 | 0.64 | 1.78 |
| Quality | OP | 0.59 | 2.21 | 0.31 | 1.08 |
| Quality | INV ASSET | −0.55 | −2.61 | −0.49 | −1.86 |
| Quality | INV BOOK | −0.40 | −1.89 | −0.17 | −0.62 |
| Quality | ACC | −0.09 | −0.68 | −0.17 | −0.77 |
| Quality | TOTAL ACC | 0.22 | 1.31 | 0.31 | 1.03 |
| Quality | NOA | 0.08 | 0.39 | 0.45 | 1.40 |
| Risk | VOL 1M | 1.08 | 3.77 | 1.07 | **2.72** |
| Risk | VOL 3Y | 0.90 | 2.99 | 0.84 | **2.10** |
| Risk | BETA | 0.36 | 1.18 | 0.55 | 1.30 |
| Risk | **IVOL** | 1.19 | 3.93 | 1.10 | **2.51** |
| Risk | **MAX** | **1.20** | **5.20** | **0.96** | **2.55** |
| Past ret | **MOM (12−1)** | 0.33 | 0.92 | 0.35 | 0.79 |
| Past ret | **RES MOM** | **0.66** | **3.36** | **0.59** | **2.11** |
| Past ret | CF MOM | 0.46 | 1.54 | 0.61 | 1.72 |
| Past ret | **REV 1M** | **1.42** | **4.35** | 0.80 | 1.88 |
| Past ret | RES REV 1M | 1.18 | 4.72 | 0.56 | 1.63 |
| Past ret | REV LT | 0.71 | 2.18 | 0.81 | 1.92 |
| Past ret | SEAS | 0.42 | 2.79 | 0.59 | 1.93 |
| Past ret | SEAS REV | 0.81 | 3.28 | 0.75 | 2.06 |
| Past ret | SEAS DIFF | 0.89 | 4.16 | 0.90 | **2.83** |
| Trading | TURN (250d) | 0.88 | 3.26 | 0.69 | 1.56 |
| Trading | **ABN TURN** | **1.72** | **5.50** | **1.33** | **3.22** |
| Trading | ILLIQ | 1.24 | 3.35 | 0.71 | 1.50 |

- **ABN TURN is the only trading anomaly significant on value-weighted portfolios.**
- **CAPM alphas for the risk variables**: VOL 1M 1.27 (4.27); VOL 3Y 1.10 (3.92); BETA 0.58 (2.02); IVOL 1.21 (3.85); **MAX 1.32 (5.69)**.
- Holdings: returns and t-stats **decay** as holding period lengthens, but **value, quality and risk are the most persistent** (VOL 3Y: 0.90 at 1 month → 0.85 at 12 months), while **momentum and seasonalities decay fastest**.

**Exact variable definitions (Appendix B):**
- **SIZE** = ln(end-of-month **total A-share** market cap) (B/H excluded — only A-shares are accessible to domestic investors).
- **BM** = book value (common shareholder equity − preferred, excl. minority interests) ÷ [shares outstanding **including B and H** × A-share price]. Negative book excluded.
- **EP** = total earnings (net profit excl. minority interest) ÷ same price proxy. **Negative earnings excluded.**
- **ROE** = total earnings ÷ book equity.
- **VOL 1M** = SD of daily returns over past 20 trading days.
- **BETA** = slope from regressing daily excess returns over past **250** days on the VW market portfolio.
- **IVOL** = SD of residuals from regressing daily excess returns over past **250** days on intercept + VW market + **size + value** factors.
- **MAX** = highest adjusted daily return over past 20 trading days, **two adjustments**: (i) daily returns on **suspended days set to −99**; (ii) **aggregate the daily returns after a stock hits the upper price limit** (e.g. +10% then +5% next day ⇒ MAX = 15%). *Without (ii) the 10% limit artificially deflates MAX and the sort cannot distinguish limit-hitters from non-hitters.*
- **MOM** = cumulative monthly return months t−12 to t−2.
- **RES MOM** = regress monthly excess returns over past 36 months on FF-3 (only stocks with a complete 36-month history); take residuals from the past 12 months excluding the most recent month; **scale by their own SD over the same period**; take the mean.
- **REV 1M** = cumulative return over past **20 trading days**.
- **RES REV 1M** = most recent month's residual return ÷ 36-month residual SD.
- **REV LT** = cumulative monthly return months **t−60 to t−13**.
- **TURN** = average daily share turnover over past **250** days (shares traded ÷ total shares outstanding).
- **ABN TURN** = average daily turnover over past **20** days ÷ average daily turnover over past **250** days.
- **ILLIQ** = average over past 20 days of `|ret| / dollar volume`.

**Break-even transaction costs (Section 4.7, Table 5):** turnover = sum of two-way turnover of the long and short legs (max 400%). Worked example: BM, 1-month: return 1.12%, turnover 76.67% (33.20% + 43.48%), break-even cost = 1.12/76.67 = **1.47%**; at 12-month holding turnover falls to 17.92% and break-even rises to **4.44%**. High-turnover anomalies: VOL 1M 265.43%, **REV 1M 355.26%**, RES REV 1M 361.93%, SEAS 366.76%, ABN TURN 254.16% ⇒ break-even as low as **0.12%** for SEAS. Benchmark: quoted/effective spreads in China 2009–2017 were **0.11%–0.37% per trade** (Gao et al. 2020).

**Size- and industry-neutral sorts (Section 5):** anomaly premiums largely unaffected by size; **profitability effects become MORE pronounced after size-neutralisation** (as in LSY); industry-neutral results track unconditional for value/volatility/turnover with **higher t-stats** (lower strategy volatility); **momentum falls** (industry momentum); **short-term reversal is somewhat STRONGER industry-adjusted**. Anomalies exist in small, mid and large caps but are **quantitatively larger for smaller stocks**.

**Long vs short leg (Table 7):** with VW portfolios against a VW index, on average **48%** of the excess return comes from the long leg. For the **risk** category the low-risk leg performs on par with the index and the **high-risk leg underperforms substantially** — the low-risk anomaly is largely a short-leg phenomenon. **REV 1M and ABN TURN are mostly driven by the short side.**

**Chinese market characteristics (Section 6):**
- **Short-sale constraints:** excess returns of the long and short sides are of **the same magnitude** ⇒ limits to arbitrage are **at most a partial** explanation of the anomalies.
- **SOE vs non-SOE:** average |excess return| **0.76%/mo (non-SOE) vs 0.79%/mo (SOE)** — "there does not seem to be much need for investors to treat SOEs differently than non-SOEs when it comes to factor investing."
- **Split-share reform (2007 break):** anomalies somewhat higher pre-reform but still substantially positive after (e.g. **BM 1.01%/mo, t=1.94** post-2007); several lose *statistical* significance largely because of reduced power in the shorter sub-sample.

### 2.2 Hou, Qiao, Zhang (2023) — 426 anomalies, 2000–2018

**Data & filters (Section 2.1):** Wind, SHSE + SZSE, **Jan 2000 – Dec 2018**. 681 firms (2000) → 2,465 (2018); 3,666 A-share stocks by end-2018 (incl. delisted). Three filters: (i) **drop firms public within past 6 months**; (ii) **drop 30% microcap firms (following LSY)**; (iii) **keep only firms with ≥75% of non-zero-volume trading days** in the recent sample periods (liquidity/data-quality + suspension control).

**Portfolio method (Section 2.2):** Jegadeesh–Titman **[k, m, n]**: information window [t−k+1, t]; sort into **deciles**; waiting period m (**m = 0 except for momentum**); hold [t+m+1, t+m+n] with n = 1, 6 or 12; **VW using previous month-end market cap**, plus EW.

**Price-limit adjustment (Section 2.2):**
> "due to daily price limit of 10%, we use the **average of the five highest daily returns** in one month to construct the maximum daily return, rather than using one highest daily return in one month as in the original procedure. We also exclude the stock in the portfolio if it is suspended with zero trading volume during the portfolio sorting date."

**Multiple-testing hurdle (Section 2.3):** double bootstrap of Harvey–Liu (2020), p₀ = 15% true anomalies ⇒ **t-cutoff = 2.85** at 5%. (BH gives 3.16; BSW 2.55; BHY too conservative.)

**Headline results (Section 3.1.1, Table 3):**
- **VW portfolios, single test (|t| > 1.96):** **98 of 426 (23%)** have significant long-short raw returns. CAPM alphas: 97. **CH-3 alphas: 16** (12 trading-related, 4 accounting; **7 from liquidity**). CH-4 alphas: 15.
- **VW, multiple test (|t| > 2.85): 27 anomalies pass.** CAPM alphas still significant. **CH-3: only 2 survive — both liquidity. CH-4: ZERO survive.**
- **EW, single test: 175 of 426 (41%).** CH-3: 59 significant (30 liquidity, 12 accounting). CH-4: 54.
- **EW, multiple test: 94 pass.**
- Magnitudes of the 98 significant VW long-short portfolios: **0.29% – 1.53% per month**; the 27 that pass 2.85: **0.54% – 1.53% per month**.

**Selected anomaly numbers (Section 3.1.2):**

| Anomaly | Raw %/mo (t) | CAPM α (t) | CH-3 α (t) | CH-4 α (t) |
|---|---|---|---|---|
| **turn1-1** (1-mo avg share turnover, **float A shares**) | **1.43 (3.17)** | 1.54 (3.39) | **1.14 (2.63)** | 0.37 (0.92) |
| **vturn1-1** (variation of share turnover) | **1.58 (3.45)** | — | **1.38 (3.38)** | 0.71 (1.82) |
| Dollar trading volume (1-mo avg) | 1.07 | 1.08 (2.51) | 0.49 (2.24) | 0.17 (0.89) |
| **IVOL** (SD of CH-3 residuals, daily, 1 month) / total vol | **0.74 – 1.19** (all t > 2) | significant | **not significant** | not significant |
| Total skewness (daily, 1 month) | 0.65 | 0.81 (t>2.7) | 0.67 (2.02) | 0.51 (1.30) |
| Idiosyncratic skewness (CH-3 residuals) | 0.83 | 0.65 (t>2.7) | **0.92 (2.81)** | **0.88 (2.16)** |
| Beta (Frazzini–Pedersen 1-yr daily; Dimson 1-mo daily) | 0.93 – 1.10 | — | explained | FP beta explained; Dimson beta not |
| **Momentum change** (recent 6m cumulative − first 6m cumulative within the year, 6-mo horizon) | 0.57 (2.79) | — | — | — |
| **Residual momentum** (11-mo CH-3 cumulative residuals over prior 36m ÷ SD) | 0.68 (t > 2.70) | — | — | — |
| **Industry momentum** (1-mo VW return of the 30% **biggest** firms in an industry) | 0.63 (2.00) | — | — | — |
| **Long-term reversal** (4-yr cumulative daily returns) + **MAX** (avg of 5 highest daily returns) | **0.81 – 0.89** (t > 2) | — | **insignificant** | LT-reversal α not explained by CH-4; **MAX α IS explained by CH-4** |
| Seasonality Ra25 (avg of 2–5 yrs' lagged **same**-month returns) | 0.65 (t > 2.30) | not explained | explained | explained |
| Seasonality Rn25 (avg of 2–5 yrs' lagged **other**-month returns) | 0.93 (t > 2.30) | not explained | explained | explained |

**Momentum failure rate:** "we construct **30 momentum strategies**, and **4 of them are statistically significant**, corresponding to a **13.3%** significance rate. In contrast, the significance rate of momentum anomalies in the United States by Hou, Xue and Zhang (2019) is **63.2%**."

**HQZ's stated mechanism for momentum failure in China:**
> "Daniel and Moskowitz (2016) show that momentum crashes in panic states, following market declines and high market volatility. **The majority of momentum does not work in China possibly due to a long period of bear market conditions since 2000.**"

**Why liquidity anomalies are so strong in China (Section on Liquidity):** daily trading volume in China is mostly retail; retail "trade excessively to chase past good performance", their excessive purchases create price bubbles in winner stocks and hence bad future returns; US retail investors mostly pursue contrarian strategies and the US market is institutionally dominated.

**Profitability:** 39% of profitability anomalies significant on raw long-short returns (US: 44.3%) — **but CH-3 reduces alphas to between −0.12% and 0.20% with t-stats below 1**.

---

## 3. Momentum / reversal in China

### 3.1 Is there a strong short-term reversal? YES — and here is the magnitude and holding period

| Source | Definition | Magnitude | t |
|---|---|---|---|
| LSY (2019) | 1-month (20-day) return, VW deciles 1−10, CAPM α | **1.49%/mo raw α; 18%/yr annualised** | 3.06 |
| LSY (2019) | same, size-neutral | **1.65%/mo** | 3.68 |
| LSY (2019) | mid-price robustness (2012–2016) | 2.21%/mo (vs 2.15% close) | — |
| JSZ (2021) | REV 1M = past 20 trading days, EW | **1.42%/mo** | 4.35 |
| JSZ (2021) | same, VW | 0.80%/mo | 1.88 |
| JSZ (2021) | RES REV 1M (residual reversal), EW | 1.18%/mo | 4.72 |
| JSZ (2021) | REV LT (t−60 → t−13), VW | 0.81%/mo | 1.92 |
| Kang–Liu–Ni (2002) | 1-week formation/1-week holding contrarian, EW | **1.86%/week** | — |
| Kang–Liu–Ni (2002) | same **skipping 1 week** | **0.08%/week** | — |
| HQZ (2023) | 4-year cumulative daily returns (long-term reversal) | 0.81–0.89%/mo | >2 (dies under CH-3) |

**Best holding period: one month (20 trading days).** The effect decays fast: JSZ find the reversal long-short falls to **0.23%/mo at 6-month** and **−0.06%/mo at 12-month** holding (EW). LSY's own reading is that reversal at the 1-year window is only **6% annualised and insignificant (t=0.90)**.
**Critical caveat:** Kang–Liu–Ni show the **weekly** contrarian profit is mostly **bid–ask bounce** (1.86%/wk → 0.08%/wk after skipping a week). LSY's mid-price check shows the **monthly** reversal is *not* bounce.
**Residual reversal adds nothing in China** (JSZ: "The added value of residual reversal over conventional short-term reversal is absent in our sample, possibly because conventional reversal already has relatively high excess returns"). **Residual momentum, by contrast, does add (0.66% EW / 0.59% VW).**

### 3.2 Why momentum fails in China — mechanisms on the record

**(a) Chui, Titman, Wei (2010) — cultural individualism.** 41 countries, 6-month formation, skip 1 month, 6-month holding, **top/bottom thirds**, EW, overlapping, USD. Screen: drop stocks below the **5th percentile** market cap each month; keep returns only within the **1st–99th percentile** each month; require ≥8-month history, ≥30 stocks per country-month, and ≥5 years of momentum-portfolio history.
- All but five countries (Argentina, Korea, Philippines, Taiwan, Turkey) have positive momentum; significant in 21.
- **China (1993:10–2003:06): Winner 1.812 (t=1.13), Loser 1.069 (t=0.76), W−L = 0.743%/mo (t=1.99).** So China shows a **weakly positive but individually insignificant-legs** momentum in this early sample.
- **China's Hofstede individualism index = 20** (one of the lowest in the sample).
- Country-average momentum (Mar 1981–Jun 2003): **0.54%/mo (t=6.48)**; country-neutral **0.58%/mo (t=4.44)**.
- **Indv-sorted: Low 0.308 (2.22), Middle 0.492 (4.75), High 0.838 (6.76); High−Low = 0.530%/mo (t=3.42).**
- Fama–MacBeth: coefficient on Indv = **0.010–0.012**, significant, robust to legal origin, anti-director rights, corruption (Cpix), accounting standards and earnings-management risk.
- **Mechanism:** individualism ⇒ overconfidence / self-attribution ⇒ momentum. **CTW also find momentum in Chinese B-shares**, which JSZ summarise as being related to **higher institutional ownership** (B-shares are restricted from domestic retail).

**(b) Cheema & Nartea (2017) — market dynamics, not individualism.** Sample: all A-shares SHSE+SZSE, **Jan 1995 – Aug 2016**, CSMAR, 240 firms at start → 1,755 at end, **317,478 firm-month observations**. Monthly returns winsorised at +100% / −95% (following CTW 2010). 6-month formation (t−6 to t−1), **skip 1 month**, **quintiles**, hold 3/6/9/12 months, overlapping.
- **Unconditional EW momentum (P5−P1):** k=3 **0.29 (1.05)**; k=6 **0.25 (0.99)**; k=9 **0.06 (0.28)**; k=12 **−0.07 (−0.32)**. VW: k=6 raw 0.36 (1.21) but **FF-3 α 0.58 (2.16)**; k=9 α 0.46 (1.97).
- **IV-weighted momentum (overweights high-IV stocks) is insignificant for every holding period and generally LOWER than EW/VW.** ⇒ the positive momentum–IV relation documented for the US by Arena et al. (2008) is **absent** in China; consistent with McLean (2010).
- **Cross-sort on IV:** EW low-IV momentum 0.46% (α 0.77%) vs high-IV 0.23% (α 0.42%); spread −0.23% (−0.25%). VW: low-IV 0.71% (α 0.90%) vs high-IV 0.31% (α 0.71%); spread −0.40% (−0.19%).
- **Cross-sort on SIZE:** small and medium terciles → small and insignificant momentum; **the BIG tercile has relatively large and significant momentum.** Footnote 7: *"we find significant momentum returns for all the holding periods once we exclude stocks that are below the median market capitalization."*
- **Reason:** "The lack of momentum is generally attributed to high turnover in Chinese markets shortening the period over which cycles of overreaction and subsequent correction occurs" (JSZ's reading). Cheema–Nartea's own conclusion: cross-country momentum differences are driven by **market dynamics** rather than by **levels of individualism**.
- **Transaction-cost context:** Mei, Scheinkman & Xiong (2009) document **0.40% per trade** in China, plus **0.10% stamp tax per trade**.

**(c) Dissecting Momentum in China (Liu, Tan, Xu, Yuan, Zhu; WP Feb 2025) — a "tug-of-war".** 6-month formation, skip month t, monthly rebalance, quintiles, **VW**; universe excludes the bottom 30% of A-shares and stocks listed <6 months.
- Classic momentum long-short: **0.22%/mo (t=0.40)** — no momentum.
- Decompose past return into **news-day** and **non-news-day** components: past-news-return quintile spread earns **+0.45%/mo (t=2.94)** on the news component but **−0.33%/mo (t=−1.37)** on the non-news component — the two roughly cancel.
- A text-based **"good news ratio"** long-short (insulated from the attention-driven buying pressure) earns **0.78%/mo VW (t=2.47)** raw and **1.02%/mo (t=2.96) under the CH-4 model**, split evenly into 0.38% on news days (t=2.44) + 0.38% on non-news days (t=2.00). In a subsample: 0.97%/mo (t=2.74), ~25% stronger.
- **Mechanism:** news-day returns are contaminated by temporary **retail attention-driven buying pressure** (reverses later); non-news-day reversals are **mispricing corrections by institutions**. The identical decomposition in the **US** produces positive returns on **both** news and non-news days — the opposite pattern.
- **Implication for a daily-bar-only strategy:** the decomposable, implementable part is the *non-news-day reversal*; the persistent part requires news text.

### 3.3 Conditional momentum that DOES work in China (ranked by implementability)

1. **Residual momentum** — JSZ: EW **0.66%/mo (t=3.36)**, VW **0.59%/mo (t=2.11)**, constructed as the mean of past-11-month FF-3 (or CH-3) residual returns from a 36-month rolling regression, each scaled by the residual SD over the same window. HQZ: **0.68%/mo (t > 2.70)** using CH-3 residuals.
2. **Large-cap-only momentum** — Cheema–Nartea fn 7: exclude stocks below the **median** market cap ⇒ significant momentum at **all** holding periods (3/6/9/12 months).
3. **Market-state-conditional momentum (DN/DN)** — Cheema–Nartea Table 4 Panel A (China, 1995–2016):

| Market state | W−L %/mo | t | N months |
|---|---|---|---|
| UP/UP | −0.12 | −0.39 | 79 |
| UP/DN | −0.07 | −0.14 | 53 |
| DN/UP | −0.69 | −1.42 | 58 |
| **DN/DN** | **+1.45** | **2.93** | **58** |

   State definition: lagged 12-month VW market return (t−11 to t) and the subsequent month t+1 return; both non-negative = UP/UP, both negative = DN/DN, mixed = transitions. Note the DN/DN state occurs only **58 of 248 months**. Sub-sorting DN/DN by IV: Low IV **2.26 (4.20)**, Med IV **1.70 (3.37)**, High IV **1.33 (2.83)**, High−Low **−0.93 (−3.06)**.
4. **Momentum change** (HQZ): cumulative daily returns of the most recent 6 months minus the first 6 months of the year → **0.57%/mo (t=2.79)** at the 6-month horizon.
5. **Industry lead-lag momentum** (HQZ): 1-month VW return of the portfolio of the **30% biggest firms within an industry** → **0.63%/mo (t=2.00)**. Buildable from your industry classification + market cap.
6. **Connected-firm momentum (CF MOM)** — JSZ: 0.46% EW (t=1.54) / 0.61% VW (t=1.72), **not significant**; collapses to ~zero after industry adjustment. Requires IBES analyst coverage. **Skip.**
7. **News-based momentum** — see 3.2(c). Requires news text.

### 3.4 Momentum crashes (Daniel & Moskowitz 2016) — the US mechanism HQZ import to China
- WML behaves like a **written call option on the market**, most valuable after bear markets. Optionality arises because the **loser** portfolio's market beta **rises** in bear markets and its value then behaves like a call.
- In bear markets the WML alpha is **2.04% per month** but the ex-ante market risk is significantly lower; **the WML beta falls by 0.518 (t = −28.4) in panic states**.
- Panic state = **bear market AND high ex-ante volatility**; defined with an ex-ante bear-market indicator `I_{B,t−1}` = 1 if the cumulative CRSP VW index return in the relevant window is negative, interacted with estimated market variance `I_B σ²`.
- **The optionality comes predominantly from the LOSER portfolio** and is significant **only in bear markets**; in bull markets the loser portfolio shows no significant optionality.
- The same asymmetry exists in **non-US equity markets** (bear-market betas lower with t-stats ≈ −7; the UK is weaker), but the optionality outside the US is often of the right sign yet statistically insignificant.
- **China application:** HQZ (2023) explicitly attribute China's missing momentum to "a long period of bear market conditions since 2000"; Cheema–Nartea's DN/DN result is the direct China analogue.

### 3.5 Sentiment conditioning of anomalies in China
- **Cheema, Nartea, Man (2020)** show the MAX premium is concentrated in high-sentiment periods (numbers in §4).
- **"Investor Sentiment and Chinese A-Share Stock Markets Anomalies"** builds a **Baker–Wurgler-style top-down index**: first principal component of four **orthogonalised** sentiment proxies — `PVOL` (volatility premium), `NIPO` (log ratio of the annual number of IPOs), `RIPO` (IPO underpricing), `TURN` (log ratio of annual total market turnover) — each orthogonalised against **CPI growth** and **industrial-production growth**. Tests **13 anomalies** (Size, Sigma, BM, SG, O-score, Issue, Accruals, Opa, Profit, GA, ROA, ROE, ITA) on **value-weighted** decile long-short portfolios, 1997–2009.
  - **9 of 13 anomalies are significantly affected by lagged sentiment.** All long-short spreads are higher following high sentiment; the high-minus-low-sentiment spread is optimal at **16.9 bps**; except for BM and ITA all spreads are significantly larger after high sentiment.
  - **The SHORT leg drives it**: short-leg high-minus-low-sentiment returns range from **−244.8 bps to −97 bps**, and the relation between short-leg return and lagged sentiment is **significantly negative** — i.e. high sentiment ⇒ the short leg becomes much more profitable.
  - Consistent with Stambaugh, Yu & Yuan (2012) in the US: mispricing is predominantly **over**pricing, and the anomalies that are hardest to arbitrage (short leg) load on sentiment.
  - **Partial implementability for a daily-bar-only strategy:** `PVOL` and `TURN` are computable from your data; `NIPO` and `RIPO` need an IPO calendar (available from baostock/tushare in principle).

---

## 4. MAX effect and the IVOL puzzle in China — is the sign the same or reversed?

### 4.1 US baseline (Bali, Cakici & Whitelaw 2011)
- NYSE/AMEX/Nasdaq, **Jul 1962 – Dec 2005**; decile portfolios sorted on MAX = maximum daily return in the previous month; reformed monthly.
- **VW decile 10 − decile 1 raw = −1.03%/mo (t = −2.83); FF-Carhart 4-factor α = −1.18%/mo (t = −4.71).**
- **EW raw = −0.65%/mo (t = −1.83); α = −0.66%/mo (t = −2.31).**
- Using the average of the **top N** daily returns, N = 5: VW raw **−1.23%**, α **−1.32%** — *stronger* than N = 1.
- Decile shape: deciles 1–7 flat at **1.00%–1.16%/mo**; decile 8 **0.86%**, decile 9 **0.52%**, decile 10 **−0.02%**. Average MAX within deciles 8/9/10 = **9% / 12% / 24%**.
- BSW's claim: **including MAX reverses the negative IV-return relation of Ang et al. (2006, 2008)** — i.e. in the US, MAX subsumes IVOL.

### 4.2 China — the sign is THE SAME as the US (high MAX → low returns). It is NOT reversed.

| Source | Sample | Definition | Magnitude | t |
|---|---|---|---|---|
| **Cheema, Nartea, Man (2020)**, IRF 20(4) | Jun 1997 – Jun 2017, DataStream, 651 → 3,041 A-shares; deciles | MAX = max daily return in month t; P1−P10 (long low MAX) | **EW 1.03%/mo; VW 0.66%/mo** | significant |
| Cheema et al. — US comparison in same paper | — | BSW US over 1926–2015 | EW 1.02%; VW 0.62% | — |
| **JSZ (2021)** | 2000–2019, CSMAR, deciles | MAX = highest adj. daily return, past 20 days, limit-aggregated | **EW 1.20%; VW 0.96%**; CAPM α **1.32%** | **5.20 / 2.55 / 5.69** |
| **Nartea, Kong, Wu (2017)**, JBF 76 | Jan 1994 – Dec 2013, DataStream, 185 → 2,525 firms, avg 1,279/mo, 305,674 firm-months; **tertiles** | MAX = max daily return in past calendar month; hold 1 month | EW raw H−L **−0.99% (t=−1.06)**, VW raw **−0.73% (t=−0.83)** — **raw spreads insignificant**; **EW FF-3 α −0.95% (t=−3.23)**; **VW FF-3 α −0.60% (t=−2.49)** | see left |
| Nartea et al. — 3-month holding | same | same | EW/VW raw spreads still insignificant; **α spreads significantly negative**; FM univariate MAX **−0.3653 (t=−4.35)**, multivariate **−0.1604 (t=−1.97)** | — |
| Nartea et al. — 6-month holding | same | same | α spreads negative and highly significant; FM univariate **−0.6138 (t=−7.44)**, multivariate **−0.3522 (t=−2.59)** | — |
| Nartea et al. — **MAX(5)** | same | avg of 5 highest daily returns | FM univariate **−0.3978 (t=−3.86)**; but **VW α spread NOT significant** | — |
| **HQZ (2023)** | 2000–2018, Wind | MAX = **average of the five highest daily returns** in the month (price-limit adjustment) | **0.81–0.89%/mo** long-short raw | > 2 |
| **LSY (2019)** | 2000–2016, WIND | MAX = highest daily return, past 20 days | R̄ 0.57; **CAPM α 0.81%/mo**; β = −0.36 | t(R̄) 1.26, **t(α) 2.02**, t(β) −6.39 |
| SCIRP (2023) | China A-shares, deciles | MAX = max daily return, prior month | **FF-3 α(high MAX) = −0.95%** | *** |

**Conclusion on the sign:** the MAX effect in China is **present with the same sign as the US** and, on an **equally-weighted** basis, of **comparable magnitude** (China EW 1.03–1.20%/mo vs US EW 0.65–1.02%/mo). On a **value-weighted** basis it is somewhat **weaker** (China VW 0.66–0.96%/mo vs US VW 1.03%/mo), and at a **1-month holding period in raw (non-alpha) terms** it can be **statistically insignificant** (Nartea et al.). **It is not reversed.**

### 4.3 The IVOL puzzle in China — also the SAME sign as the US
- **JSZ (2021):** high IVOL underperforms. **EW 1.19%/mo (t=3.93); VW 1.10%/mo (t=2.51); CAPM α 1.21%/mo (t=3.85).**
- **HQZ (2023):** "Chinese firms with higher volatility also underperform firms with lower volatility, which confirms the finding in Ang, Hodrick, Xing and Zhang (2006)." Long-short IVOL raw returns **0.74%–1.19%/mo with t always above 2**; CAPM α significant, but **CH-3 and CH-4 α not significant**.
- **LSY (2019):** 1-month volatility CAPM α **1.03%/mo (t=2.31)**, β = −0.34 (t=−5.55); **CH-3 α 0.23% (t=0.44)** — the volatility anomaly is *captured by CH-3* because low-volatility stocks load on VMG (t=3.86).
- **Nartea, Wu & Liu (2013)** (JIFMIM 27, 137–160) document a **highly significant anomalous negative IV effect in China** — I could not obtain the full text; this is Nartea–Kong–Wu's own description of it (reported in their Table 10 discussion).

### 4.4 Does MAX subsume IVOL in China? **The evidence conflicts — do not assume the US result.**
- **Nartea, Kong & Wu (2017):** "the **negative IV effect survives in a multivariate setting that includes MAX**. This indicates that **both the MAX and IV effects appear to coexist in China** and does not support the suggestion of Bali et al. (2011) that MAX is the true effect for which idiosyncratic volatility is only a proxy." Their bivariate sorts also show the 1-month MAX effect **can be explained by momentum** (EW α spread −0.36%, t=−1.61) and **by IVOL** (VW α spread −0.31%, t=−1.67).
- **SCIRP (2023):** the opposite — in Fama–MacBeth regressions, IVOL's coefficient is significantly negative **only when MAX is omitted**; after including MAX, MAX remains negative and significant (coefficient magnitude rises from 0.325 to 0.328) and IVOL loses significance. Conclusion: "it is the existence of the MAX effect that [drives] the negative future returns... MAX is the fundamental reason for negative future returns of stocks."
- **Practical reading:** in China, expect **MAX and IVOL to be highly correlated and partly redundant**, but not perfectly substitutable. Keep both; **orthogonalise one against the other** and check which carries the incremental information.

### 4.5 The MAX effect is conditional on investor sentiment (highest-value China-specific finding)
**Cheema, Nartea & Man (2020)**, sentiment index `STM_t = 0.544·PE_t + 0.588·TO_t + 0.598·IA_t` (PE ratio, turnover ratio, number of newly opened individual investor accounts; first PC explains **76.79%** of variance; **levels** index):

| Condition | EW MAX premium | t | 
|---|---|---|
| High sentiment | **+1.85%/mo** | 3.83 |
| Low sentiment | +0.22%/mo | 0.51 |
| **High − Low** | **+1.63%/mo** | **2.30** |
| FF-adjusted α, High − Low | **+2.37%/mo** | **4.21** |

- **VW** premium is also significant **only** following high sentiment: **1.62%/mo (t=2.27)**.
- Time-series regression of the EW MAX premium on lagged sentiment: coefficient **0.80 (t=2.06)** (a competing specification gives a lagged-sentiment coefficient with t=5.66).
- Robustness: EW MAX premium **0.90%/mo (t=2.94)** excluding 2008–09; **1.20%/mo (t=3.39)** excluding 2016; in 1992–1996 the EW premium was **6.46%/mo** and the full-sample-1992 EW premium is 2.42%/mo (t=1.94).
- **Interpretation:** the MAX premium is driven by **investor optimism during high-sentiment periods**, not by the attention-grabbing property of high-MAX stocks in low-sentiment periods.
- **Reconciliation with Cheon & Lee (2017)**, who report an ~**0.01%/mo** MAX premium for China and a **1.96%/mo** premium for the US (highest individualism index): Cheema et al. show that near-zero China figure is not robust — using the same DataStream data and similar decile portfolios they find 1.03%/mo EW.

---

## 5. IMPLEMENTABLE TAKEAWAYS

Priority-ordered. Each item states the change, the exact parameter values from the papers, and the evidence. "Your factor" refers to the factor in your current list that it touches.

### P0 — Structural changes that affect everything

**T1. Build CH-3 and CH-4 as your benchmark, and residualise your alpha against them.**
*Evidence:* HQZ (2023) test 426 anomalies in China. On **value-weighted** portfolios, 98 have significant raw long-short returns and 97 have significant CAPM alphas — **but only 16 survive CH-3**, and after the **t > 2.85** multiple-testing hurdle **only 2 survive CH-3 (both liquidity) and ZERO survive CH-4**. LSY: CH-3 leaves 8 of 10 CAPM-significant anomalies with average |α| 0.30%/mo vs FF-3's 0.84%.
*Action:* Compute CH-3 exactly as in §1.4 (top 70% by **total-share** market cap; 2 size groups at the median; 3 EP groups at the 30th/70th percentiles; 6 VW portfolios; SMB = ⅓(S/V+S/M+S/G) − ⅓(B/V+B/M+B/G); VMG = ½(S/V+B/V) − ½(S/G+B/G); MKT = VW top-70% minus the 1-year deposit rate; **keep negative-EP stocks as growth**). Report every factor's CH-3/CH-4 alpha, not just its raw IC. Then decide which of your 17 factors carry incremental alpha.

**T2. Add CH-4's fourth factor — abnormal turnover (PMO) — and neutralise on it.**
*Evidence:* LSY: abnormal turnover = past month's turnover ÷ past year's turnover; PMO built exactly like VMG but long-low/short-high, with the new SMB = average of the EP-neutralised and turnover-neutralised SMBs. PMO and the new SMB average **12% and 11% annualised**. CH-4 explains the two anomalies CH-3 cannot (reversal α 0.49%, t=0.87; abnormal turnover α 0.00%, t=−0.01) and lifts the GRS p-value from 0.15 to 0.41.
*Your factors:* this is your **TURN_SHOCK**. Make sure it is defined **exactly** as (20-day average daily turnover) ÷ (250-day average daily turnover), **float-share based** (HQZ's turn1-1 uses **float A shares**; JSZ/LSY use total shares — test both), and that it is size-neutralised like VMG.

**T3. Tighten the universe filters to the literature's standard.**
*Evidence / exact parameters:*
- Drop stocks public within the past **6 months** (LSY, JSZ, HQZ — all three).
- Drop stocks with **< 120 trading days in the past 12 months** AND **< 15 trading days in the most recent month** (LSY); equivalently, JSZ's **< 5 suspended days in the past month** and **< 120 suspended days in the past year**; HQZ's **≥ 75% non-zero-volume trading days**.
- **Exclude stocks suspended on the rebalance date** (JSZ; HQZ).
- **Exclude ST and PT stocks.** JSZ do this explicitly because ST/PT stocks carry a **5% daily price limit** instead of 10%, which distorts volatility, MAX and reversal measurement. **You are not currently doing this** and it is the cheapest single upgrade.
- Keep the bottom-30% market-cap screen — but note LSY/HQZ/JSZ all rank on **total A-share** market cap, not float. If you only have float cap, test both.

### P1 — Measurement fixes to factors you already own

**T4. MAX5: fix the price-limit and suspension handling, and check the holding horizon.**
*Evidence:*
- **HQZ:** "due to daily price limit of 10%, we use the **average of the five highest daily returns** in one month… rather than using one highest daily return" — i.e. your MAX5 construction is exactly the literature's recommended China adaptation. Keep it.
- **JSZ:** before ranking, (i) set daily returns on **suspended days to −99** (JSZ) or **delete zero-return suspension days** (JSZ's alternative wording: "we explicitly delete 'zero return' observations on days that a stock is suspended, which otherwise could be the highest return during a negative month"), and (ii) **aggregate returns across consecutive limit-up days** (if a stock hits +10% and returns +5% the next day, MAX = 15%). Chevron: JSZ attribute the aggregation fix to Cheema et al. (2020).
- **Nartea et al.:** the China MAX effect on **raw** returns is insignificant at a **1-month** holding but the FF-3 **alpha** spread is already significant at 1 month (−0.95% EW, t=−3.23; −0.60% VW, t=−2.49) and **strengthens at 3 and 6 months** (FM univariate coefficient −0.365 at 3 months, −0.614 at 6 months). They attribute this to **price-limit-induced lags of up to six months**.
- *Action:* (a) aggregate limit-hit returns before computing MAX5; (b) exclude suspension days rather than treating them as zero; (c) test MAX5 with a **3-month** (and 6-month) signal decay in addition to 1-month; (d) because high-MAX stocks are also high-beta (LSY: β = −0.36, t=−6.39) and low-BM, **check MAX5's alpha after CH-3** — LSY finds CH-3 leaves only α = 0.27% (t=0.65) and HQZ finds CH-3 kills the MAX anomaly entirely.

**T5. Gate MAX5 (and the whole book) on a market sentiment / turnover state.**
*Evidence:* Cheema–Nartea–Man: EW MAX premium **+1.85%/mo (t=3.83) after HIGH sentiment vs +0.22% (t=0.51) after LOW**; the FF-adjusted gap is **+2.37%/mo (t=4.21)**. VW premium significant only after high sentiment (**1.62%/mo, t=2.27**).
*Action:* Build a China sentiment levels index. The literature's standard is the Baker–Wurgler **top-down** approach: first principal component of four proxies — `PVOL` (volatility premium), `NIPO` (log number of IPOs), `RIPO` (IPO underpricing), `TURN` (log aggregate market turnover) — each **orthogonalised** against CPI growth and industrial-production growth. Cheema et al. instead use `STM = 0.544·PE + 0.588·TO + 0.598·IA` (first PC explains 76.79%). **Two of the four (PVOL, TURN) and PE/TO are computable from OHLCV + turnover + float cap alone**; `IA` (new individual accounts) and `NIPO/RIPO` need external data. Start with the turnover/volatility-only version and scale risk by it.

**T6. IVOL: redefine against CH-3 residuals, and treat it as partly redundant with MAX5.**
*Evidence:* JSZ: **IVOL = SD of residuals from regressing daily excess returns over the past 250 days on intercept + VW market + size + value factors**; China long-short EW 1.19% (t=3.93), VW 1.10% (t=2.51), CAPM α 1.21% (t=3.85). HQZ: IVOL = SD of residuals of daily excess returns on **CH-3** over one month; raw 0.74–1.19%/mo, all t > 2; **CH-3/CH-4 alphas insignificant**. Nartea et al.: MAX does **not** subsume IVOL in China (both survive multivariate Fama–MacBeth); SCIRP (2023): MAX **does** subsume IVOL. LSY: the volatility anomaly's CH-3 α is 0.23% (t=0.44) because low-vol stocks load on **value**.
*Action:* (a) compute IVOL off CH-3 residuals, not the market model alone; (b) run MAX5 and IVOL through a joint Fama–MacBeth regression and drop the one that loses significance; (c) do not expect IVOL to survive CH-3.

**T7. Reversal: keep it as your highest-conviction signal, and adopt LSY's daily-sliced rebalancing.**
*Evidence:* LSY: 1-month reversal CAPM α **1.49%/mo (t=3.06)**, size-neutral **1.65% (t=3.68)**; "loser minus winner" at 18% annualised (t=2.96); mid-price robustness shows it is **not** bid–ask bounce. JSZ: EW 1.42%/mo (t=4.35). HQZ: liquidity/past-return anomalies are the ones that survive CH-3.
*Exact recipe (LSY Appendix A.2):* sort **each day** on the most recent **20-trading-day** cumulative return; rebalance a **one-fifth slice** held for **5 trading days**; average across the five slices; compound the daily returns to a monthly figure. This is the literature's answer to the **355% monthly turnover** JSZ measure for REV 1M.
*Caveat:* for any **weekly** reversal, **skip one week** — Kang–Liu–Ni show the 1-week contrarian profit collapses from **1.86%/week to 0.08%/week** once you skip a week (bid–ask bounce).
*Also:* **residual reversal adds nothing** in China (JSZ) — do not spend effort there.

**T8. REV_LT: use the exact 60→13 month window and value-weight it.**
*Evidence:* JSZ: REV LT = cumulative monthly return months **t−60 to t−13**; VW 0.81%/mo (t=1.92), EW 0.71% (t=2.18). HQZ: 4-year cumulative daily returns, 0.81–0.89%/mo (t > 2) but **CH-3 explains it away**; the alpha is **not** explained by CH-4. LSY: past returns over windows of 1, 3, 6, 12 months **and 5 years** all negatively predict, in monotonically weakening magnitude.
*Action:* keep REV_LT (skip the most recent 12 months), VW, and expect it to be absorbed by CH-3.

**T9. TURN / ILLIQ: rebuild around the turnover family, which is where China's alpha is.**
*Evidence (VW, 1-month holding, JSZ 2021):*
- **ABN TURN** (20d/250d) EW 1.72% (t=5.50), **VW 1.33% (t=3.22)** — the **only** trading anomaly significant VW.
- TURN (250d) EW 0.88% (t=3.26), VW 0.69% (t=1.56).
- ILLIQ EW 1.24% (t=3.35), VW 0.71% (t=1.50).
HQZ: **turn1-1** raw 1.43% (t=3.17), CH-3 α **1.14% (t=2.63)**, CH-4 α 0.37%; dollar-volume version raw 1.07%, CH-3 α 0.49% (t=2.24).
*New factor to add:* **vturn1-1 = the variation (dispersion) of daily share turnover over the past month**, long low / short high. HQZ: raw **1.58%/mo (t=3.45)**, **CH-3 α 1.38% (t=3.38)**, CH-4 α 0.71% (t=1.82) — **the strongest CH-3 survivor among all 426 anomalies tested**. You do not currently have a turnover-*stability* factor; this is the highest-expected-value single addition.
*Also note:* JSZ find the **turnover level** anomaly is significant on the **VW** portfolio only as ABN TURN, so prioritise TURN_SHOCK over raw TURN.

**T10. SIZE: treat it as an exposure to hedge, not a standalone alpha.**
*Evidence:* LSY SMB 1.03%/mo (t=3.25), 2000–2016. JSZ VW size spread 0.64%/mo (t=1.21) over 2000–2019 and: *"The last three years of our sample have seen substantially negative returns for the size factor."* Without the microcap screen the size return is 1.56% (t=2.67) — i.e. the raw "size premium" is largely a shell premium (LSY). LSY: including shells raises SMB to 1.36% but breaks factor pricing.
*Action:* keep SIZE as a risk control (neutralise or hedge), expect little standalone alpha post-2016, and **never** relax the bottom-30% screen to capture it.

**T11. Industry handling: neutralise reversal and the risk factors; do not bother for value.**
*Evidence:* JSZ — industry-neutral results track unconditional for value/volatility/turnover but **t-stats rise** (lower strategy volatility); **momentum falls** (industry momentum exists); **short-term reversal is somewhat STRONGER industry-adjusted** (consistent with Hameed–Mian 2015). LSY: 37 Shenyin-Wanguo industries; CH-3 explains **82%** of an industry's return variance vs 72% for the market alone.
*Your factor:* your **COMMOVE** is an industry-comovement factor. Note it is **not in any of these papers' 32/426/14 anomaly lists** — it is untested in this literature. Also note industry-neutralisation is the natural way to make COMMOVE and your industry dummies non-redundant.

### P2 — Reduce or drop

**T12. Do not build momentum; use residual momentum and industry momentum instead.**
*Evidence:* HQZ: only **4 of 30** momentum strategies significant (**13.3%**) vs **63.2%** in the US. JSZ: MOM (12−1) EW 0.33% (t=0.92), VW 0.35% (t=0.79). Cheema–Nartea: unconditional EW momentum 0.25%/mo (t=0.99) at k=6.
*Replacements that DO work:*
- **RES_MOM** (you have it): JSZ EW **0.66% (t=3.36)**, VW **0.59% (t=2.11)**; HQZ **0.68%/mo (t>2.70)**. Construction: 36-month rolling regression of monthly excess returns on **CH-3** factors, take the past **11** months' residuals, scale each by the residual SD over the same window, take the mean.
- **Large-cap-only momentum:** Cheema–Nartea fn 7 — exclude stocks below the **median** market cap ⇒ significant momentum at **all** holding periods (3/6/9/12 months).
- **Market-state gating:** +1.45%/mo (t=2.93) in the **DN/DN** state, but this is only 58 of 248 months — use it as an overlay switch, not a standalone factor.
- **Industry momentum:** 1-month VW return of the industry's 30% biggest firms → 0.63%/mo (t=2.00) (HQZ). Directly buildable from your industry field.
- **Momentum change:** recent-6-month minus first-6-month cumulative return → 0.57%/mo (t=2.79) at a 6-month horizon (HQZ).

**T13. De-prioritise accruals, investment, BETA and illiquidity as alpha sources.**
*Evidence:* LSY — investment α ≤ 0.22%/mo and accruals α ≤ 0.42%/mo are **not significant** in China (unlike the US); illiquidity is not quite significant (α up to 0.83%/mo) and CH-3 reduces it to 0.23% (t=1.14). JSZ — INV ASSET and INV BOOK have the **wrong sign**; ACC/TOTAL ACC/NOA insignificant; **BETA insignificant** (EW 0.36, t=1.18; VW 0.55, t=1.30). HQZ — beta long-short 0.93–1.10%/mo but explained by CH-3.
*Your factors:* keep **BETA** as a risk control (LSY: β is insignificant in Fama–MacBeth, t=−0.09; and CH-3's volatility/beta exposure is what you want to neutralise). Keep **ILLIQ** but expect CH-3 to absorb most of it.

**T14. Be explicit about costs, and prefer slow signals.**
*Evidence:* JSZ break-even transaction costs: **BM 1.47%** at 1-month holding rising to **4.44%** at 12-month holding (turnover falls 76.67% → 17.92%); **REV 1M turnover 355.26%**, RES REV 1M 361.93%, SEAS 366.76%, VOL 1M 265.43%, **ABN TURN 254.16%** (break-even only 0.12% for SEAS). China quoted/effective spreads 2009–2017: **0.11%–0.37% per trade** (Gao et al. 2020). Cheema–Nartea cite **0.40% per trade plus 0.10% stamp tax**.
*Action:* compute break-even cost = (1-month long-short return) ÷ (sum of two-way turnover of long and short legs). Anything with break-even cost below ~0.5%/trade is not implementable. This will likely disqualify high-frequency reversal slices unless executed with the LSY 5-day slicing and passive/pass-through execution.

**T15. Do not spend effort on SOE status.**
*Evidence:* JSZ: average |excess return| **0.76%/mo for non-SOEs vs 0.79%/mo for SOEs**; "there does not seem to be much need for investors to treat SOEs differently than non-SOEs when it comes to factor investing." Their own words: "A possible reason for the lack of empirical attention… could be our 'non-finding'."

**T16. Untested in this literature — flag as unvalidated.**
**INTRADAY, OVERNIGHT, LEAD_LAG, CO_SKEW, HIGH52** do not appear in LSY's 14-anomaly list, JSZ's 32-anomaly list, or HQZ's 426-anomaly list (HQZ's "other" category covers R&D, leverage, analyst forecasts — not these). They may be genuine, but there is no evidence in the papers I read either for or against them. Related-but-different signals that ARE documented: **idiosyncratic skewness** (HQZ: 0.83%/mo raw, CAPM α 0.65%, **CH-3 α 0.92% t=2.81**, CH-4 α 0.88% t=2.16 — this is the closest thing to your **ISKEW** and it is one of the few risk anomalies that survives CH-3); and **return seasonalities** (JSZ: SEAS 0.42% EW t=2.79; SEAS REV 0.81% EW t=3.28; SEAS DIFF 0.89% EW t=4.16, VW 0.90% t=2.83; HQZ: Ra25 0.65%/mo, Rn25 0.93%/mo, both t>2.30, explained by CH-3/CH-4). **SEAS DIFF is probably worth adding.**

### P3 — Quick reference: the exact recipes you need

```
UNIVERSE (all papers agree)
  - A-shares, SHSE + SZSE, main boards + ChiNext (+ STAR for JSZ)
  - drop if listed < 6 months
  - drop if suspended >= 120 days in past year  (LSY)  / >= 120 in JSZ wording
  - drop if suspended >= 5 days in past month    (JSZ) / < 15 trading days (LSY)
  - drop if suspended on the rebalance date      (JSZ, HQZ)
  - drop ST / PT stocks                          (JSZ)   <-- ADD THIS
  - drop bottom 30% by market cap (TOTAL A shares incl. non-tradable, LSY/HQZ/JSZ)

CH-3 (LSY)
  size split   : median market value of the top-70% universe
  value split  : EP, 30th / 70th percentiles; negative-EP stocks -> growth (15% of universe)
  portfolios   : 6 intersections, VALUE-weighted by all outstanding A shares
  SMB = 1/3(S/V+S/M+S/G) - 1/3(B/V+B/M+B/G)
  VMG = 1/2(S/V + B/V) - 1/2(S/G + B/G)
  MKT = VW return of top 70% - 1-year deposit rate

CH-4 (LSY)
  PMO on ABNORMAL TURNOVER = (past-month share turnover) / (past-year share turnover)
  built exactly like VMG, but LONG low turnover / SHORT high turnover
  new SMB = average( EP-neutralised SMB , turnover-neutralised SMB )

ABN TURN (LSY / JSZ)   = mean daily turnover over past 20 days / mean daily turnover over past 250 days
TURN      (LSY / JSZ)  = mean daily turnover over past 250 days (shares traded / total shares)
turn1-1   (HQZ)        = 1-month average of trading volume / FLOAT A shares
vturn1-1  (HQZ)        = within-month VARIATION of daily share turnover      <-- ADD
ILLIQ     (Amihud)     = mean over past 20 days of |ret_t| / dollar volume_t
EP        (LSY)        = net profit EXCLUDING non-recurrent gains/losses / (last month-end close x TOTAL shares)
BM        (LSY)        = (total shareholder equity - book value of preferred) / (close x TOTAL shares)
ROE       (LSY)        = earnings / book equity
MAX       (LSY)        = max daily return over past 20 trading days
MAX5      (HQZ)        = mean of the FIVE highest daily returns in the month   (price-limit adjusted)
MAX       (JSZ)        = max daily return, past 20 days, with suspension days = -99 and
                         consecutive limit-up returns AGGREGATED
VOL_1M    (JSZ)        = SD of daily returns, past 20 days
IVOL      (JSZ)        = SD of residuals, daily excess returns on MKT+SMB+VMG, past 250 days
IVOL      (HQZ)        = SD of residuals, daily excess returns on CH-3, past 1 month
REV_1M    (JSZ)        = cumulative return, past 20 trading days
REV_LT    (JSZ)        = cumulative monthly return, months t-60 to t-13
RES_MOM   (JSZ)        = 36-month rolling FF-3/CH-3 residual returns; use past 11 months;
                         scale each by its own SD over that window; take the mean
SEAS_DIFF (JSZ)        = average same-calendar-month return - average other-month return
                         (cross-sectionally demean returns first; require >= 5 years history)
```

---

## 6. Direct answers to the seven extraction questions

**(1) Exact anomaly list and significance.** Full tables in §1.6 (LSY: 14 tested, 10 significant vs CAPM, 8 explained by CH-3), §2.1 (JSZ: 32 tested, VW 1-month numbers for all 32), §2.2 (HQZ: 426 tested, 98 significant VW / 175 EW, 16 CH-3 / 15 CH-4, 27 at t>2.85, 2 CH-3 and 0 CH-4 after the hurdle). Size, value (EP), ROE, turnover, volatility, MAX, reversal, illiquidity: all listed with the paper's own numbers above.

**(2) Exact construction recipes.** §1.4, §1.9, §2.1 (Appendix B verbatim), §2.2, §3.3, §5 P3 quick-reference. Breakpoints: median size; 30/70 EP; 30/40/30 turnover; deciles (LSY, JSZ, HQZ) or tertiles (Nartea et al.). Weighting: **value-weighted is the default in LSY/JSZ/HQZ**; JSZ also report EW and note EW overweights microcaps. Rebalancing: monthly, except **reversal, which LSY sort daily and rebalance in 1/5 slices held 5 days**. Small-stock exclusion: **all three drop the bottom 30%**. ST exclusion: **JSZ only, explicitly**. Listed < 6 months: **all three**. < 12 months: **none of them**.

**(3) Why US anomalies fail in China.** Shell-value contamination (LSY, §1.10 — 83% of 133 reverse mergers from the bottom 30%; ~29.5% of their market value is shell value), retail dominance (101m accounts, 88% of float), short-sale constraints, IPO/reverse-merger channel. **Price limits** are cited as a *measurement* problem for MAX (HQZ, JSZ, Nartea et al.), not as a reason for anomaly failure. **T+1 is not discussed in any of the papers I read.** Momentum failure is attributed to **market dynamics / bear-market states** (Cheema–Nartea, HQZ citing Daniel–Moskowitz) and to a **retail news-day-overreaction vs non-news-day-correction tug-of-war** (Dissecting Momentum in China), and originally to **individualism/overconfidence** (CTW).

**(4) Parameters.** LSY: 2000–2016, 610→1,872 stocks, ≥50 per portfolio (≥60 per decile), 4 filters, release-date-matched accounting, no ST screen. JSZ: 2000–2019, 1,060→3,750 stocks, 6 filters incl. ST, both EW and VW, 1/6/12-month holdings. HQZ: 2000–2018, 681→2,465 stocks, 3 filters, deciles, t>1.96 / t>2.85.

**(5) Reversal magnitudes and best holding period.** §3.1. Best holding period = **1 month / 20 trading days**. LSY VW α 1.49%/mo (t=3.06), size-neutral 1.65% (t=3.68), 18% annualised (t=2.96); JSZ EW 1.42% (t=4.35). Weekly reversal is bid–ask contaminated (1.86%/wk → 0.08%/wk after a skip week).

**(6) Mechanism for momentum failure.** §3.2 — three independent accounts: cultural individualism/overconfidence (CTW: High−Low Indv momentum spread 0.530%/mo, t=3.42; China Indv = 20), market dynamics (Cheema–Nartea: momentum only in DN/DN; and they explicitly conclude that cross-country momentum differences reflect **market dynamics rather than individualism**), and retail news-day overreaction vs institutional non-news-day correction (Dissecting Momentum). HQZ add bear-market conditions since 2000 (Daniel–Moskowitz).

**(7) Conditional momentum that works.** §3.3 — residual momentum (0.66%/mo EW t=3.36), large-cap-only momentum (all holding periods), DN/DN state (1.45%/mo, t=2.93), momentum change (0.57%/mo, t=2.79), industry momentum (0.63%/mo, t=2.00).

**(8) MAX / IVOL sign in China vs the US.** §4 — **SAME SIGN, not reversed.** MAX: China EW 1.03–1.20%/mo vs US EW 0.65–1.02%/mo; China VW 0.66–0.96%/mo vs US VW 1.03%/mo. IVOL: China long-short 0.74–1.19%/mo, t > 2, same sign as US Ang et al. The **low-MAX / low-IVOL premium is PRESENT**, somewhat weaker VW, and in the case of MAX it is **conditional on high investor sentiment** (1.85%/mo after high vs 0.22% after low). Whether MAX subsumes IVOL is **contested in China** (Nartea et al.: no, both coexist; SCIRP 2023: yes).

---

## 7. What I could NOT access (no numbers invented for these)

| Item | Status |
|---|---|
| **Naughton, Truong, Veeraraghavan (2008)**, "Momentum strategies and stock returns: Chinese evidence", PBFJ 16(4), 476–492 | **Full text not obtained** (Elsevier paywall, no OA copy found; ScienceDirect/Wiley/SSRN blocked from this environment). Known only via JSZ's footnote 18: it finds a **significant** momentum effect for Chinese A-shares over **1995–2005** — the main exception to the "no momentum in China" consensus. |
| **Cheema & Nartea (2017)**, "Momentum returns, market states, and market dynamics: Is China different?", International Review of Economics & Finance 50, 85–97, doi:10.1016/j.iref.2017.04.003 | **Full text not obtained** (Unpaywall: closed, no OA locations; a CORE PDF link 404'd). The market-state methodology and results I report come from the **companion paper** Cheema & Nartea (2017) PBFJ, which replicates and extends it — the PBFJ paper explicitly states it replicates Cheema & Nartea (2017). |
| **Nartea, Wu, Liu (2013)**, "Does idiosyncratic volatility matter in emerging markets? Evidence from China", JIFMIM 27, 137–160 | **Full text not obtained** (Elsevier paywall, Unpaywall: closed). The IVOL result I report is Nartea–Kong–Wu's own characterisation: "a highly significant anomalous negative IV effect consistent with the findings of Nartea et al. (2013)". |
| **"Momentum returns, market states…"**, **Naughton et al.**, **Nartea et al. 2013**, **Cheema & Nartea (2014) "Momentum returns and information uncertainty"** | All inaccessible for the same reason. |
| LSY's **Internet Appendix** (the WIND data item codes, English + Chinese) | Not obtained. The main text refers to it for variable definitions; the substantive definitions are in Appendix A.2, which I did read in full. |
| LSY's Appendix A.3 (seven-step IPO process), A.5 (shell-value covariation), A.6 (ex-financials Table 2), A.7 (CH-3 vs FF-5) | A.3 read (summarised in §1.2); A.5–A.7 only via the main text's summary (CH-3 GRS p = 0.88 vs FF-5 non-market factors; FF-5 alphas for CH-3's SMB/VMG give GRS p = 0.0003; ex-financials Table 2 "virtually unchanged"). |
| HQZ's **Online Appendix B** (per-anomaly k, m, n) and Online Appendix D (long-leg-only returns) | Not obtained; the paper states the parameters are marked per anomaly there. The 98-anomaly numeric table (Table 4) is present in the PDF but partially garbled in text extraction for Panels B–G beyond the numbers quoted above. |
| JSZ Table 6 (size-neutral and industry-neutral returns by anomaly) and Tables C1–C6 | Only the narrative results are reported above (which anomalies move, and by how much for ROE/EP/12-month turnover); the full numeric grids were not cleanly extractable. |
| Housekeeping | `web_search` (Firecrawl) was rate-limited for the whole session and every general search engine (Google, Bing, DDG, Yandex, Startpage, Brave, Ecosia, Qwant, Mojeek, Marginalia, scholar.archive.org) was captcha- or bot-walled from this IP. All discovery was done through **Crossref + Unpaywall + OpenAlex APIs** and **DuckDuckGo via the `r.jina.ai` reader proxy**; downloads were direct from publisher/repository URLs. ScienceDirect, Wiley Online Library, SSRN and JSTOR were unreachable directly, which is why four papers above are missing. |
