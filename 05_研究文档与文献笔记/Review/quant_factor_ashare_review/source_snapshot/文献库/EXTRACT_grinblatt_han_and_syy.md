<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# Deep Extraction: Behavioral Factors and Time-Varying Conditioning of Anomaly Returns

**Prepared for:** replication-grade implementation (A-share market, limited data)
**Date of extraction:** this session

---

## 0. SOURCES ACTUALLY READ (and what could NOT be accessed)

| # | Paper | Full text obtained? | What I actually read |
|---|-------|--------------------|----------------------|
| 1 | Grinblatt & Han (2005), JFE 78(2), 311–339 | **YES — accepted manuscript of the published JFE article** | `papers/gh_ut.pdf` → `papers/gh_ut.txt` (88,737 chars), downloaded from the University of Toronto TSpace repository, item handle `http://hdl.handle.net/1807/135191`, PDF at `https://utoronto.scholaris.ca/bitstreams/3a09de05-9370-4e68-a03d-ccce917a5cb6/download`. Title page states: *"Version Pre-print or submitted manuscript… This is a submitted manuscript of an article published by Elsevier in the Journal of Financial Economics"*, dated **August 2004**, DOI `10.1016/j.jfineco.2004.10.006`. Also read the earlier NBER working-paper version `papers/gh_nber.txt` (NBER WP **w8734**, "The Disposition Effect and Momentum", `https://www.nber.org/system/files/working_papers/w8734/w8734.pdf`) for the explicit algebraic definition of the normalizing constant `k`. |
| 2 | Stambaugh, Yu & Yuan (2012), JFE 104(2), 288–302 | **PARTIAL — NBER working-paper version, not the JFE typeset version** | `papers/syy_nber.pdf` → `papers/syy_nber.txt` (70,832 chars), NBER WP **w16898**, March 2011, `https://www.nber.org/system/files/working_papers/w16898/w16898.pdf`. This is the pre-publication version of the JFE article (DOI `10.1016/j.jfineco.2011.12.001`). I could **not** obtain the final JFE typeset text: ScienceDirect returns **HTTP 403** for `https://www.sciencedirect.com/science/article/pii/S0304405X11002935`, and the Penn institutional repository records (`https://repository.upenn.edu/handle/20.500.14332/34488`, `https://repository.upenn.edu/fnce_papers/380`) returned only an 827-byte JavaScript shell with no PDF link. **All numbers below are from the NBER WP and may differ in the published version.** |
| 3 | Cooper, Gutierrez & Hameed (2004), JF 59(3), 1345–1365 | **YES — the published Journal of Finance article** | `papers/cgh_jf2004.pdf` → `papers/cgh_jf2004.txt` (61,030 chars). Found via the **OpenAIRE** API and downloaded from a Wharton course page: `http://www-stat.wharton.upenn.edu/~steele/Courses/434/434Context/Momentum/cooper_gutierrez_hameed_JF2004.pdf`. Header of the file reads *"THE JOURNAL OF FINANCE • VOL. LIX, NO. 3 • JUNE 2004 / Market States and Momentum / MICHAEL J. COOPER, ROBERTO C. GUTIERREZ JR., and ALLAUDEEN HAMEED"*. |
| 4 | Baker & Wurgler (2006), JF 61(4), 1645–1680 | **PARTIAL — NBER working-paper version** | `papers/bw_nber.pdf` → `papers/bw_nber.txt` (108,647 chars), NBER WP **w10449**, `https://www.nber.org/system/files/working_papers/w10449/w10449.pdf`. The Wiley published PDF returns **HTTP 403** (`https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.1540-6261.2006.00885.x`). See Section 4. |
| + | **Cheema & Nartea (2017)**, IREF 50, 85–97 — *China-specific replication of Cooper et al., used in §3.5 and §6.4* | **YES — accepted manuscript** | `papers/cheema_china.pdf` → `papers/cheema_china.txt` (67,102 chars), `https://researchcommons.waikato.ac.nz/bitstream/10289/11238/5/IREF_accepted.pdf` (found via OpenAIRE). DOI `10.1016/j.iref.2017.04.003`. **This is a separate paper, not one of the four requested** — it is cited only as the China-specific evidence on market-state conditioning. |

**Routes that failed and are therefore NOT sources:** ScienceDirect (403 on both articles), Wiley Online Library pdfdirect (403 on both JF articles), `web_search` (every engine failed — firecrawl keyless rate limit, 86,000 s retry-after), DuckDuckGo HTML/lite (`WinError 10060` / SSL EOF), Bing (empty result set), Mojeek (SSL EOF), BASE (Anubis bot challenge), Semantic Scholar Graph API (HTTP 429/404), OpenAlex (HTTP 429).

---

## 1. PAPER 1 — GRINBLATT & HAN (2005): Capital Gains Overhang (CGO)

### 1.1 The exact reference-price / CGO formulas

**Model recursion (equation (1) of the paper, `gh_ut.txt` line 264):**

```
R_{t+1} = V_t * P_t + (1 - V_t) * R_t
```

> *"New reference prices are thus weighted averages of old reference prices and the prices at which new shares trade."* (line 262–263)
> *"We believe that the updating weight, V_t, should be related to the stock's turnover ratio, since the cost basis is the reference price that motivates the mental account."* (line 269–270)

`V_t` is the **turnover ratio at date t** (weekly share volume ÷ shares outstanding). It is used **directly** as the updating weight.

**Explicit sum actually used in the empirics (equation (4), `gh_ut.txt` line 356–368):**

```
              ∞     ⎛                    n-1              ⎞
R_t   =      Σ     ⎜ V_{t-n}  *   Π    [1 - V_{t-n+τ}]   ⎟  * P_{t-n}
             n=1   ⎝                   τ=1               ⎠
```

The paper states (line 383–387):
> *"the term in parentheses multiplying P_{t-n} is a weight and that all the weights sum up to one. The weight on P_{t-n} is just the probability that a share was last purchased at date t−n and has not been traded since then."*

**Truncated, rescaled version actually computed (Table 3 note, `gh_ut.txt` lines 1264–1280 — this is the operational formula):**

```
                   1    260  ⎛                     n-1              ⎞
R_{t-1}   =       ---   Σ   ⎜ V_{t-1-n}  *   Π    [1 - V_{t-1-n+τ}]  ⎟ * P_{t-1-n}
                   k    n=1  ⎝                    τ=1               ⎠
```
> *"with k a constant that makes the weights on past prices sum to one."* (line 1280)

The NBER version (`gh_nber.txt` lines 870–896) gives **`k` explicitly** — and it is **not a fixed number, it is the sum of the unnormalised weights, recomputed every week for every stock**:

```
       260
k  =    Σ   V_{t-n}  *  Π_{τ=1}^{n-1} [1 - V_{t-n+τ}]
       n=1
```
> *"the scaling constant that makes the fractions sum to one, k = …"* (`gh_nber.txt` lines 886–896)

**The regressor (the capital gains overhang), `gh_ut.txt` line 429–432:**

```
g_{t-1}  =  ( P_{t-2}  -  R_{t-1} ) / P_{t-2}
```
> *"Theory says that this key regressor should employ P_{t-1} instead of P_{t-2}. We lag the market price by one week to avoid confounding market microstructure effects, such as bid-ask bounce."* (lines 436–438)

Equivalently, per the Table 3 note (lines 1264–1265): *"g = the capital gains regressor, computed as one less the ratio of the beginning of week t−1 reference price to the end of week t−2 price."*

So: **`g = 1 − R/P`, i.e. `g = (P − R)/P`, a number in (−∞, 1).**

### 1.2 ⚠️ There is NO separately calibrated `delta` parameter — direct answer

The premise in the request ("they calibrate it — what value?") **does not hold for this paper**. I grepped both full texts for `delta`, `δ`, `decay`, `calibrat*`, `shrink` and read every hit:

- `gh_ut.txt`: **zero** occurrences of a delta/decay parameter. The only "geometric" reference is to *"a constant geometric perturbation of the rational type's demand function"* (line 218) — a demand-function assumption, **not** a price weight.
- `gh_nber.txt`: the only "geometric" hit is *"restricting loss realization aversion to be a geometric deviation from fully rational"* (line 1243) — again not a price weight.

**The decay is endogenous and stock-specific: the weight on `P_{t-n}` is `V_{t-n} · Π(1−V_{t-n+τ})`, where `V` is that stock's own realised turnover.** There is no free parameter to tune. The only two "calibration" choices the paper makes are:

1. **Truncation horizon: 5 years = 260 weeks.** *"we truncate the estimation at five years and rescale the weights to sum to one"* (lines 441–442). Robustness: *"our regression results remain about the same when return and turnover data over three or seven prior years are used"* (lines 446–448); identical statement in `gh_nber.txt` lines 943–944.
2. **The per-stock, per-week normaliser `k`** (above), which makes the 260 weights sum to one.

The implied decay rate is governed by the data. The paper notes (`gh_nber.txt` lines 905–911):
> *"with a constant weekly turnover of V, the average holding period is 1/V weeks or approximately 2/V years. Assuming a constant weekly turnover of 1%, which is approximately the mean for entire sample period, this implies an average holding period of 2 years."*

### 1.3 Practical data requirements — and what the paper does NOT say

**Sample/data (lines 388–398, 444–446):**
- Weekly returns, weekly turnover (**weekly trading volume ÷ number of outstanding shares**), and market capitalisation.
- **MiniCRSP** database; **all ordinary common shares traded on NYSE and AMEX**; **NASDAQ firms are not available**.
- Sample: **July 1962 – December 1996, 1799 weeks**. Regressions run on **July 1967 – December 1996** (1,539 weekly cross-sections).
- Weekly (not daily, not monthly) frequency is chosen deliberately: *"Our choice of weekly data arises from the need to have a reasonable proxy for a critical variable, the capital gains overhang. This requires higher frequency data than monthly data provide and transaction prices that are less influenced by market microstructure than daily data provide."* Volume numbers on weekly MiniCRSP were revised for reliability (Lim, Adamek, Lo & Wang 2003).

**Turnover for every past month?** — Yes, effectively for every past **week** over a 5-year window: the formula needs `V_{t-1-n}` for `n = 1 … 260`, i.e. **260 consecutive weekly turnover observations plus 260 consecutive weekly prices**. Turnover is also used as a regressor separately: `V̄ = average weekly turnover over the prior 52 weeks`.

**Initialisation of the reference price at the start of the sample — the paper's answer is to NOT initialise it; it requires a full 5-year burn-in:**
> *"Stocks that lack at least five years of historical return and turnover data at a particular week are excluded from the cross-sectional regression for that week."* (lines 444–446)
> *"This allows us to estimate the reference price in a consistent manner across the sample period. The five-year cutoff, while arbitrary, admits a reasonable portion of our sample period: July 1967 on."* (lines 442–444)

This is the cleanest implementable rule: **drop any stock-week lacking 260 weeks of history; never seed `R` with an arbitrary value.** (No warm-up/seed value is specified because none is used.)

**Stock splits and missing turnover — ⚠️ NOT ADDRESSED.** I grepped both full texts for `split`, `missing`, `initialize`, `initial`, `adjust`. Results:
- `split`: **0 matches** in either full text.
- `missing`: **0 matches**.
- `initial`: only "initial wealth" (line 82) and "initial test" (line 491) — unrelated.
- The paper's price series is described only as "prices" from MiniCRSP with the `P_{t-1}` vs `P_{t-2}` lag justified by bid-ask bounce (lines 436–438). **The paper never states whether prices are split-adjusted, nor what it does with missing turnover or zero-volume weeks.**

I will **not** invent a rule. The defensible inference (flagged as inference, not as the paper's text): because `g = (P−R)/P` is a *ratio* of a weighted average of past prices to a recent price, an unadjusted split inside the 260-week window would corrupt `g` catastrophically, so the series must be total-return/split-adjusted in practice. For A-shares this means **use 后复权 (backward-adjusted) prices** consistently for both `P` and `R`.

**Other sample facts (`gh_ut.txt` Table 1 Panel A, lines 920–962):** time-series averages of cross-sectional statistics for the regressors
`r = a0 + a1·r_{−4:−1} + a2·r_{−52:−5} + a3·r_{−156:−53} + a4·V̄ + a5·s + a6·g`:

| | r_{−4:−1} | r_{−52:−5} | r_{−156:−53} | V̄ | s | **g** |
|---|---|---|---|---|---|---|
| Mean | 0.0119 | 0.1493 | 0.3487 | 0.0092 | 18.7207 | **0.0560** |
| Median | 0.0045 | 0.0940 | 0.2098 | 0.0072 | 18.7251 | **0.1062** |
| Std | 0.1073 | 0.4192 | 0.7585 | 0.0079 | 1.9441 | **0.2508** |
| 10th pct | −0.0959 | −0.2538 | −0.3227 | 0.0025 | 16.1399 | **−0.2810** |
| 90th pct | 0.1223 | 0.5816 | 1.1097 | 0.0181 | 21.2322 | **0.3122** |

So the cross-sectional **90th − 10th percentile spread of `g` is ≈ 0.593**; the paper reports a **time-series average of 76% and a median of 60%** for that spread (lines 454–456).

### 1.4 Exact relationship between CGO and future returns

**Sign convention (the single most important line):** from equation (3), `gh_ut.txt` lines 300–312:

```
E_t[ (P_{t+1} − P_t) / P_t ]  =  (1 − w) · V_t · (P_t − R_t)/P_t
```
> *"This equation suggests that a stock's expected return is monotonically increasing in the marginal investor's (percentage) unrealized capital gain, (P_t − R_t)/P_t."* (lines 311–312)

**⇒ LONG HIGH-CGO (large unrealized capital gain), SHORT LOW-CGO.** Higher `g` ⇒ higher expected return. There is no sign flip. The mechanism's second component is that *for a fixed gain, higher current turnover `V_t` implies a larger forecast absolute return* (lines 312–318).

**Fama-MacBeth cross-sectional regression (equation (5), line 426):**
```
r = a0 + a1·r_{−4:−1} + a2·r_{−52:−5} + a3·r_{−156:−53} + a4·V̄ + a5·s + a6·g
```
Weekly cross-sections, week t return `r_t = (P_t − P_{t−1})/P_{t−1}` as dependent variable; coefficients averaged over 1,539 weekly regressions; **t-statistics computed from the time series of the weekly cross-sectional coefficient estimates** (lines 401–402, 1281–1282).

**Table 3 Panel C — EXACT numbers** (`gh_ut.txt` lines 1383–1443):

| Period | a1 (r_{−4:−1}) | a2 (r_{−52:−5}) | a3 (r_{−156:−53}) | a4 (V̄) | a5 (s) | **a6 (g)** |
|---|---|---|---|---|---|---|
| **All** | −0.0425 (−35.9364) | **−0.0002 (−0.6794)** | −0.0007 (−5.0871) | −0.0188 (−0.9364) | −0.0004 (−5.2885) | **0.0040 (7.7885)** |
| **Jan** | −0.0520 (−10.9905) | −0.0001 (−0.0477) | −0.0025 (−3.8964) | −0.0620 (−0.9768) | −0.0026 (−8.4381) | **−0.0117 (−4.9519)** |
| **Feb–Nov** | −0.0407 (−32.6251) | **−0.0000 (−0.0768)** | −0.0006 (−3.6950) | −0.0291 (−1.3143) | −0.0002 (−2.8816) | **0.0050 (9.4191)** |
| **Dec** | −0.0498 (−10.8151) | −0.0022 (−1.8953) | −0.0005 (−1.3410) | 0.1238 (1.7980) | 0.0001 (0.2702) | **0.0104 (6.2673)** |

**Magnitude interpretation (the paper's own arithmetic, lines 552–563):** average coefficient **0.0040** on `g`; with a **median** p90−p10 spread in `g` of about **60%**, this implies *"winners outperform losers by about 0.004 × 60% = 0.24% per week, or 12.5% per year."*

**Monthly-return version (Table 7 Panel A, `gh_ut.txt` lines 2175–2235),** dependent variable = next **month's** return, same `g`:
| Period | a6 (g) |
|---|---|
| All | **0.0127 (5.4241)** |
| Jan | **−0.0345 (−4.1412)** |
| Feb–Nov | **0.0152 (6.3402)** |
| Dec | **0.0344 (4.6423)** |

**Double-sorted portfolio evidence (Table 2, verified against the PDF's coordinate layout — the two column groups are `January` and `February-December`, x-positions 137/182/228/273/318 and 362/404/446/488/530 on PDF page 24):**
Panel B (first sort on past 1-year return into R1 losers … R5 winners, then on CGO into G1 … G5 within each). **G5−G1 spread, February–December, per week:**
| within | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| G5−G1 | **0.0008** (1.6146) | **0.0010** (2.6852) | **0.0017** (4.8102) | **0.0012** (3.3838) | **0.0026** (6.7453) |

In **January** the same G5−G1 spreads are **negative and large**: −0.0175, −0.0155, −0.0143, −0.0154, −0.0123, t = −6.51, −7.67, −6.20, −7.45, −5.51. The paper: *"The portfolio returns during the January months are not consistent with a stable PT/MA parameter λ… the January returns of high capital gain stocks tend to be below those of the low capital gain stocks. This may reflect a December tax-loss selling effect."* (lines 522–526). Narrative text summarises the non-January G5−G1 spreads as *"ranging from about 0.12% to 0.25% per week (about 6% to 13% per annum)"* (line 508) — note this is an approximation; the table values above run 0.08%–0.26%.

Panel C (reverse sort order: first on CGO, then on past return) shows the **converse does NOT hold.** The R5−R1 spread *within* each CGO quintile, February–December: G1 −0.0003 (t = −0.6167), G2 0.0001 (t = 0.2956), G3 0.0006 (t = 1.4524), G4 0.0008 (t = 1.8355), G5 0.0017 (t = 3.6596) — only the top CGO quintile reaches significance. (January values for the same row: 0.0040 (1.5204), 0.0031 (1.9304), 0.0032 (2.2502), 0.0040 (2.4970), 0.0051 (2.9409).) *"Panel C indicates that the reverse is not true."* (line 509–510).

### 1.5 How much of momentum does CGO explain? The horse race

**Result: CGO completely subsumes intermediate-horizon momentum.** Compare Table 3 Panel B (without `g`) to Panel C (with `g`), same sample, same controls:

| Regressor | Panel B (no `g`) | Panel C (with `g`) |
|---|---|---|
| a2 = r_{−52:−5}, All months | **0.0014 (3.5703)** | **−0.0002 (−0.6794)** |
| a2 = r_{−52:−5}, Feb–Nov | 0.0021 (5.1324) | **−0.0000 (−0.0768)** |
| a2 = r_{−52:−5}, Dec | 0.0049 (3.7745) | −0.0022 (−1.8953) |
| a2 = r_{−52:−5}, Jan | −0.0086 (−4.5561) | −0.0001 (−0.0477) |
| a6 = g | — | 0.0040 (7.7885) / Feb–Nov 0.0050 (9.4191) |

Panel A/B numbers: `gh_ut.txt` lines 1331–1371.

Paper text (lines 547–552): *"When the capital gains overhang regressor is included in the regression, there is no longer an intermediate horizon momentum effect. The coefficient, a2, is insignificant, both overall and from February through November. However, except for January, there is a remarkably strong cross-sectional relation between the capital gains overhang variable and future returns."*

Abstract (line 53–55): *"A variable proxying for aggregate unrealized capital gains appears to be the key variable that generates the profitability of a momentum strategy. **Past returns have no predictability for the cross-section of returns once this variable is controlled for.**"*

**Supplementary horse races, all confirming the same conclusion:**
- **vs. average-turnover reference price (Table 4 Panel A/B):** replacing each week's actual `V` with the firm's average weekly turnover over weeks `t−52 … t−1` (`V̄`, applied to all 260 terms) still yields a significant coefficient, `a6 = 0.0043 (8.0694)` overall (`gh_ut.txt` lines 1472–1484), and intermediate momentum again dies (`a2 = −0.0003, t = −0.9434`). But in the **head-to-head** regression (Panel B) *"The inclusion of this variable eliminates the significance of the alternative gain variable"* — original `g` = 0.0028 (2.4381), alternative `ḡ` = 0.0014 (1.2590) (lines 1534–1548).
- **vs. `gain × lagged turnover` (Table 4 Panel C):** regressing on `V_{−1} · g` instead of `g` **approximately doubles the t-statistic**: `a6 = 0.4876 (15.6377)` overall, 0.5271 (16.2106) Feb–Nov, 0.7522 (6.6208) Dec (lines 1596–1648) — versus `t = 7.7885` for `g` alone. This is exactly the model's prediction (footnote 4, lines 373–379: *"The model also suggests multiplying the gain by one-period lagged turnover… there may be a cross-sectional relation between a firm's typical turnover V and w, which we cannot estimate."*)
- **vs. Lee & Swaminathan (2000) volume–loser interaction (Table 4 Panels D–F):** the L&S variable is significant on its own (`a5 = −0.0447, t = −2.9342`, lines 1658–1667) but becomes **insignificant once `g` is added**, while `g` stays highly significant. *"In Panel F, our capital gain-volume interaction variable replaces the gain variable. Again, the Lee and Swaminathan variable is insignificant."* (lines 694–695).
- **vs. 52-week-high as reference price (Table 6 Panel B):** when both are included, **both are significantly positive** — *"This is what we would expect if the model was correct and both variables are imperfect proxies for the theoretical variable."* (lines 723–727).
- **Subperiods (Table 5):** *"There is only about a one standard error difference between the average coefficients on the capital gain regressor in the two subperiods [July 1967–March 1982 vs April 1982–December 1996]. In both subperiods, the average coefficient is highly significant and positive, while the average coefficient for the intermediate horizon past return is never significant in the presence of the capital gain variable."* (lines 711–717).

### 1.6 Theoretical mechanism (prospect theory / mental accounting → momentum)

Some investors ("PT/MA investors") evaluate each stock in a separate mental account against a **reference price `R_t`**, which is their **cost basis** — the price at which they acquired the shares. Because they are loss-averse in the prospect-theory sense, they are reluctant to realise losses, so their demand is *increasing* in `(R_t − P_t)` (they hold on to losers) while rational investors' demand is increasing in `(F_t − P_t)`. Aggregating and clearing the market gives an equilibrium price that is a **weighted average of fundamental value and the reference price**: `P_t = w·F_t + (1−w)·R_t`, with `w = 1/(1+μλ) < 1`, so **price underreacts to fundamental news**. Each period, shares change hands and the reference price is updated toward the traded price, `R_{t+1} = V_t·P_t + (1−V_t)·R_t`, with `V_t` the turnover ratio. Since `P` sits between `F` and `R`, and `R` is pulled toward `P`, both ultimately converge to `F` — and the expected price change is proportional to `V_t·(P_t − R_t)`. **A stock with a large unrealised capital gain (`P >> R`) therefore has a predictably positive expected return as the reference price catches up to the market price; a stock with a large unrealized loss has a predictably negative one.** Because past returns are a noisy proxy for the accumulated capital gain, this mechanism *generates* momentum — but the capital-gain variable is the sufficient statistic, which is why past returns lose all predictive power once `g` is controlled for. The model also predicts a **hump-shaped** momentum intensity in the ranking horizon, because at long horizons the gain itself reverts to zero as positions are replenished at recent prices (lines 332–347).

---

## 2. PAPER 2 — STAMBAUGH, YU & YUAN (2012): Sentiment-conditioned anomalies

> **Version warning:** everything in this section is from **NBER WP w16898 (March 2011)**, not the final JFE typeset article.

### 2.1 The three hypotheses (exact statements, `syy_nber.txt` lines 214, 223–224, 231–232)

> **Hypothesis 1:** The anomalies should be stronger following high investor sentiment. (line 214)
> **Hypothesis 2:** The short legs of the long-short strategies should have lower returns (greater profits) following high investor sentiment. (lines 223–224)
> **Hypothesis 3:** The long legs of the long-short strategies should have similar returns following high and low investor sentiment. (lines 231–232)

Economic logic (lines 195–224): combining **market-wide sentiment** with **Miller's short-sale-impediment argument**. With shorting impeded, the most optimistic investors set the price; when sentiment is high those optimistic views are too optimistic → **overpricing**, concentrated in the short leg. Underpricing is unlikely because the rational view lies inside the cross-section of views.

### 2.2 Sentiment proxy — and the real-time issue

**They use the Baker & Wurgler (2006) monthly market-based sentiment index, in LEVELS.** `syy_nber.txt` lines 255–261:
> *"We measure investor sentiment using the monthly market-based sentiment series constructed by Baker and Wurgler (2006). The index spans over 42 years, from July 1965 to December 2007. Baker and Wurgler (2006) form their composite sentiment index by taking the first principal component of six measures of investor sentiment… The six measures are the closed-end fund discount, the number and the first-day returns of IPO's, NYSE turnover, the equity share in total new issues, and the dividend premium."*

**Real-time availability — the honest answer: the paper does not construct a real-time/recursive version.** They use the full-sample BW index as published, whose components include *annual* series (equity share, number of IPOs) and *raw* series that BW themselves orthogonalise using **full-sample** macro regressions. The timing convention that makes the test implementable is a **one-month lag only**: the classification uses the **previous month's** index value:
> *"A high-sentiment month is one in which the value of the BW sentiment index in the **previous month** is above the median value for the sample period, while the low-sentiment months are those with below-median values."* (lines 406–409)

Note the median is the **full-sample** median — a look-ahead in the classification threshold. The regressions use `S_{t−1}` explicitly (see 2.3).

Also note the macro-orthogonalisation already inside BW (lines 547–550):
> *"Baker and Wurgler (2006) remove macro-related variation from their sentiment index by regressing raw sentiment measures on six macro variables: the growth in industrial production, the growth in durable, nondurable, and services consumption, the growth in employment, and a flag for NBER recessions."*

**Alternative proxy (robustness, lines 578–595):** the **University of Michigan Consumer Sentiment Index**, orthogonalised by taking *"the residuals from a regression of the Michigan index on the six macro-related variables used by Baker and Wurgler"* (lines 584–586). Results are similar but *"some of them are weaker"*. They also note (footnote 10) that the Conference Board Consumer Confidence Index gives largely the same results.

### 2.3 EXACT regression specifications

**This is NOT a panel regression.** Despite the user's framing, SYY run **pure time-series regressions of an aggregate portfolio-return series on the lagged sentiment level** — one regression per anomaly leg. There is no cross-sectional dimension in the estimation, no fixed effects, no clustering. Standard errors are **White (1980) heteroskedasticity-consistent**.

**(a) The dummy-variable (high/low) specification** — Table 2 (excess returns) and Table 3 (FF3 benchmark-adjusted returns). Table 3 note, `syy_nber.txt` lines 1216–1219:
```
R_{i,t}  =  a_H · d_{H,t}  +  a_L · d_{L,t}  +  b·MKT_t  +  c·SMB_t  +  d·HML_t  +  ε_{i,t}
```
> *"where d_{H,t} and d_{L,t} are dummy variables indicating high- and low-sentiment periods, and R_{i,t} is the excess return in month t on either the long leg, the short leg, or the difference."*

Table 2 is the same regression **without** the `MKT/SMB/HML` terms (it reports raw excess returns over the one-month T-bill). Table 2 note, lines 953–958.

**(b) The predictive regression on the sentiment LEVEL** — Table 4 (excess returns), note at lines 1479–1489:
```
R_{i,t}  =  a  +  b · S_{t−1}  +  u_t
```
**(c) The benchmark-adjusted predictive regression** — Table 5, note at lines 1589–1599:
```
R_{i,t}  =  a  +  b · S_{t−1}  +  c·MKT_t  +  d·SMB_t  +  e·HML_t  +  u_t
```
> *"S_t is the **level** of the investor-sentiment index of Baker and Wurgler (2006)… All t-statistics are based on the heteroskedasticity-consistent standard errors of White (1980)."*

**Answers to the specific questions:**
- **LEVEL, not change.** `S_{t−1}` is the level. `S` is *"scaled to have zero mean and unit standard deviation"* (line 489) — so `b` is the effect of a one-standard-deviation move in the index.
- **Timing/lag convention:** sentiment dated `t−1` predicts the return in month `t` — a **one-month lag**. In the dummy version, month `t` is classified by the previous month's index value.
- **Newey–West is NOT used.** White (1980) HC standard errors only.
- **Stambaugh (1999) predictive-regression bias** is checked and dismissed: *"The correlations between the predictive-regression residuals and the innovations in sentiment level obtained from a first-order autoregression are small. Applying Stambaugh's bias correction… produces only small changes in the second decimal place of some of the coefficients"* (lines 525–530).
- **Robustness** (Table 6): adding five lagged macro variables — *default premium* (BAA−AAA yield spread), *term premium* (20-yr minus 1-yr Treasury), *real interest rate* (30-day T-bill minus CPI inflation), *inflation rate*, and *CAY* (Lettau–Ludvigson consumption-wealth) — leaves coefficients and t-statistics close to Table 5 (lines 551–565).

### 2.4 The 11 anomalies and portfolio construction (lines 276–401)

Anomalies, all defined relative to the **Fama–French three-factor** benchmark (not CAPM): (1) failure probability, (2) Ohlson's O-score, (3) net stock issues, (4) composite equity issues, (5) total accruals, (6) net operating assets, (7) momentum, (8) gross profitability, (9) asset growth, (10) return on assets, (11) investment-to-assets.

Construction (lines 359–368): *"we obtain **value-weighted** portfolio returns within each **decile** of the anomaly's sorting variable. We then construct a long-short strategy using the extreme deciles, **1 and 10, with the long leg being the higher-performing decile**… We also construct a 'combination' strategy that takes equal positions across the long-short strategies constructed in any given month."*

Sample: *"Most of the portfolio returns cover the period from August 1965 through January 2008"*; O-score and ROA begin January 1972, failure-probability begins December 1974 (lines 368–372). Benchmark-adjusted return = estimate of `a_i` in `R_{i,t} = a_i + b·MKT_t + c·SMB_t + d·HML_t + ε_{i,t}` (equation (1), lines 394–396).

Baseline anomaly magnitudes (lines 399–401): **combination strategy average benchmark-adjusted spread = 87 bp/month**; individual strategies range **43 bp (composite equity issues) to 177 bp (momentum)**.

### 2.5 EXACT RESULTS — high vs low sentiment

**Table 2 — excess returns (percent per month), Combination strategy** (`syy_nber.txt` lines 1189–1207). Table layout is by leg (Long Leg / Short Leg / Long−Short), each with columns High Sent. / Low Sent. / High−Low:

| Leg | High Sent. | Low Sent. | **High − Low** |
|---|---|---|---|
| Long Leg | 0.56 (1.72) | 0.95 (3.51) | **−0.39 (−0.93)** |
| Short Leg | −0.68 (−1.54) | 0.65 (1.96) | **−1.32 (−2.41)** |
| Long − Short | 1.23 (6.64) | 0.31 (2.64) | **+0.93 (4.25)** |

**Table 3 — FF3 benchmark-adjusted returns, Combination strategy** (lines 1454–1472):
| Leg | High Sent. | Low Sent. | **High − Low** |
|---|---|---|---|
| Long Leg | 0.30 (5.62) | 0.26 (5.40) | **+0.04 (0.62)** |
| Short Leg | −0.92 (−6.46) | −0.26 (−2.95) | **−0.66 (−3.89)** |
| Long − Short | 1.22 (7.92) | 0.52 (5.01) | **+0.70 (3.74)** |

**Table 4 — predictive regression on lagged sentiment LEVEL, excess returns, Combination strategy** (lines 1576–1582): Long Leg `b = −0.43 (−1.85)`; Short Leg `b = −0.93 (−2.90)`; Long−Short `b = 0.50 (3.79)`.
**Table 5 — same with FF3 controls, Combination strategy** (per text lines 503, 517–519): Short Leg `t = −3.01`; Long−Short `t = 2.98`; Long Leg slope **0.00, t = 0.15**.

**Cross-sectional breadth and the paper's own summary (lines 412–471):**
- H1: *"each of the long-short spreads exhibits higher average profits following high sentiment: all of the values in the last column of each table are positive."* In Table 2, **t-statistics for 8 of 11** anomalies reject the null at 5% one-sided; in Table 3, **7 of 11**. *"the long/short profits on the combined strategy in high-sentiment months account for **80%** of that strategy's profits earned across all months"* (Table 2); **70%** in Table 3.
- H2: *"the short legs of all 11 anomaly strategies have a lower average return following high sentiment, and 10 of them have t-statistics that reject the no-difference null… at a 0.05 significance level."* In Table 3, **78% of the short-leg profits across all months occur in the months following high sentiment.**
- H3: *"In Table 2, the long legs do have higher returns following low sentiment, but only 1 of the 11 anomalies (investment/assets) has a t-statistic that rejects the no-difference null… In Table 3, **none** of the t-statistics reject the no-difference null in favor of higher returns following low sentiment. In fact, **8 of the 11 differences go in the opposite direction**, although only 1 anomaly (net stock issues) has a significant one-tailed t-statistic (1.69)."*
- Predictive regressions: Table 4, spreads of **all 11** anomalies have positive slope; **9 of 11** significant at 5% one-tailed; short-leg coefficients **all 11 negative and all 11 significant**; long-leg coefficients all negative but only 5 of 11 significant, combination `−0.43 (−1.85)`. Table 5 (FF3): long-short **8 of 11** significant; short leg **8 of 11** significant; long leg **7 of 11 slopes positive, none significant**, combination slope 0.00 (t = 0.15).

### 2.6 ASYMMETRY — which leg drives it? (the critical answer)

**The effect is strongly ASYMMETRIC, and it is driven almost entirely by the SHORT leg.** The paper's own abstract (`syy_nber.txt` lines 38–41):
> *"First, each anomaly is stronger—its long-short strategy is more profitable—following high levels of sentiment. Second, **the short leg of each strategy is more profitable following high sentiment**. Finally, **sentiment exhibits no relation to returns on the long legs** of the strategies."*

Quantitatively, on the FF3 benchmark-adjusted combination strategy (Table 3, the cleaner specification):
- Short leg contributes **−0.66** of the **+0.70** bp high-minus-low difference — i.e. **~94% of the effect** comes from the short leg.
- Long leg contributes **+0.04** (t = 0.62) — economically zero.
- On raw excess returns (Table 2): short leg **−1.32 (t = −2.41)**, long leg **−0.39 (t = −0.93)** — ~77% from the short leg, and the long leg is statistically insignificant.

In the sentiment-level predictive regressions, the same asymmetry: Table 5 combination long leg slope **0.00 (t = 0.15)** vs short leg slope **−0.92 (t = −3.01)** (excess-return Table 4: −0.43 vs −0.93).

**Mechanism the authors give:** with short-sale impediments, the marginal (most optimistic) investor sets the price; high sentiment ⇒ overpricing ⇒ the *short* leg's stocks are overpriced and earn low subsequent returns; underpricing of the long leg is not expected because the rational valuation is already the most optimistic view in low-sentiment states.

### 2.7 Answers to the specific follow-up questions on SYY (stated unambiguously)

1. **Timing: strictly LAGGED one month; never contemporaneous.** The regression is literally `R_{i,t} = a + b·S_{t−1} + u_t` (line 1479–1482) / `R_{i,t} = a + b·S_{t−1} + c·MKT_t + d·SMB_t + e·HML_t + u_t` (line 1589–1592). The dummy version: *"A high-sentiment month is one in which the value of the BW sentiment index in the **previous month** is above the median"* (lines 406–409).
2. **LEVEL, not change.** *"S_t is the **level** of the investor-sentiment index of Baker and Wurgler (2006)"* (line 1484). `S` is standardised to zero mean / unit sd (line 489); returns are in percent per month.
3. **Median: FULL-SAMPLE — this is look-ahead.** *"above the median value **for the sample period**"* (line 407). Do not copy this into a live test; use an expanding/rolling median of `S_{t−1}` (coded in `ashare_cgo_recipe.py::classify_sentiment_realtime`).
4. **SIGN, in plain words: after HIGH sentiment the SHORT-LEG stocks FALL MORE, so the anomaly becomes MORE profitable, and the effect is essentially all short-leg.** Benchmark-adjusted combination (Table 3): short leg −0.92 (t = −6.46) high vs −0.26 (t = −2.95) low, difference **−0.66 (t = −3.89)**; long leg difference **+0.04 (t = 0.62)**; long-short difference **+0.70 (t = 3.74)**. Raw excess returns (Table 2): short leg **−0.68 (t = −1.54)** after high sentiment — an outright *negative* excess return — vs **+0.65 (t = 1.96)** after low, difference **−1.32 (t = −2.41)**.
5. **⚠️ Leg-assignment trap:** SYY's long leg is *"the higher-performing decile (as reported by previous studies and confirmed in our sample period)"* (lines 360–362), **not** decile 10. For some anomalies the short leg is decile 10 and for others decile 1. Assigning legs by decile number rather than by the anomaly's documented direction flips the sign.
6. **Standard errors: White (1980) HC only — NOT Newey–West** (lines 957–958, 1223, 1488–1489, 1598–1599). **No panel regression is used** — one time-series regression per leg on an aggregate return series.

---

## 3. PAPER 3 — COOPER, GUTIERREZ & HAMEED (2004): market-state conditioning

> Source: the **published JF article**, `papers/cgh_jf2004.txt`. Line numbers refer to it.

### 3.1 The EXACT UP/DOWN market-state definition

**Authoritative statement — Table I note (lines 298–307), verbatim:**
> *"At the beginning of each month t, all NYSE and AMEX firms are allocated into deciles based on their lagged six-month returns (from t − 5 to t − 1, skipping month t). Stocks priced under $1 at time t are excluded. **Non-negative (negative) returns of the VW CRSP index over months t − 36 to t − 1 define UP (DOWN) markets.** Profits of the momentum portfolios (winner minus loser deciles) are cumulated across three holding periods: months t + 1 to t + 6, months t + 1 to t + 12, and months t + 13 to t + 60."*

And in the body (lines 249–259):
> *"at the beginning of each portfolio's testing period, we identify the state of the market… For the initial discussion, we employ the return on the CRSP value-weighted index (including dividends) over the 36 months prior to the beginning of the strategy's holding period. If the market's three-year return is non-negative (negative), we define the state of the market as 'UP' ('DOWN'). We also consider a two-year and a one-year definition of the market's state… the findings are robust."*

**Exact answers to item 12:**
- **Lookback window: 36 months**, specifically **months t−36 through t−1** — i.e. ending the month *before* the formation month t, and skipping month t entirely.
- **Threshold: zero.** *"Non-negative"* = **UP** (so exactly zero counts as UP); *"negative"* = **DOWN**. It is the **cumulative (compound) return** over the window, not an average.
- **Measured at the portfolio-formation date**, using only data through **t−1**. There is a **one-month gap** (month t is skipped for bid-ask-bounce reasons, exactly as in the momentum ranking), so the state variable is **strictly real-time — no look-ahead.**

**Alternative windows tested (Section II.B, lines 885–908):** 24-month and 12-month lagged market returns. UP/DOWN counts: **124 DOWN out of 798 months (36-month)**, **219 DOWN months = 27% of the sample (12-month)**; *"The number of DOWN states increases as the number of months defining the market's state decreases."* Results are robust, and in fact *"defining the state of the market with lagged two-year or one-year market returns provides strong evidence that UP-market momentum is greater than DOWN-market momentum for the 6/6 strategy even when using the three-factor model."*

### 3.2 How the "market portfolio" is defined, and the timing (item 14)

- **Market portfolio = the CRSP value-weighted index *including dividends*.** Verbatim (line 254–255): *"the return on the CRSP value-weighted index (including dividends) over the 36 months prior to the beginning of the strategy's holding period."*
- **Not NYSE-only.** The *test assets* are NYSE + AMEX firms (line 298: *"all NYSE and AMEX firms"*; line 181: *"all NYSE and AMEX stocks listed on the CRSP monthly file"*), but the *state variable* is the **CRSP VW index**, which is the standard all-CRSP value-weighted market.
- **Uses only information available at the time — YES.** The window is t−36 … t−1 and the holding period begins at t+1, so there is a full month of separation between the last observation used and the first return predicted.

### 3.3 Momentum portfolio construction and the regression specification

**Data (lines 181–186):** all NYSE and AMEX stocks on the CRSP monthly file; **sample January 1926 – December 1995**; sorts at the end of each month t into **deciles** on prior six-month returns; formation returns from **t−5 to t−1, skipping month t** *"to mitigate bid–ask bounce effects"*; stocks with formation-period price **below $1 excluded**. Long the highest decile (winners), short the lowest decile (losers).

**Profit construction (lines 195–240):** a time series of raw profits is formed for each of the 60 event months; CAPM / FF3 risk-adjusted profits are built event-month by event-month as `r_adj_kt = r_kt − Σ_i β̂_ik f_it` (their equation (1)); monthly profits are then cumulated into CARs over (K1,K2) = (1,6), (1,12), (13,60) (their equation (2)). Because the CARs overlap, they use a **heteroskedasticity- and autocorrelation-consistent (HAC) variance estimate (Gallant 1987) with the number of lags set equal to the number of overlapping months — 5, 11, or 47.**

**Regression specification (lines 260–269) — the exact conditioning test:**
> *"To test if the mean profits are equal to zero in each state respectively, we regress the time-series of CARs on an UP dummy variable and a DOWN dummy variable, with no intercept. To test if the mean momentum profits following UP and DOWN markets are equal, we regress the time-series of CARs on an UP dummy variable and an intercept."*

And: *"Throughout the paper, we convert profits (and coefficients of reported regressions) to monthly figures by dividing by six."*

This is a **pure time-series regression on state dummies** — the same architecture as SYY's dummy specification, not a panel.

### 3.4 EXACT results (item 13)

**Table I, 1929:01–1995:12, 36-month state (lines 311–381).** Mean monthly profits; N = 674 UP months, 124 DOWN months.

| | **UP markets** (N=674) | | | **DOWN markets** (N=124) | | |
|---|---|---|---|---|---|---|
| Holding period | Months 1–6 | Months 1–12 | Months 13–60 | Months 1–6 | Months 1–12 | Months 13–60 |
| **Raw mean profit** | **0.93 (8.41)** | **0.72 (9.43)** | **−0.36 (−3.23)** | **−0.37 (−0.65)** | **−0.50 (−1.44)** | **−0.67 (−3.19)** |
| CAPM alpha | 1.12 (9.84) | 0.87 (10.90) | −0.28 (−2.31) | 0.01 (0.03) | −0.15 (−0.49) | −0.51 (−2.61) |
| FF3 alpha | 1.28 (10.80) | 1.05 (11.87) | −0.15 (−1.77) | 0.64 (1.44) | 0.42 (1.69) | −0.11 (−0.76) |

**Table I Panel C — test of equality (UP − DOWN = 0), t-statistics only:** raw profit **(2.26)** for months 1–6, **(3.47)** for months 1–12, **(1.54)** for months 13–60; CAPM alpha (2.22), (3.22), (1.19); FF3 alpha **(1.40)**, **(2.36)**, (−0.29).

**Reading:** momentum is **0.93%/month following UP markets (t = 8.41)** and **−0.37%/month following DOWN markets (t = −0.65, insignificant)**; the difference is significant for the 6/6 and 6/12 strategies on raw and CAPM profits (t = 2.26 and 3.47), but **only for 6/12 under FF3** (t = 2.36) — the paper says so explicitly: *"there is less evidence that momentum profits are greater following UP markets than DOWN markets when using the Fama–French model. The differences are only significant for the 6/12 strategy."*

**Long-run reversal (their key second result):** *"Following UP markets, there is significant evidence of long-run reversal…"* — the UP-market 6/6 profit of +0.93% turns into **−0.36% (t = −3.23)** over months 13–60 (FF3: −0.15, t = −1.77). *"The up-market momentum reverses in the long-run."*

**Continuous version (Table V, lines 941–956):** regressing momentum profits on the lagged 36-month market return and its square gives a **positive** coefficient on the level and a **negative** coefficient on the square — *"profits do not increase linearly with lagged market returns"*; profits *"peak at the median levels of lagged market return."*

### 3.5 ⚠️ CRITICAL FOR A-SHARES: the effect is REVERSED in China

Evidence from Cheema & Nartea (2017), *"Momentum returns, market states, and market dynamics: Is China different?"*, *International Review of Economics & Finance* 50, 85–97 — read from the **accepted manuscript** at `https://researchcommons.waikato.ac.nz/bitstream/10289/11238/5/IREF_accepted.pdf` (`papers/cheema_china.txt`). **This is a different paper from Cooper et al.; it is cited here as the China-specific replication.**

Their rule (lines 358–360) is CGH's: *"We employ value-weighted market returns at the portfolio formation date to define the market state. If the lagged 36-month value-weighted market return is non-negative (negative), then the market state is classified as UP (DOWN)."* Sample **January 1995 – December 2015**, **A-shares only** (line 236–239: B-shares excluded as ~0.5% of market cap); VW market returns from 1993–1995 used to seed the 36-month lag.

**Table 3 Panel A — China, 36-month UP and DOWN markets** (percent per month, t in parentheses):

| | E-W UP (A) | E-W DOWN (B) | E-W **A−B** | V-W UP (A) | V-W DOWN (B) | V-W **A−B** |
|---|---|---|---|---|---|---|
| N | 166 | 74 | | 166 | 74 | |
| P1 (Losers) | 2.51 (2.96) | 0.82 (0.77) | | 1.67 (2.15) | 0.55 (0.59) | |
| P5 (Winners) | 2.26 (2.65) | 1.89 (2.15) | | 1.44 (1.82) | 1.74 (2.25) | |
| **P5−P1** | **−0.25 (−1.01)** | **1.07 (2.11)** | **−1.33 (−2.62)** | **−0.23 (−0.71)** | **1.19 (2.17)** | **−1.42 (−2.30)** |

**This is the EXACT OPPOSITE of the US result.** In China, momentum returns follow **DOWN** markets (1.07%/mo, t = 2.11 EW; 1.19%, t = 2.17 VW) and are **absent or slightly negative** after UP markets (−0.25%, t = −1.01; −0.23%, t = −0.71). The UP-minus-DOWN difference is **−1.33% (t = −2.62) EW** and **−1.42% (t = −2.30) VW**. With 24-month states the DOWN-market advantage is 0.70% (EW) / 0.89% (VW) per month; the pattern also holds at 12 months.

**Note which leg drives the Chinese reversal:** the *loser* leg. P1 losers earn **2.51% after UP markets vs 0.82% after DOWN** (EW), while winners are nearly identical (2.26 vs 1.89). So the Chinese effect is a **loser-leg** phenomenon — the mirror image of SYY's US short-leg asymmetry. The authors attribute the absence of UP-market momentum to lottery-seeking demand for small/loser stocks after market gains, which makes winners and losers equally attractive.

**Bottom line for the parent's A-share test:** do **not** expect CGH's UP-market momentum. The published China evidence says the sign flips. A reversed sign in an A-share market-state test is the **documented** result, not necessarily a coding error.

---

## 4. PAPER 4 — BAKER & WURGLER (2006): the SENTIMENT index recipe

> **Version warning:** from **NBER WP 10449 (April 2004)**, not the published JF typeset version (Wiley 403, no mirror). The WP has **no Appendix**; all construction detail is in Section IV.C (from line 485) plus Table 2 / Figure 1 notes. The WP never mentions varimax or any rotation — the index is the **unrotated** first principal component.

### 4.1 The six proxies — exact list, construction, sign, loadings, timing

All six are **annual**, 1962–2001 (*"The sentiment proxies are measured annually from 1962 through 2001"*, line 498).

| # | Symbol | Construction (exact) | Frequency / source | Direction | Loading | Timing **in** the index (year *t*) | Timing matched to **returns** of year *t* |
|---|---|---|---|---|---|---|---|
| 1 | **CEFD** | *"the average difference between the NAV of closed-end stock fund shares and their market prices"*; year-end **value-weighted** average discount, closed-end **stock** funds | annual, year-end; Neal & Wheatley (1998) 1962–93; CDA/Wiesenberger 1994–98; turn-of-year WSJ 1999–2001 | **High = LOW sentiment** | **−0.358** | `CEFD_t` | `CEFD_{t−1}` |
| 2 | **TURN** | NYSE turnover = reported share volume ÷ average shares listed, **detrended**: *"the log of the ratio is detrended by its moving average"*, *"the past five-year average"* | annual; NYSE Fact Book | **High = HIGH sentiment** | **+0.402** | `TURN_{t−1}` | `TURN_{t−2}` |
| 3 | **NIPO** | *"the annual number of initial public offerings"* — a **count**, unscaled | annual; Jay Ritter's website (updating Ibbotson, Sindelar & Ritter 1994) | **High = HIGH sentiment** | **+0.414** | `NIPO_t` | `NIPO_{t−1}` |
| 4 | **RIPO** | *"the average annual first-day returns of initial public offerings"* (%) | annual; Jay Ritter's website | **High = HIGH sentiment** | **+0.464** | `RIPO_{t−1}` | `RIPO_{t−2}` |
| 5 | **S** | *"gross equity issuance divided by gross equity plus gross long-term debt issuance"*, Federal Reserve Bulletin. Footnote 7: *"includes seasoned offerings… and scales by total external finance to isolate the composition of finance from the level."* | annual | **High = HIGH sentiment** | **+0.371** | `S_t` | `S_{t−1}` |
| 6 | **PD-ND** | *"the year-end log difference of the average market-to-book ratios of payers and nonpayers"* — year-end log(**VW avg M/B payers**) − log(**VW avg M/B nonpayers**) | annual, year-end; Baker & Wurgler (2003) data | **High = LOW sentiment** | **−0.431** | `PD-ND_{t−1}` | `PD-ND_{t−2}` |

**Reconstructed equation (2):**
```
SENTIMENT_t = −0.358·CEFD_t + 0.402·TURN_{t−1} + 0.414·NIPO_t
              + 0.464·RIPO_{t−1} + 0.371·S_t − 0.431·(PD−ND)_{t−1}
```
Line 629–630: *"each individual sentiment proxy enters with the expected sign."* Loadings are **not** equal-weighted (|0.358| to |0.464|, largest on RIPO, smallest on CEFD).

### 4.2 How the index is built — the exact procedure

**Step 1 — Orthogonalise FIRST, before the PCA. YES.** Verbatim (lines 499–505):
> *"To isolate the sentiment component of the proxies from business cycle components, we orthogonalize each proxy with respect to several macroeconomic variables. Specifically, we regress each of the raw proxies on growth in the industrial production index (Federal Reserve Statistical Release G.17), growth in consumer durables, nondurables, and services (all from BEA National Income Accounts Table 2.10), and a dummy variable for NBER recessions. We take the residuals from these regressions as a cleaner proxy that is independent of major business cycle effects. All of our results are based on these cleaned proxies."*

> **⚠️ The WP is internally inconsistent about the macro list.** The body text lists **three** regressors. The **Table 2 note (line 2028) and Figure 1 note (lines 1402–1403) list four**, adding **"the growth in employment"** (no source given). A replicator must decide or test both. **The regression frequency is never stated** (annual, 40 obs, is the strong implication); no equation, coefficients, or N given.

**Step 2 — Detrend TURN only** (log turnover minus its 5-year moving average). *"Turnover displays an exponential positive trend over our period, however, and the May 1975 elimination of fixed commissions have a visible effect. As a partial solution, the raw turnover ratio TURN is detrended by the five-year moving average"* (lines 524–529). Window alignment ("past five-year average" — includes *t*? centred?) is **not stated**.

**Step 3 — Standardise** the cleaned components: *"each of the six components was first cleaned of macroeconomic conditions and standardized"* (lines 624–625).

**Step 4 — TWO-STAGE PCA.** Verbatim (lines 567–574):
> *"We start by estimating the first principal component of the six proxies and their lags. This gives us a first-stage index with twelve loadings, one for each of the current and lagged proxies. We then compute the correlation between the first-stage index and the current and lagged values of each of the proxies. Finally, we construct SENTIMENT as the first principal component of the **correlation matrix** of six variables – each proxy's lead or lag, whichever has the higher correlation with the first-stage index."*

- **First stage** = first PC of the 6 proxies at *t* **and** at *t−1* → **12 loadings**.
- **Selection rule:** per proxy keep whichever of `proxy_t`, `proxy_{t−1}` correlates more with `F1`. Six winners survive; they are fixed by Eq. (2) and the Table 2 row labels: **CEFD_t, TURN_{t−1}, NIPO_t, RIPO_{t−1}, S_t, PD-ND_{t−1}**.
- **Final** = first PC of the **correlation matrix** of those six.

**Step 5 — Reported properties** (lines 625–628): **PC1 explains 53%** of the standardised sample variance; **only the first eigenvalue is above 1.00**; **corr(12-term first-stage index, 6-term SENTIMENT) = 0.96**.

**Step 6 — Sign, scaling, updating, use.** Positive = optimistic sentiment (hence CEFD and PD-ND load negatively). No rotation; no sum-normalisation or rescaling described. **Annual** updating. **Monthly cross-sectional returns of calendar year *t* are conditioned on `SENTIMENT_{t−1}`** (Table 3b header is literally `SENTIMENTt-1`, line 2583).

**Step 7 — Total lag from raw proxy to predicted return.** CEFD, NIPO, S use year **t−1**; TURN, RIPO, PD-ND use year **t−2**; the index itself is used as `SENTIMENT_{t−1}`. Verbatim (lines 3177–3179): *"Coefficients are matched to the closed-end fund discount (CEFD), the number of IPOs (NIPO), and the equity share (S) for the calendar year one year before t. Coefficeints [sic] are matched to detrended log turnover (TURN), the average annual first-day return (RIPO), and the dividend premium (PD-ND) for the calendar year two years before t."*

**Step 8 — Individual proxy correlations with SENTIMENT** (Table 2). Panel B (**cleaned** — these residuals feed the index): CEFD **−0.64**, TURN **0.72**, NIPO **0.74**, RIPO **0.83**, S **0.66**, PD-ND **−0.77** (all significant at 1%). Panel A (raw): CEFD −0.60, TURN 0.69, NIPO 0.66, RIPO 0.80, S **0.43**, PD-ND −0.76. **The two weakest are consistently S and CEFD; the strongest are RIPO and PD-ND.** ⚠️ These are in-sample correlations with an index that *contains* the proxy, so they are inflated; the WP reports **no leave-one-out** correlations.

**Step 9 — Main result / which stocks are sentiment-prone** (lines 1195–1205): *"when sentiment appears to be high, stocks that are likely to be relatively attractive to optimists and speculators and at the same time unattractive candidates for arbitrage – **young stocks, small stocks, unprofitable stocks, non-dividend paying stocks, high-volatility stocks, extreme-growth stocks, and distressed stocks** – experience low future returns relative to other stocks. Conditional on low sentiment, on the other hand, these cross-sectional patterns attenuate or reverse."* Also: *"those using IPO volume as the sentiment proxy are particularly strong"* (lines 892–893); *"a two-SD increase in the sentiment index raises the size coefficient by 0.2 percentage points per month"* (lines 897–899).

### 4.3 Is CEFD essential?

It is the one proxy a non-US replicator generally cannot build, and it is also the proxy the paper singles out (lines 678–683):
> *"CEFD is perhaps our cleanest 'general' indicator of investor sentiment – it is not mechanically connected to sentiment for any one segment of the cross-section. In contrast, the volume of IPOs and the dividend premium could in principle be driven by a concentrated demand for firms with IPO characteristics or dividends, respectively."*

That is why Table 3a re-runs the headline sorts on CEFD alone. The WP reports **no without-CEFD index**, so it provides no evidence on how much changes if it is dropped. **Dropping it produces a BW-*style* variant, not BW's SENTIMENT** — and this should be labelled as such.

---

## 5. A-SHARE PROXY FEASIBILITY FOR EACH BAKER-WURGLER COMPONENT

Constraint set: **daily OHLCV + turnover rate + float market cap + shares outstanding + listing date + industry only.**

| BW proxy | Feasible? | Why / how |
|---|---|---|
| **CEFD** | **NO — not reproducible and not approximable** | Needs listed closed-end funds with both market price and published NAV, value-weighted, year-end, stock funds only. Nothing in the constraint set substitutes. *(If fund data were allowed — e.g. tushare `fund_daily` + `fund_nav` for 封闭式基金 — this becomes buildable. Under the stated constraints, it does not.)* |
| **TURN** | **YES** | Market turnover = Σ volume ÷ Σ shares listed. From the available fields: Σ(turnover_rate_i × shares_i) ÷ Σ shares_i (a shares-weighted mean of turnover rates). Then take **logs** and subtract a **5-year (or 12-month) moving average**. Critical in China: the A-share turnover series is dominated by structural breaks (2005 股权分置改革, 2007 bubble, 2015 leverage episode, 2016 熔断, registration-based IPO reform), so detrending is not optional. |
| **NIPO** | **YES — counting new listings per period is exactly BW's NIPO** | BW's NIPO is the **annual count**, unscaled. Count `list_date` per period from the stock list. ⚠️ **Caveat specific to China:** IPO supply is **administratively gated** by the CSRC (2013 full suspension, 2015–16 slowdown, 2023–24 tightening). So Chinese NIPO reflects **regulatory supply policy at least as much as investor sentiment**, which attenuates BW's identifying logic (firm supply responds to sentiment with a lag). Consider substituting IPO *proceeds* or the count of *accepted applications*, or at minimum flag the caveat. |
| **RIPO** | **ONLY IF the issue price is available — and it is regime-contaminated** | First-day return = (first-day close − issue price) ÷ issue price. Post-listing OHLCV bars alone will **not** give you the issue price; you need the listing/issue announcement. ⚠️ **In China the first-day return is capped by regulation** (44% first-day limit since 2014 on the main boards; no limit for 科创板/创业板 registration-based IPOs since 2019). Its time-series variation is therefore driven substantially by **price-limit regime changes**, not by sentiment. Treat with caution. |
| **S** (equity share in new issues) | **NO — not as BW define it. See the specific answer below.** | Denominator requires aggregate **long-term debt** issuance. |
| **PD-ND** | **NO** under the stated set | Needs book value (for M/B) and dividend records. *(If `daily_basic.pb` + `dividend` data were allowed: PD−ND = log(VW avg pb of payers) − log(VW avg pb of non-payers), "payers" = firms paying a dividend in period t−1. Then it is feasible. Also flag that Chinese dividend policy has strong regulatory-trend components — 分红指引 2013, 新国九条 2024 — so it must be detrended.)* |

**Net: 2 of 6 directly buildable (TURN, NIPO); 1 conditionally and contaminated (RIPO); 3 not buildable (CEFD, S, PD-ND).**

### 5.1 Specific question: can "equity share in new issues" be proxied by Δ(shares outstanding) ÷ market cap?

**No — that is a different variable, and using it as BW's `S` would be a mislabelling.** Three independent reasons:

1. **The denominator is the whole point.** BW's `S` = gross equity ÷ (**gross equity + gross long-term debt**). Footnote 7 says it *"scales by total external finance to isolate the composition of finance from the level."* Dividing Δshares by **market cap** replaces the composition denominator with a *size* denominator — it reintroduces the level and discards exactly the content BW designed the variable to carry. Aggregate long-term debt issuance is not in the constraint set.
2. **Δshares is a different economic object.** BW's numerator is **gross equity issuance including seasoned offerings**; Δshares outstanding is a **net** quantity. Δlog(shares) over 12 months is the well-known **Net Stock Issuance** anomaly variable — which is **anomaly #3 in Stambaugh-Yu-Yuan (2012)**, an *individual-stock mispricing characteristic*, not a market-level sentiment proxy. It is a legitimate and useful factor, but it is a different thing. Do not label it "equity share in new issues."
3. **Δshares is contaminated by non-issuance events in China.** **送股/转增股本** (bonus share issues and capitalisation of reserves) change shares outstanding without any external financing, and 高送转 is both extremely common and itself a sentiment/attention phenomenon. Splits, share-based M&A and 限售股解禁 add further noise. Δshares in A-shares would be substantially a **高送转 indicator**, not an issuance indicator. (If you use it at all, you must first strip 送转 events using the 除权除息 records.)

**Correct fix if you want a genuine BW-style `S`:** obtain aggregate **社会融资规模** (Total Social Financing) broken into equity financing vs. total, or gross equity issuance vs. gross long-term debt issuance, from macro data. That is outside the stated constraint set but is the only route to BW's actual variable.

**Best legitimate use of what you have:** build **Net Share Issuance** = Δlog(shares outstanding), 送转-adjusted, as a *cross-sectional anomaly characteristic* (SYY anomaly #3) — not as a sentiment component.

### 5.2 Specific question: can IPO volume/first-day returns be proxied by counting new listings per month?

- **NIPO: yes, exactly.** Counting new listings per period *is* BW's NIPO (they use annual counts, unscaled). Monthly counting is a frequency change, which is a reasonable adaptation for a shorter Chinese sample — but note BW's index is **annual**, so a monthly version is already a variant.
- **RIPO: not from the count.** First-day return needs the issue price, which is not in a post-listing OHLCV panel. If issue prices are obtainable, note the price-limit regime caveat in §5.

### 5.3 Two structural caveats on transplanting BW to A-shares

1. **BW's index is ANNUAL (40 observations, 1962–2001).** A monthly A-share index is a frequency adaptation; PCA loadings on monthly data will differ, and the 53%-of-variance benchmark will not transfer. With 5–6 proxies and a short monthly sample, the first PC is also much noisier.
2. **CEFD is missing**, and it is the only component BW describe as free of single-segment mechanical ties. Any Chinese index built on TURN + NIPO (+ RIPO) is a **BW-style variant** and will be dominated by turnover and IPO activity.

---

## 6. IMPLEMENTABLE TAKEAWAYS FOR A-SHARE MARKET WITH LIMITED DATA

### 6.0 Available vs. unavailable

**Available:** daily OHLCV bars · turnover rate · float market cap · shares outstanding · industry classification · ST status · listing date · market index level.
**Not available (reliably):** analyst data · fund flows · IPO first-day returns (issue prices) · book value / dividends · aggregate debt issuance · closed-end fund NAV.

### 6.1 Which measures are buildable — verdict table

| Measure | Source paper | Buildable? | Notes |
|---|---|---|---|
| **Capital gains overhang (CGO)** | Grinblatt & Han (2005) | **YES — fully, and faithfully** | Needs only adjusted prices + turnover. See §6.2. |
| `V̄` (52-week average turnover) control | Grinblatt & Han | **YES** | Turnover rate available. |
| Size control `s = log(mktcap)` | Grinblatt & Han | **YES** | Use **float** market cap for A-shares. |
| `V_{−1}·g` interaction | Grinblatt & Han | **YES** | Doubles the t-stat in GH (0.4876, t = 15.64). Cheap to add. |
| **Market-state UP/DOWN conditioning** | Cooper, Gutierrez & Hameed (2004) | **YES** | Uses only the market index level. See §6.4. |
| **Sentiment-conditioned anomalies** | Stambaugh, Yu & Yuan (2012) | **PARTIAL** | The methodology is fully replicable; the *proxy* must be replaced. See §6.3. |
| **BW sentiment index** | Baker & Wurgler (2006) | **NO (as specified)** | Only TURN and NIPO buildable; CEFD, S, PD-ND are not. Any version is a BW-*style* variant. |

### 6.2 Concrete recipe — Capital Gains Overhang from daily prices + turnover only

**Everything below is verified numerically** (`papers/cgo_validate.py`, `papers/cgo_closed_form.py`).

**Step 1 — Prices.** Use **后复权 (backward-adjusted)** close. This is essential and is an *inference*: GH never discuss splits (zero occurrences of "split"/"missing" in either version), but `g = (P − R)/P` is a ratio of a 260-period price average to a recent price, so an unadjusted split inside the window destroys the measure.

**Step 2 — Grid.** Resample to **weekly (W-FRI)**: `P` = last adjusted close of the week; `V` = **sum** of daily turnover rates within the week ÷ 100 (turnover is a *flow* — sum, not mean). GH use weekly deliberately: *"This requires higher frequency data than monthly data provide and transaction prices that are less influenced by market microstructure than daily data provide."* A daily (1250-day window) version is a defensible adaptation for A-shares and avoids weekday/holiday alignment problems; if you use it, keep the window at 5 years.

**Step 3 — Reference price.** GH's eq. (4)/(25), truncated at *N* = 260 and normalised by *k*:
```
R_t = (1/k) · Σ_{n=1..N} [ V_{t−n} · Π_{τ=1..n−1}(1 − V_{t−n+τ}) ] · P_{t−n}
k   =        Σ_{n=1..N} [ V_{t−n} · Π_{τ=1..n−1}(1 − V_{t−n+τ}) ]
```
**Do NOT implement the double loop.** The products telescope. With `cumS_m = Π_{j≤m}(1 − V_j)` and `Z_m = V_m / cumS_m`, the truncated, *k*-normalised reference price is **exactly** a Z-weighted moving average of the last *N* prices:
```
R_t = [ Σ_{m=t−N..t−1} Z_m · P_m ] / [ Σ_{m=t−N..t−1} Z_m ]
```
I verified this closed form against the brute-force double loop to **2.3e-14** absolute error (machine precision). Compute `Z` **relative to each window's start** to avoid underflow of `Π(1−V)` over long histories (the common factor cancels in the ratio).

**Step 4 — the CGO regressor.**
```
g_{t−1} = ( P_{t−2} − R_{t−1} ) / P_{t−2}
```
GH lag the **price** one extra period relative to the reference price, *"to avoid confounding market microstructure effects, such as bid-ask bounce."*

**Step 5 — Initialisation.** **Do not seed `R` with an arbitrary value.** GH never do: *"Stocks that lack at least five years of historical return and turnover data at a particular week are excluded from the cross-sectional regression for that week."* Require **N consecutive periods of both price and turnover** and drop stock-weeks that fail. Robustness: GH confirm results are similar with 3 or 7 years.

**Step 6 — Missing turnover / zero-volume weeks.** `V = 0` (the reference price simply does not update) — this is the model's own semantics (if nothing trades, no cost basis changes). Missing *price* → carry `R` forward. ⚠️ **GH themselves say nothing about missing data or splits**; these are the model-implied choices, not quotations.

**Step 7 — Normalisation `k` is immaterial for A-shares.** Verified: with constant weekly turnover of 1% (the GH sample mean), `k ≈ 0.927`; at **2%/week `k ≈ 0.995`**; at 5%/week `k ≈ 0.999998`. A-share turnover routinely exceeds 2%/week, so `k ≈ 1` and the simple recursion `R ← V·P + (1−V)·R` is adequate. Still normalise for correctness on low-turnover names.

**Step 8 — Test.** Fama–MacBeth cross-sectional regression each week (GH eq. 5):
```
r = a0 + a1·r_{−4:−1} + a2·r_{−52:−5} + a3·r_{−156:−53} + a4·V̄ + a5·s + a6·g
```
Coefficients averaged over weeks; **t-statistics from the time series of the weekly coefficient estimates**. Controls: `V̄` = average weekly turnover over the prior 52 weeks; `s` = log market cap.

**Step 9 — ⚠️ SEPARATE JANUARY.** The CGO sign **reverses in January** in GH: `g`'s coefficient is +0.0040 (t = 7.79) overall and +0.0050 (t = 9.42) for Feb–Nov, but **−0.0117 (t = −4.95) in January**. GH attribute this to December tax-loss selling: *"the January returns of high capital gain stocks tend to be below those of the low capital gain stocks."* China has **no capital-gains tax on A-share equity trades**, so this mechanism is absent — **test whether the January reversal exists in A-shares at all; do not assume it.** (There may still be a Spring Festival / window-dressing seasonality, but that is a different mechanism.)

**Step 10 — Sign convention.** **LONG HIGH-CGO, SHORT LOW-CGO.** `E[R] = (1−w)·V_t·(P_t−R_t)/P_t` is *increasing* in the gain. There is no sign flip.

**Step 11 — Universe screens.** GH: ordinary common shares only (CRSP sharecode 10/11), NYSE + AMEX, ≥5 years of history, **equal-weighted** portfolios, quintile sorts on all included stocks (no NYSE-only breakpoints). A-share analogues: exclude **ST/\*ST** (5% price limit and distorted turnover; GH have no analogue, so this is an adaptation, and you should test both ways); exclude **科创板 (688/689)** and **北交所** if you want a homogeneous universe; exclude the first ~1 year after listing (次新股 limit-up sequences make turnover meaningless — this is stricter than GH's 5-year screen and is subsumed by it).

**Step 12 — Cheap extensions GH support.** (a) Add `V_{−1}·g` — GH: *"if it were used in Table 3 Panel C in place of the gain variable it approximately doubles the t-statistic."* (b) Add a 52-week-high reference price alongside `g` — GH find **both** significant, *"what we would expect if the model was correct and both variables are imperfect proxies."*

### 6.3 Concrete recipe — a market-level sentiment proxy from market data only

**Honest scope:** this is a **BW-style variant**, not BW's SENTIMENT, because CEFD, S and PD-ND are unavailable.

**Four proxies, all from the constraint set:**
| | Proxy | BW analogue | Construction |
|---|---|---|---|
| P1 | Market turnover, **detrended** | `TURN` | `log(Σ(volume_i) / Σ(shares_i))` or the shares-weighted mean turnover rate; subtract a **5-year (or 12-month) moving average**. BW: *"the log of the ratio is detrended by its moving average"*. |
| P2 | **Count of new listings** in the period | `NIPO` | Count of `list_date` in the period. ⚠️ administratively gated in China — see §5. |
| P3 | **Index level ÷ its 12-month moving average − 1** | stand-in for `CEFD` | CEFD is a valuation-based gauge of investor optimism; the index-vs-trend gap is the closest market-data-only analogue. This is **my substitution, not BW's variable**. |
| P4 | **Limit-up ratio** (optional, A-share-specific) | none | Share of stocks hitting the up-limit among all limit events (or the cross-sectional dispersion of daily returns). Fully buildable from OHLCV + the limit rules. |

**Procedure (mirrors BW's steps):**
1. **Orthogonalise each proxy on macro controls FIRST**, before the PCA — this is what BW do. Use whatever is obtainable: industrial production growth, CPI/PPI, M2 growth, an NBER-style recession dummy (China: use a dated recession/credit-crunch indicator). Keep the **residuals**. (BW's list — industrial production growth, consumption growth, *[employment growth]*, NBER recession dummy — is not all available for China; use the obtainable subset and document it.)
2. **Standardise** the residuals (z-scores).
3. **Two-stage PCA.** First stage: first PC of the proxies at *t* **and** *t−1* (2×k loadings). Then per proxy keep whichever of the two correlates more with the first-stage index. Final: first PC of the **correlation matrix** of the selected series.
4. **Sign** it so that **higher = more optimistic**, verified by `corr(PC1, detrended turnover) > 0`.
5. Report the variance explained by PC1 (BW's benchmark is 53% with six proxies; expect lower with fewer, noisier proxies — do not force the comparison).
6. **Timing: use `S_{t−1}`** for returns in month *t* (SYY's convention), and **use an expanding or rolling median, NOT the full-sample median**, to classify high/low — SYY's full-sample median is look-ahead.

**A better-fitting alternative given the data you have:** since the GH CGO is fully buildable, the **cross-sectional mean (or median) of `g`** is itself a market-wide sentiment/market-state proxy, and GH report that the cross-sectional percentiles of `g` correlate 0.50/0.60/0.62 with the past one-year S&P 500 change (lines 458–459). That gives you a **continuous, real-time market-state conditioner built from exactly the data you have** — and it links naturally to Cooper et al.'s binary UP/DOWN state (§6.4). This is my synthesis, not a claim in either paper.

### 6.4 Market-state conditioning (Cooper, Gutierrez & Hameed) — fully implementable

**Buildable from the market index level alone.** Exact rule (CGH Table I note):

```
state_t = UP    if the index return over months [t-36, t-1] >= 0
          DOWN  if the index return over months [t-36, t-1] <  0
```
- Window: **36 months ending at t−1** (skip month t). Threshold: **zero** ("non-negative" = UP).
- Compound (cumulative) return, not an average.
- Strictly real-time: the last observation is t−1 and the holding period starts at t+1.
- Robustness: 24-month and 12-month windows also work (and the 12-month version gives *more* DOWN observations — 219/798 vs 124/798).

Test specification (CGH, lines 260–269): regress the time-series of holding-period CARs on **an UP dummy and a DOWN dummy, no intercept** (tests each state's mean against zero); then on **an UP dummy plus an intercept** (tests UP = DOWN). Profits converted to monthly by dividing by 6. **HAC/Gallant (1987) standard errors with lags = number of overlapping months (5, 11, or 47).**

Momentum construction: NYSE+AMEX, **deciles** on lagged 6-month returns **t−5 to t−1 skipping t**, exclude stocks priced under $1, long top decile / short bottom decile, holding t+1..t+6, t+1..t+12, t+13..t+60, overlapping, monthly rebalanced. A-share adaptation: quintiles on t−6..t−1, exclude the first year post-listing and ST names, and note that the 涨跌停 price-limit regime truncates the extreme returns that drive momentum.

**⚠️ Expect the REVERSED sign in A-shares — see §3.5.** Cheema & Nartea (2017) find Chinese momentum follows **DOWN** markets (+1.07%/mo EW, t = 2.11) and is absent after UP markets (−0.25%, t = −1.01), i.e. UP−DOWN = **−1.33% (t = −2.62)** — the mirror image of CGH's US +0.93%/−0.37%. The Chinese reversal is driven by the **loser leg** (P1: 2.51 after UP vs 0.82 after DOWN).

### 6.5 What NOT to claim

- Do **not** call a Chinese turnover+IPO index "the Baker-Wurgler sentiment index". It is a BW-style variant missing the component BW call their cleanest general indicator.
- Do **not** call Δshares/market cap "the equity share in new issues". It is Net Stock Issuance, a cross-sectional anomaly characteristic.
- Do **not** assume a tunable `delta` in the CGO model. There isn't one.
- Do **not** carry GH's January sign reversal into A-shares without testing — the tax-loss-selling mechanism behind it does not exist for A-share equity trades.
- Do **not** assume SYY's asymmetry (short-leg-driven) transfers to A-shares. It rests on an active short-sale market (融券 is small and intermittently restricted) and on the BW index.
- Do **not** expect Cooper et al.'s UP-market momentum in A-shares. The published China evidence finds the **opposite sign** (momentum after DOWN markets), driven by the loser leg — see §3.5.
- Do **not** attribute a varimax/rotated index to Baker & Wurgler. The index is the **unrotated** first principal component; the WP contains no mention of rotation.
