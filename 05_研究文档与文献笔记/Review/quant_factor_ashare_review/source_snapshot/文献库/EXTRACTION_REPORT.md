<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# Methodology-Level Extraction: Two JFE Papers on Volatility-Managed Momentum

Prepared from **full text**, not abstracts. Every number below is quoted from the retrieved text.
Sections I could not access are flagged explicitly in `ACCESS & HONESTY NOTES`.

---

## 0. ACCESS & HONESTY NOTES (read first)

### Paper 1 — Barroso & Santa-Clara (2015)
- **Retrieved: the complete author-accepted manuscript (the final post-refereeing text), 39 pages.**
  Page 1 header reads: *"Momentum has its moments / Pedro Barroso and Pedro Santa-Clara / Forthcoming in the
  Journal of Financial Economics / First version: April 2012 / This version: November 2014"*, and footnote 1
  states *"A previous version of this paper circulated with the title 'Managing the Risk of Momentum.'"*
- URL actually read (PDF, 444,689 bytes):
  `http://www.snifferquant.com/gyantal/Incode/papers/Momentum%20Has%20Its%20Moments(scaling%20Momentum%20by%20vol),2014.pdf`
- Local copies: `BSC2015.pdf` (original PDF) and `BSC2015.txt` (PyMuPDF text extraction, 39 pages, 53,543 chars).
- Also read: ScienceDirect article preview (abstract + editorial "Section snippets"):
  `https://www.sciencedirect.com/science/article/abs/pii/S0304405X14002566`
- **Version caveat:** this is the accepted manuscript, not the Elsevier typeset/paginated version. Equation numbers,
  table numbers and numerical results are the paper's own and match the published abstract exactly
  (0.53 → 0.97 Sharpe, 18.24 → 2.68 kurtosis, −2.47 → −0.42 skew, −78.96% → −28.40% minimum, −96.69% → −45.20%
  drawdown, 57.82% OOS R²). Equation numbering below follows the manuscript.
- One artifact: page 1 carries an unrelated reader's handwritten-style annotation inserted before the abstract.
  All paper text from page 2 onward is clean. I have ignored the annotation as non-authorial.

### Paper 2 — Cederburg, O'Doherty, Wang & Yan (2020)
- **Retrieved: the complete published JFE article**, including all tables (1–9) and Appendix A / Table A1.
  URL read: `https://www.lehigh.edu/~xuy219/research/COWY.pdf`
  Local copies: `COWY.pdf`, `COWY.txt` (146,063 chars). Header on p.1: *"Journal of Financial Economics 138 (2020) 95–117"*.
- **⚠ TWO CORRECTIONS TO THE REQUEST'S PREMISES:**
  1. **The PII in the request is wrong.** `S0304405X20301438` does **not** resolve to this paper. The correct PII is
     **`S0304405X2030132X`** — verified two ways: (a) DOI `10.1016/j.jfineco.2020.04.015` 302-redirects to
     `https://linkinghub.elsevier.com/retrieve/pii/S0304405X2030132X`; (b) Crossref returns
     `alternative-id: S0304405X2030132X` with `volume 138, issue 1, pages 95-117`.
  2. **Transaction costs are NOT part of this paper's argument.** A web-search snippet floating around attaches the
     sentence *"We investigate whether transaction costs, arbitrage risk, and short-sale impediments explain the
     abnormal returns of volatility-managed equity portfolios"* to this paper. **That sentence is not in this paper**
     and does not match its abstract. I searched the full text for `transaction cost`, `trading cost`, `costs`,
     `friction`, `arbitrage`, `short-sale`, and `turnover`: **zero hits for every cost/friction term.** The paper's
     stated driver is **structural instability in the spanning regressions**, not trading frictions. See §7.
- The paper's true abstract, verbatim: *"Using a comprehensive set of 103 equity strategies, we analyze the value of
  volatility-managed portfolios for real-time investors. Volatility-managed portfolios do not systematically
  outperform their corresponding unmanaged portfolios in direct comparisons. Consistent with Moreira and Muir (2017),
  volatility-managed portfolios tend to exhibit significantly positive alphas in spanning regressions. However, the
  trading strategies implied by these regressions are not implementable in real time, and reasonable out-of-sample
  versions generally earn lower certainty equivalent returns and Sharpe ratios than do simple investments in the
  original, unmanaged portfolios. This poor out-of-sample performance for volatility-managed portfolios stems
  primarily from structural instability in the underlying spanning regressions."*

### Route failures (for reproducibility)
OpenAlex / Semantic Scholar / Unpaywall all report **both** papers as CLOSED ACCESS. SSRN is DNS-poisoned on this
machine and returns HTTP 403 through its real Cloudflare IP (pinned via DoH: 104.16.39.248). r.jina.ai, CORE, BASE,
scholar.archive.org, ResearchGate all blocked. Firecrawl (the `read_page`/`web_search` backend) was hard
rate-limited (HTTP 429, retry_after ≈ 86,000 s). Working routes: direct download of `.edu`-hosted PDFs via
PowerShell, and driving the Playwright browser as both search engine and downloader.

---

# PAPER 1 — Barroso & Santa-Clara (2015), "Momentum has its moments"

## 1. The EXACT volatility-scaled momentum weight

Two equations define the whole strategy.

**Variance forecast (Eq. 5, p.11 of manuscript):**

```
sigma_hat^2_{WML,t}  =  21 * SUM_{j=0}^{125} r^2_{WML, d_{t-1} - j} / 126
```

**Scaled (risk-managed) return (Eq. 6, p.12):**

```
r_{WML*,t}  =  ( sigma_target / sigma_hat_t ) * r_{WML,t}
```

Reading off the components:

| Ingredient | Exact answer from the paper |
|---|---|
| **Target volatility constant** | *"We pick a target corresponding to an annualized volatility of 12%."* (p.12, and footnote 15). **`sigma_target = 12% annualized`.** |
| **Estimation window** | **126 daily returns = the previous SIX months** (`j = 0 … 125`). Appendix A: *"We use these daily returns to calculate the realized variances in the previous 21, 63, and 126 sessions at the end of each month."* |
| **Whose volatility?** | The **momentum (WML) strategy's own returns** — *not* the market's. Footnote 13 explicitly separates this from the time-series-momentum literature: *"there it serves a different purpose: use the asset-specific volatility to prevent the results from being dominated by high-volatility assets only. We do not consider asset-specific volatilities…"* |
| **Units** | Eq. (5) is a **monthly** variance (daily sum-of-squares × 21/126). To keep units consistent in the ratio, `sigma_target = 0.12/sqrt(12) ≈ 3.464%` per month. Equivalently, in annualized form: `w_t = 0.12 / sqrt(252 * (1/126) * SUM r^2) = 0.12 / sqrt(2 * SUM_{j=0}^{125} r^2)`. Both give the identical weight (verified algebraically: `(0.12/sqrt(12))*sqrt(6) = 0.12/sqrt(2)`). |
| **Window robustness** | Footnote 14: *"We also used one-month and three-month realized variances as well as exponentially weighted moving average (EWMA) with half-lifes of one, three, and six months. All work well with nearly identical results."* |
| **Look-ahead** | Footnote 7: *"This approach relies only on past data and thus does not suffer from look-ahead bias."* |

**Version note on the 21-day vs 126-day question.** The paper contains *two* realized-variance definitions and they
serve different purposes:
- **Eq. (2)** — used only in Section 3 for the *predictability* analysis: `RV_{i,t} = SUM_{j=0}^{20} r^2_{i, d_t - j}`
  (sum of squared daily returns over the **previous 21 sessions**, unscaled, non-overlapping monthly).
- **Eq. (5)** — used for the **actual weight**: the **126-day (6-month)** sum of squares.

So a literal reading of the ScienceDirect "Section snippets" preview (*"For each month, we compute the realized
variance RV_t from daily returns in the previous 21 sessions"*) describes Eq. (2), **not** the trading rule. **The
trading rule uses 126 days.** This is the single most common mis-reading of this paper.

## 2. How realized variance is computed from daily returns

**Sum of squared daily returns.** No demeaning, no `sd × sqrt(21)`.

- `RV_{i,t} = SUM_{j=0}^{20} r^2_{i,d_t-j}` (Eq. 2) — *"The realized variances are the sum of squared daily returns
  in each month"* (Table 2 caption).
- `sigma_hat^2_{WML,t} = 21 * SUM_{j=0}^{125} r^2 / 126` (Eq. 5) — i.e. the mean squared daily return over 126 days,
  times 21 (trading days per month) to express it as a monthly variance.
- Footnote 11: *"Correcting for serial correlation of daily returns does not change the results significantly."*
- Footnote 15 warns that the daily-frequency target is not directly comparable to a monthly-return annualization:
  *"The annualized standard deviation from monthly returns is higher than 12% as volatilities at daily frequency are
  not directly comparable with those at lower frequencies due to the small positive autocorrelation of daily returns."*

**Implementable translation:** `sigma_hat_t = sqrt( (21/126) * SUM_{i=1}^{126} r_{d_i}^2 )`, a monthly standard
deviation computed with an assumed zero daily mean.

## 3. Timing convention

**Yes — strictly information through the last trading day of month t−1 sets the weight for month t.**

Eq. (5) sums `r^2_{WML, d_{t-1} - j}` for `j = 0 … 125`, where `{d_t}` is defined as *"the time series of the dates
of the last trading sessions of each month"*. So the window **ends on the final trading session of month t−1** and
runs back 126 sessions. The weight is then applied to `r_{WML,t}`, the *monthly* momentum return of month t.
The paper states it *"could be implemented in real time"* and is free of look-ahead bias (footnote 7).

## 4. Reported results — exact numbers

**Table 1 & Table 3, sample 1927:03 – 2011:12 (monthly returns):**

| Statistic | WML (plain) | WML* (risk-managed) | Change |
|---|---|---|---|
| Maximum one-month return | **26.18%** | **21.95%** | −4.23 pp |
| **Minimum one-month return** | **−78.96%** | **−28.40%** | +50.56 pp |
| Mean excess return (annualized) | **14.46%** | **16.50%** | **+2.04 pp/yr** |
| Standard deviation (annualized) | **27.53%** | **16.95%** | **−10.58 pp/yr** |
| **Excess kurtosis** | **18.24** | **2.68** | −15.56 |
| **Skewness** | **−2.47** | **−0.42** | +2.05 |
| **Sharpe ratio** | **0.53** | **0.97** | **+0.44 (nearly doubles)** |
| Information ratio vs WML | — | **0.78** | — |
| Maximum drawdown | **−96.69%** | **−45.20%** | +51.49 pp |

Other exact numbers:
- The two worst momentum crashes: **1932: −91.59% over two months; 2009: −73.42% over three months.**
- *"A dollar invested in the momentum strategy in July 1932 would recover it only in April 1963, 31 years later."*
- **Persistence of risk:** AR(1) coefficient of monthly **squared** returns = **0.40** for raw momentum vs **0.14**
  for risk-managed. Autocorrelation of raw momentum is significant up to **15 lags**; for risk-managed momentum only
  **1 lag**. (`sigma_hat_t` itself has an AR(1) coefficient of **0.97** — highly persistent month to month.)
- **Decade performance:** in the 1930s pure momentum loses ~**90%** while scaled momentum preserves the investment;
  in the 2000s pure momentum loses **28%** of wealth while risk-managed momentum ends the decade **+88%**.
- **Turnover (1951:03–2010:12, stock-level): 74% per month for WML vs 75% for WML\*** — i.e. volatility scaling adds
  only ~1 pp of monthly turnover.
- **Round-trip cutoff cost** that renders profits insignificant at 5%: **1.27 pp for WML vs 1.75 pp for WML\*** —
  *"the transaction costs that would remove the significance of the profits of risk-managed momentum are 38% higher
  than for conventional momentum."*

**Table 2 — AR(1) of one-month realized variances** (`RV_t = alpha + rho*RV_{t-1} + eps`):

| Sample | Portfolio | alpha (t) | rho (t) | R² | **OOS R²** | avg RV (sigma) | sd of RV |
|---|---|---|---|---|---|---|---|
| 1927:03–2011:12 | RMRF | 0.0010 (6.86) | 0.60 (23.92) | 36.03 | **38.81** | 14.34 | 9.97 |
| 1927:03–2011:12 | **WML** | 0.0012 (5.21) | **0.70 (31.31)** | 49.10 | **57.82** | **17.29** | **13.64** |
| 1963:07–2011:12 | RMRF | 0.0009 (5.65) | 0.58 (17.10) | 33.55 | 25.46 | 13.76 | 8.48 |
| 1963:07–2011:12 | SMB | 0.0004 (8.01) | 0.33 (8.32) | 10.68 | −8.41 | 7.36 | 3.87 |
| 1963:07–2011:12 | HML | 0.0001 (4.88) | 0.73 (25.84) | 53.55 | 53.37 | 6.68 | 4.29 |
| 1963:07–2011:12 | WML | 0.0009 (3.00) | 0.77 (29.29) | 59.71 | 55.26 | 16.40 | 13.77 |

OOS R² definition (Eq. 4): `R²_OOS = 1 − SUM_{t=S}^{T−1}(alpha_hat_t + rho_hat_t*RV_{i,t} − RV_{i,t+1})² / SUM_{t=S}^{T−1}(mean(RV_{i,t}) − RV_{i,t+1})²`,
using an initial 240-month training sample and an expanding window, with all parameters estimated only from
information up to time t.

Monthly realized volatility of momentum *"varies substantially over time, from a minimum of 3.04% (annualized) to a
maximum of 127.87%."*

**Unconditional factor comparison (Table 1, same sample):** RMRF mean 7.33, sd 18.96, kurtosis 7.35, skew 0.17,
SR 0.39; SMB mean 2.99, sd 11.52, kurtosis 21.99, skew 2.17, SR 0.26; HML mean 4.50, sd 12.38, kurtosis 15.63,
skew 1.84, SR 0.36.

## 5. Caps, floors, average weight, range

**No cap or floor is imposed.** The paper states: *"As WML is a zero-investment and self-financing strategy, we can
scale it without constraints."* There is no mention of truncation, winsorization or bounding anywhere in the text
(I searched `cap`, `truncat`, `constrain`, `limit`, `maximum weight`).

The **realized** distribution of the weight over 1927:03–2011:12 (Fig. 4):

- **Range: 0.13 to 2.00**
- **Average: 0.90** — *"slightly less than full exposure to momentum"*
- Lows clustered in the **early 1930s, 2000–02, and 2008–09** (i.e. the high-volatility/crash regimes).
- Because WML is long-short, a weight of 2.00 means leverage of 2 on *each* leg (4× gross exposure) — flagged in a
  marginal annotation on the manuscript.

Note the weight is **not** normalized to average 1. Averages of 0.90 arise naturally because the 12% target is a
daily-frequency annualization and the weight is `E[sigma_target/sigma_hat]` (a harmonic-mean-type object), which
exceeds `sigma_target/E[sigma_hat]`.

## 6. Stated theoretical/economic justification — and *which* crash component it removes

The paper's argument has three legs.

**(a) Momentum risk is highly variable and *predictable*.** OOS R² of the AR(1) on monthly realized variance is
**57.82%** for momentum vs **38.81%** for the market — *"19.01 percentage points higher than a similar autoregression
for the variance of the market portfolio, which is already famously predictable."* Momentum is both the most volatile
(avg RV 17.29 vs 14.34) and the most persistent (rho 0.70 vs 0.60) of the factors.

**(b) The predictable part is the STRATEGY-SPECIFIC risk, not the market-beta risk.** Market-model decomposition
(Eq. 7, p.14):

```
RV_{wml,t}  =  beta_t^2 * RV_{rmrf,t}  +  sigma^2_{eps,t}
```
with realized variances and betas estimated from **six months of daily returns**.
- *"On average, the market component `beta_t^2 * RV_rmrf,t` accounts for only **23% of the total risk of momentum**.
  Almost 80% of the momentum risk is specific to the strategy."*

**Table 4 — AR(1) of each component (six-month, non-overlapping periods; OOS after a 20-year initial window):**

| Variable | alpha (t) | rho (t) | R² | **OOS R²** |
|---|---|---|---|---|
| `sigma^2_wml` | 0.0012 (2.59) | 0.70 (12.58) | 48.67 | **43.82** |
| `sigma^2_rmrf` | 0.0012 (4.29) | 0.50 (7.37) | 24.53 | **6.70** |
| `beta^2` | 0.3544 (6.05) | 0.21 (2.83) | 4.59 | **5.33** |
| `beta^2 * sigma^2_rmrf` (market component) | 0.0007 (2.73) | 0.47 (6.80) | 21.67 | **20.87** |
| `sigma^2_eps` (specific component) | 0.0007 (2.69) | 0.72 (13.51) | 52.21 | **47.06** |

The conclusion is explicit: *"The most predictable component of momentum variance is the specific risk with an OOS
R-square of 47.06%, more than double the predictability of the part due to the market. Hedging the market risk alone,
as in Daniel and Moskowitz (2012), fails because most of the risk is left out."* And `beta_t^2` is *"the least
predictable component of momentum risk"* (OOS R² only 5.33%).

**So: what does scaling remove?** It removes the **strategy-specific (residual/idiosyncratic) volatility component** —
the ~77% of momentum risk that is not market beta and that is *not* reachable by beta-hedging. It does **not**
primarily remove the time-varying market exposure.

**(c) There is also a negative conditional risk-return relation for momentum.** Sorting months into quintiles by
realized variance over the previous six months and looking at the following 12 months: *"In the years after a calm
period, the Sharpe ratio is 1.72 on average. By contrast, after a turbulent period the Sharpe ratio is only 0.28 on
average."* For the **market**, by contrast, *"no obvious trade-off exists between risk and return."* So for momentum
both the numerator and the denominator of the Sharpe ratio move favorably with lagged volatility.

**Robustness the paper reports:**
- **Subsamples (Table 6):** first half kurtosis 23.57 → 4.53; no-crash sample kurtosis 7.15 → 2.47; post-war
  (1945:01–2005:12) skewness −0.91 → −0.17; first half skewness −3.05 → −0.72. Sharpe increase between **0.29**
  (post-war) and **0.47** (second half). *"Even in the samples without a major crash, risk-managed momentum shows a
  very high information ratio of at least 0.60."*
- **International (Table 5, 1980:07–2011:10, France/Germany/Japan/UK):** Sharpe ratio rises in all four markets;
  Japan **0.08 → 0.24** (triples, though still modest); UK gain of **0.68**; information ratio ranges from **0.34**
  (Japan) to **1.19** (UK). Risk-managed momentum has *positive* skew in France, Germany and the UK. In the other
  17 countries with less reliable data, *"it does improve the Sharpe ratio in all of those countries."*
- **Unreported but stated:** *"forecasting the risk of momentum with the realized variance of the market in the
  previous six months produces very similar results"*; and realized volatility of momentum has forecasting content
  *"much greater and more robust than the bear-market indicator."*

---

# PAPER 2 — Cederburg, O'Doherty, Wang & Yan (2020), "On the performance of volatility-managed portfolios"

## 0. Their construction (Eqs. 3–4) — note it differs from Barroso & Santa-Clara

```
f_{sigma,t} = ( c* / sigma_hat^2_{t-1} ) * f_t                                (Eq. 3)

sigma_hat^2_t = ( 22 / J_t ) * SUM_{j=1}^{J_t} ( f^j_t )^2                    (Eq. 4)
```

- `f_t` = monthly excess return of the original factor/portfolio.
- `sigma_hat^2_{t-1}` = realized variance of **daily** returns during **the single month preceding** the formation
  date (**one month**, not six).
- `c*` = constant chosen so that `f_t` and `f_{sigma,t}` have the **same unconditional volatility over the full
  sample** — *"The scaling parameter is not known to an investor in real time, but we note that some performance
  measures (e.g., Sharpe ratios and appraisal ratios) are invariant to the choice of c*."*
- `c*/sigma_hat^2_{t-1}` is the required **leverage**.
- Daily returns are **not demeaned** — it is a sum of squares, scaled by `22/J_t` (22 trading days/month).

**Footnote 7 is the paper's own map of how prior studies differ from Eqs. (3)–(4)** — see §11.

**Table 1 Panel D — percentiles of the implied leverage `c*/sigma_hat^2_{t-1}`** (MKT, SMB, HML, MOM, RMW, CMA, ROE, IA, BAB):

| | MKT | SMB | HML | MOM | RMW | CMA | ROE | IA | BAB |
|---|---|---|---|---|---|---|---|---|---|
| P01 | 0.04 | 0.03 | 0.04 | 0.04 | 0.04 | 0.06 | 0.06 | 0.06 | 0.04 |
| P50 | 0.96 | 0.81 | 1.02 | 1.01 | 1.11 | 0.97 | 1.08 | 0.96 | 1.00 |
| P99 | 6.47 | 5.07 | 5.89 | **8.64** | 5.02 | 4.56 | 4.73 | 4.45 | 5.09 |
| Corr. with original factor | 0.63 | 0.63 | 0.57 | **0.48** | 0.59 | 0.68 | 0.68 | 0.70 | 0.62 |

*"Although the median investment position for each of the dynamic portfolios is around one, the 99th percentile of
required leverage exceeds 400% in each case and reaches as high as 864% for the momentum strategy."*

## 7. The criticism — stated precisely

**The paper is a three-step argument, not a transaction-cost argument.**

**(a) Direct comparisons: no systematic improvement.** Table 2, all 103 strategies: **53 positive / 50 negative**
Sharpe-ratio differences; only **8 significantly positive** vs **4 significantly negative**.
A bootstrap (100,000 samples) under the null of *persistent volatility but no risk-return relation* expects
**66 of 103** positive; the observed **53** is well below that, **two-tailed p = 0.01** → the null is rejected, and
the rejection implies **risk-return relations tend to be POSITIVE** across the 103 strategies (a positive
risk-return trade-off degrades volatility management). Under the null of *equal* expected Sharpe ratios, **p = 0.84**
→ cannot reject equal performance.

**(b) In-sample spanning alphas are genuinely there.** `f_{sigma,t} = alpha + beta*f_t + eps_t` (Eq. 5). Broad sample:
**77 of 103 positive, 23 significant** (vs 3 significantly negative); with FF3 controls, **70 positive, 21 significant**.
The standout is momentum: **alpha = 12.39% per year, t = 7.31, appraisal ratio 0.86.**

**(c) But positive alpha is a LOWER bar than a Sharpe improvement — and the implied strategy is not implementable.**
This is the paper's sharpest technical point:
- Because `f_{sigma,t}` and `f_t` have identical full-sample volatility by construction, a higher Sharpe ratio only
  requires `mean(f_sigma) > mean(f)`.
- A **positive spanning alpha**, however, only requires `mean(f_sigma) > rho_hat * mean(f)`, where `rho_hat` is the
  unconditional correlation between scaled and unscaled — and `rho_hat = beta_hat`.
- *"As shown in Table 1, these correlations range from 0.48 to 0.70 for the equity strategies examined by Moreira and
  Muir (2017). Volatility scaling could, therefore, lead to a **30% or larger drop in Sharpe ratio** and at the same
  time produce a positive spanning regression intercept."*
- Achieving the in-sample utility gain requires **ex post optimal** weights that are unknown in real time. Eq. (10):
  `x*_sigma = alpha_hat / ( gamma * sigma_hat^2_f * (1 − rho_hat^2) )`; Eq. (11): `y*_t = x*_sigma*(c*/sigma_hat^2_{t-1}) + x*`.
  *"An investor could only achieve the utility gains by combining the scaled and unscaled versions of a particular
  factor using weights that are unknown prior to observing the full sample of factor returns. As such, these types of
  strategies are not implementable in real time."*
- Worked example (market): in-sample `alpha = 4.63%/yr`, appraisal ratio 0.32 → ex post optimal mix is **72% scaled
  MKT + 28% original MKT**, giving SR 0.53 vs 0.42 and, at gamma = 5, CER 2.79% vs 1.76%. *"An investor with
  gamma = 5 has a CER of 2.79% for the ex post optimal combination strategy versus only 1.76% for the original factor."*
  The real-time version gets **0.42 vs 0.46 SR and 1.56% vs 1.75% CER — i.e. it loses.**

**On transaction costs specifically: the paper says nothing.** No cost model, no break-even cost, no turnover
analysis. The failure is attributed to (i) the non-implementability of ex-post optimal weights and (ii) structural
instability in the spanning regression parameters.

## 8. Out-of-sample testing design

**Definition of in-sample vs out-of-sample:**
- `T` monthly excess-return observations for a given factor/anomaly portfolio.
- **In-sample** = the full sample, used for the spanning regressions (Eq. 5), the ex post optimal weights
  (Eqs. 7–11) and the reported "combination strategy (ex post optimal)" benchmarks.
- **Out-of-sample** = months **K+1 … T**, where the first **K = 120 months (10 years)** form the initial training
  period and estimation uses an **expanding window**. Monthly rebalancing.

**Real-time recursion** (start of each month t in the OOS period):
1. Compute the real-time scaling parameter `c_t` = the constant that makes the original and volatility-managed
   factors have **the same variance over the training period preceding month t**.
2. Estimate `mu_hat_t` and `Sigma_hat_t` from the training period.
3. Portfolio weights: `[x_{sigma,t} ; x_t] = (1/gamma) * Sigma_hat_t^{-1} * mu_hat_t`  (Eq. 15).
4. Position in the original factor: `y_t = x_{sigma,t} * ( c_t / sigma_hat^2_{t-1} ) + x_t`  (Eq. 16).
5. Portfolio excess return for month t = `y_t * f_t`. ⇒ time series of `T − K` returns.

**Parameters / constraints:** risk aversion **gamma = 5**; leverage constraint **|y_t| ≤ 5** in the base case.
Robustness: **L ≤ 1.0, L ≤ 1.5, and L ≤ ∞ (unconstrained)**. Robustness on K: **120, 240, 360**; also a
**rolling-window** instead of expanding.

**Test statistics:**
- Sharpe difference: `Delta_SR = mu_hat_i/sigma_hat_i − mu_hat_j/sigma_hat_j` (Eq. 17), tested with
  **Jobson & Korkie (1981)** incorporating the **Memmel (2003)** correction.
- CER difference: `Delta_CER = ( mu_hat_i − (gamma/2)*sigma_hat^2_i ) − ( mu_hat_j − (gamma/2)*sigma_hat^2_j )` (Eq. 18),
  tested with the **DeMiguel, Garlappi & Uppal (2009b)** statistic `z_DGU` (asymptotically standard normal).

**Data:** 9 factors + 94 anomaly portfolios = **103 equity strategies**. Factors: MKT/SMB/HML from FF3, MOM,
RMW/CMA from FF5, ROE/IA from Hou-Xue-Zhang q-factor, BAB from Frazzini-Pedersen. Anomalies: value-weighted decile
long-short hedge portfolios from 94 continuous characteristics in Hou et al. (2015) / McLean & Pontiff (2016),
constructed from CRSP/Compustat/IBES, NYSE+Amex+Nasdaq common stocks **July 1926 – December 2016**, excluding
financials (SIC 6000–6999), market cap below the 1st NYSE decile, and price < $5. Category counts: accruals 10,
intangibles 10, investment 11, market 1, momentum 9, profitability 22, trading 21, value 19. Factor sample starts:
MKT/SMB/HML Aug 1926; MOM Jan 1927; RMW/CMA Aug 1963; ROE/IA Feb 1967; BAB Feb 1931; all end Dec 2016.

## 9. Which factors/portfolios it works for — and yes, momentum is the exception

**Direct comparisons across the 9 factors** (Table 1, Sharpe difference [p-value]):

| | MKT | SMB | HML | MOM | RMW | CMA | ROE | IA | BAB |
|---|---|---|---|---|---|---|---|---|---|
| Original mean (%/yr) | 7.80 | 2.57 | 4.84 | 7.94 | 2.92 | 3.72 | 6.52 | 4.99 | 8.23 |
| Original sd (%/yr) | 18.61 | 11.12 | 12.14 | 16.39 | 7.71 | 6.97 | 8.83 | 6.48 | 10.71 |
| Original SR | 0.42 | 0.23 | 0.40 | **0.48** | 0.38 | 0.53 | 0.74 | 0.77 | 0.77 |
| Managed mean (%/yr) | 9.55 | 0.86 | 4.64 | **16.17** | 3.94 | 2.79 | 9.39 | 4.69 | 10.81 |
| Managed SR | 0.51 | 0.08 | 0.38 | **0.99** | 0.51 | 0.40 | **1.06** | 0.72 | **1.01** |
| **ΔSR [p]** | +0.09 [0.30] | −0.15 [0.09] | −0.02 [0.86] | **+0.50 [0.00]** | +0.13 [0.29] | −0.13 [0.23] | **+0.32 [0.01]** | −0.05 [0.68] | **+0.24 [0.01]** |

(Note: sd is identical across original and managed **by construction** — `c*` forces it.)

**In-sample spanning alphas, per factor (Table 3, Panel A.1; %/yr, t-stat):**
MKT **4.63 (3.08)**; SMB **−0.76 (−0.87)**; HML 1.87 (1.88); MOM **12.39 (7.31)**; RMW 2.23 (2.57); CMA 0.26 (0.39);
ROE **4.97 (5.10)**; IA 1.18 (1.83); BAB **5.74 (5.97)**. Betas (=correlations) 0.63, 0.63, 0.57, 0.48, 0.59, 0.68,
0.68, 0.70, 0.62. Appraisal ratios 0.32, −0.09, 0.19, **0.86**, 0.36, 0.05, 0.77, 0.26, 0.68.

**Out-of-sample real-time combination (Table 5, Panel A; gamma = 5, |y| ≤ 5, K = 120):**

| | MKT | SMB | HML | MOM | RMW | CMA | ROE | IA | BAB |
|---|---|---|---|---|---|---|---|---|---|
| SR: combination (real time) | 0.42 | 0.14 | 0.38 | **0.92** | 0.44 | 0.52 | **1.13** | 0.70 | **1.09** |
| SR: original (real time) | 0.46 | 0.19 | 0.43 | 0.49 | 0.31 | 0.56 | 0.78 | 0.68 | 0.79 |
| **ΔSR [p]** | −0.04 [0.64] | −0.06 [0.37] | −0.06 [0.41] | **+0.44 [0.00]** | +0.13 [0.53] | −0.03 [0.20] | **+0.36 [0.00]** | +0.02 [0.74] | **+0.30 [0.00]** |
| CER: combination (%) | 1.56 | 0.00 | 1.41 | **8.47** | 1.96 | 2.74 | **12.25** | 4.19 | **10.88** |
| CER: original (%) | 1.75 | 0.38 | 1.61 | 2.29 | 0.91 | 3.09 | 5.44 | 3.68 | 6.23 |
| **ΔCER [p]** | −0.19 [0.83] | −0.37 [0.27] | −0.20 [0.73] | **+6.18 [0.00]** | +1.04 [0.57] | −0.35 [0.21] | **+6.81 [0.00]** | +0.51 [0.60] | **+4.65 [0.00]** |

*"The combination portfolios outperform the original factors in five of the nine cases"* — MOM, RMW, ROE, IA, BAB
positive; MKT, SMB, HML, CMA negative. **MOM, ROE, BAB are significant at the 1% level.**

**Broad sample (Tables 2 & 6):** 103 strategies — ΔSR **53 [8] positive / 50 [4] negative**; real-time combination
ΔSR **45 [8] / 58 [2]**, ΔCER **31 [7] / 72 [7]**; versus ex-post optimal, **1 [0] / 102 [39]** (SR) and
**0 [0] / 103 [41]** (CER).

**So: does it work for momentum but not others? YES — momentum is by far the strongest and cleanest case:**
- **Table 2 (direct comparisons, by strategy type): momentum is the ONLY category where all strategies improve —
  9 positive, 0 negative, 5 of them significant.**
- **Table 4 (spanning regressions, by strategy type): momentum is the only category with 9/9 positive AND 9/9
  significant — both univariate and with FF3 controls.**
- **Table 6 (real-time combination, by strategy type): momentum ΔSR 8 [4] / 1 [0]; ΔCER 8 [5] / 1 [0].**
- The paper says so plainly: *"Panel C reveals that the majority of the significantly positive Sharpe ratio
  differences are attributable to the nine momentum strategies. Volatility management improves performance for every
  momentum strategy, and five of the nine performance differences are statistically significant at the 5% level. The
  findings are consistent with the impressive performance of volatility-managed momentum portfolios demonstrated by
  Barroso and Santa-Clara (2015) and Daniel and Moskowitz (2016)."*

**Outside momentum, only three strategies show significant outperformance:** **ROE**, **BAB**, and **Loughran &
Wellman's (2012) enterprise multiple**.

**Four portfolios are significantly WORSE** with volatility management: growth in book equity (Lockwood &
Prombutr 2010), change in sales less change in inventory (Abarbanell & Bushee 1998), 1/share price (Miller &
Scholes 1982), and long-term reversal (De Bondt & Thaler 1985).

**Adding FF3 to the investment opportunity set makes things worse** (Table 5 Panel B / Table 6 Panel B): MOM ΔSR
falls from +0.44 to **+0.20** [0.00] and ΔCER from +6.18 to **+4.13** [0.00]; ROE ΔCER collapses to +1.38 (p = 0.25);
BAB ΔCER +4.04 [0.00]; SMB/HML/MKT all negative. Broad sample: only **32 of 103** positive on either measure;
`*"volatility-managed portfolios are even less useful to investors with access to market, size, and value strategies."*`

**Momentum robustness:** *"Barroso and Santa-Clara (2015) notably find that managing the risk of momentum is robust
to using alternative windows (i.e., one, three, or six months) to estimate realized volatility. We complement this
result by showing that the performance of volatility-managed momentum is also robust to several alternative
definitions of the momentum strategy itself."*

## 10. Why the in-sample alphas do not survive out-of-sample

The paper explicitly weighs three candidates: **(i) estimation risk, (ii) low power, (iii) structural instability** —
and rejects the first two.

**(i) Estimation risk — rejected.** Five reasons given: the design already mitigates it (leverage constraint,
risk-free asset in the opportunity set per Kirby & Ostdiek 2012, expanding-window estimation); DeMiguel et al. (2009b)
note estimation risk is less problematic with few test assets (here only 2); *"both the combination strategy and the
benchmark suffer from estimation risk, and it is not obvious why one of the two would be more adversely impacted"*;
longer training samples (K = 240, 360) do not improve results; and Panel B of Table 8 shows anomaly alphas **do**
translate into real-time gains far more often.

**(ii) Low power — rejected.** *"If volatility management is systematically beneficial to investors, then we should
see a majority of performance differences that are positive in Tables 6 and 7. Low power might be an explanation for
why an individual result is statistically insignificant, but it does not account for why most of the performance
differences have the wrong sign."*

**(iii) Structural instability in the spanning regressions — ACCEPTED** (this is the paper's central claimed driver).
Bai & Perron (1998, 2003) tests allowing an unknown number of breaks:

**Table 9:**

| | Total | N_b = 0 | 1 | 2 | 3 | ≥4 | mean N_b |
|---|---|---|---|---|---|---|---|
| **Spanning regressions (VM portfolios)** | 103 | **0** | 10 | 52 | 34 | 7 | **2.37** |
| Spanning regressions + FF3 controls | 103 | 1 | 8 | 53 | 35 | 6 | **2.37** |
| CAPM anomaly regressions (comparison) | 102 | 15 | 38 | 39 | 9 | 1 | **1.44** |
| FF3 anomaly regressions (comparison) | 100 | 10 | 25 | 36 | 21 | 8 | **1.92** |

*"For the univariate spanning regressions, none of the tests suggests zero breaks, and 41 out of 103 tests identify
three or more breaks… In contrast, structural breaks are less common in the standard time-series anomaly regressions…
In the volatility-managed portfolios setting, the prevalence of breaks often works to the detriment of real-time
investors who rely on past data in portfolio construction."*

**Additional specific mechanism given for the MARKET factor (Internet Appendix result, summarized in the paper):**
*"we show that the strong in-sample performance for volatility-scaled MKT is concentrated in the period surrounding
the Great Depression, which occurs early in the sample. Out-of-sample investors adopting the combination strategy
tend to favor the volatility-managed version of the market factor based on its strong early sample performance. These
investors experience unfavorable investment results, however, because the scaled MKT [subsequently underperforms]."*
So for MKT the alpha **is** driven by a specific early subsample — the paper's own words for the market factor.

**What the paper explicitly does NOT blame:** transaction costs (never analyzed); a failure of volatility
persistence (they *rely* on volatility clustering — it is the basis of their null hypothesis); negative in-sample
utility gains (in-sample gains are positive by construction from positive alphas — it is the **out-of-sample** CERs
that turn negative).

**The economic condition they state for volatility management to work:**
> *"Volatility management is likely to be successful if volatility is persistent and the risk-return relation is
> flat. In this scenario, a portfolio's conditional Sharpe ratio is negatively associated with its lagged volatility,
> and investors can capitalize on these dynamics… If lagged volatility is negatively related to average return for a
> given strategy, volatility management becomes even more attractive. A positive risk-return trade-off, in contrast,
> makes volatility management less effective."*

Their data reject the "flat/no relation" null (p = 0.01; 53 observed vs 66 expected under the null), implying the
risk-return relation is **positive** on average across the 103 strategies — which mechanically undermines volatility
management. Note this is the *opposite* of what Barroso & Santa-Clara document for momentum specifically (they find a
*negative* conditional risk-return relation for momentum: Sharpe 1.72 after calm vs 0.28 after turbulent).

## 11. "Volatility-managed" (scale by market vol) vs "constant-volatility" (scale by the strategy's own vol)

**The paper does not run this horse race, and does not claim one is more robust.** I checked: there is no
"constant volatility" discussion and no market-vol-vs-own-vol comparison anywhere in the text. What the paper does:

- **Its own construction (Eqs. 3–4) is the "constant-volatility" / own-vol type**: scale by the strategy's **own**
  lagged realized variance, with `c*` set to **match the original factor's full-sample volatility** — *not* to hit an
  absolute target.
- **Footnote 7 is the only taxonomy in the paper**, and it lists five dimensions on which studies differ from
  Eqs. (3)–(4):
  1. *"scaling by realized standard deviation rather than by realized variance (e.g., Barroso and Santa-Clara, 2015;
     Barroso and Maio, 2018)"*
  2. *"using a parametric model to estimate volatility (e.g., Daniel and Moskowitz, 2016; Moreira and Muir, 2017)"*
  3. *"using longer intervals to estimate lagged realized volatility (e.g., Barroso and Santa-Clara, 2015;
     Daniel and Moskowitz, 2016)"*
  4. *"scaling by market volatility rather than by factor-specific volatility (e.g., Eisdorfer and Misirli, 2020)"*
  5. *"selecting c* to achieve a target level of strategy risk rather than to match the standard deviation of the
     original factor."*
- **The only empirical hint**, and it is indirect: Cederburg et al. use a **one-month** realized-variance window and
  still find volatility-managed momentum works out-of-sample; Barroso & Santa-Clara use **six months**. Cederburg
  also cite BSC's own evidence that 1-, 3- and 6-month windows all work. So on the *window-length* dimension, both
  short and long windows succeed for momentum. On the *own-vol vs market-vol* dimension, **neither paper provides a
  direct comparison** — BSC report (unreported results) that using *market* realized variance over the previous six
  months to forecast *momentum's* risk gives *"very similar results"*, which is suggestive but is about forecasting
  momentum risk, not about a horse race of trading rules.

**Honest bottom line: neither paper establishes that one scaling choice is more robust than the other.**
The distinction is acknowledged only as descriptive taxonomy. Anyone claiming Cederburg et al. showed
"constant-volatility beats volatility-managed" (or vice versa) is over-reading the paper.

## 12. Recommendation for practitioners

Verbatim from the conclusion:
- *"Direct investments in volatility-scaled strategies are straightforward to implement in real time, and studies
  following this approach offer compelling empirical evidence that these dynamic portfolios are superior to their
  static versions. The evidence is isolated to a handful of strategies (e.g., the market, momentum, and
  betting-against-beta factors), however, making it difficult to draw broad conclusions."*
- *"Across a broad sample of 103 equity portfolios, volatility management degrades and improves performance at about
  the same frequency. From a practical perspective, the results suggest that direct investments in volatility-managed
  portfolios are not a panacea of improved performance."*
- *"Combination strategies that incorporate volatility management, in contrast, exhibit systematically strong
  in-sample performance… We also demonstrate, however, that structural instability in the spanning regression
  parameters limits the appeal of this approach to investors conditioning their portfolios on real-time information.
  The Sharpe ratios and CERs for the out-of-sample combination portfolios are dramatically less impressive than those
  earned by their in-sample versions. Moreover, the real-time combination strategies routinely underperform simpler
  strategies constrained to invest in the original, unscaled portfolios."*
- *"Overall, our findings suggest a more tempered interpretation of the practical value of volatility-managed
  portfolios relative to prior literature."*
- *"our findings in this section indicate that attempting to use volatility management in real time tends to degrade
  performance relative to constraining the investment universe to the original factors and anomaly portfolios."*
- Comparative benchmark: traditional anomaly portfolios fare much better in real time — *"Adding an anomaly portfolio
  to the CAPM market factor in real time, for example, leads to a Sharpe ratio improvement in 75 out of 102 cases and
  a CER improvement in 68 out of 102 cases"* (vs 45/103 and 31/103 for volatility management).

**Robustness evidence behind the recommendation (Table 7, N = 103, ΔSR / ΔCER as +[sig] / −[sig], with binomial p):**

| Design | ΔSR | p | ΔCER | p |
|---|---|---|---|---|
| Base case (expanding, K=120, gamma=5, L≤5) | 45 [8] / 58 [2] | 0.237 | 31 [7] / 72 [7] | **0.000** |
| Rolling window | 49 [2] / 54 [1] | 0.694 | 17 [1] / 86 [19] | **0.000** |
| gamma = 2 | 48 [9] / 55 [2] | 0.555 | 35 [8] / 68 [7] | **0.001** |
| gamma = 10 | 45 [8] / 58 [2] | 0.237 | 31 [7] / 72 [7] | **0.000** |
| K = 240 | 45 [9] / 58 [10] | 0.237 | 36 [8] / 67 [11] | **0.003** |
| K = 360 | 40 [9] / 63 [6] | 0.030 | 31 [8] / 72 [8] | **0.000** |
| L ≤ 1.0 | 49 [10] / 54 [2] | 0.694 | 38 [4] / 65 [5] | **0.010** |
| L ≤ 1.5 | 47 [10] / 56 [3] | 0.431 | 38 [7] / 65 [7] | **0.010** |
| L ≤ ∞ | 45 [8] / 58 [2] | 0.237 | 31 [7] / 72 [7] | **0.000** |

*"All nine of the binomial p-values for the CER differences produce statistical rejections of the null hypothesis of
equal performance at the 5% significance level."* And with FF3 controls, **all 18 binomial p-values reject at 1%.**

**Table 8 — the key asymmetry (volatility management vs ordinary anomalies):**

| | Total | alpha + [sig] / − [sig] | ΔSR + [sig] / − [sig] | ΔCER + [sig] / − [sig] |
|---|---|---|---|---|
| **Panel A: spanning regressions (VM)** | 103 | 77 [23] / 26 [3] | 45 [8] / 58 [2] (p=0.237) | 31 [7] / 72 [7] (p=0.000) |
| Spanning + FF3 controls | 103 | 70 [21] / 33 [3] | 32 [2] / 71 [13] (p=0.000) | 32 [3] / 71 [10] (p=0.000) |
| **Panel B: anomaly regressions (CAPM)** | 102 | 93 [73] / 9 [3] | 75 [19] / 27 [1] (p=0.000) | 68 [18] / 34 [1] (p=0.001) |
| Anomaly regressions (FF3) | 100 | 81 [60] / 19 [5] | 66 [18] / 34 [1] (p=0.002) | 55 [19] / 45 [1] (p=0.368) |

---

# IMPLEMENTABLE TAKEAWAYS

Concrete, codeable rules, each with the caveats the papers impose. **Where the two papers conflict, I say so.**

### A. Barroso & Santa-Clara constant-volatility momentum (the "cMOM" rule)

**A1. Signal.** Standard cross-sectional WML: rank NYSE/Amex/Nasdaq stocks on cumulative returns from **month
t−12 to t−2** (skip month t−1), decile breakpoints from NYSE, **value-weight** within deciles, long top decile /
short bottom decile. Rebalance monthly.

**A2. Volatility forecast at the end of each month t−1** — use **exactly 126 daily WML returns** ending on the last
trading day of month t−1, squared, **not demeaned**:

```
sigma_hat_t = sqrt( (21/126) * SUM_{i=1}^{126} r_i^2 )      # monthly units
```

Do **not** use `sd(daily returns) * sqrt(21)` — that demeans and is not what the paper does. Do **not** use the
market's volatility; use **WML's own** returns.

**A3. Weight for month t:**

```
w_t = sigma_target / sigma_hat_t ,   sigma_target = 0.12 / sqrt(12)   # = 0.034641 per month
# equivalently, annualised: w_t = 0.12 / sqrt(2 * SUM_{i=1}^{126} r_i^2)
```

Apply to the WML return: `r_scaled_t = w_t * r_WML_t`. The strategy stays self-financing; `w_t` is the dollar
amount in **each** leg (so gross exposure is `2*w_t`).

**A4. Timing.** The window must end on the last trading session of month t−1; the weight applies to month t.
No look-ahead. (BSC footnote 7.)

**A5. Do NOT cap or floor the weight** if you are replicating the paper exactly — *"As WML is a zero-investment and
self-financing strategy, we can scale it without constraints."* The observed range over 1927:03–2011:12 was
**0.13 to 2.00**, mean **0.90**. **Caveat (mine, not the paper's):** a realized max of 2.00 with no stated cap means
any live implementation should decide its own leverage policy; BSC's own range came from 85 years including the
Depression, and Cederburg et al. document that the analogous 1-month rule reaches the **99th percentile at 8.64×**
for momentum and imposes `|y| ≤ 5` purely *"to guard against our results being driven by extreme outliers."*

**A6. Robustness switches that the paper validated.** Windows of **21, 63, or 126** sessions all work; EWMA with
half-lives of 1, 3, or 6 months also works. Using the **market's** realized variance over the previous six months to
forecast momentum risk gives *"very similar results."* None of these are the headline result — the headline is the
126-day own-return rule.

**A7. Expected magnitudes (1927:03–2011:12, US).** SR 0.53 → 0.97; excess kurtosis 18.24 → 2.68; skew −2.47 → −0.42;
worst month −78.96% → −28.40%; max drawdown −96.69% → −45.20%; mean return +2.04 pp/yr; volatility −10.58 pp/yr;
information ratio 0.78. Turnover barely changes (74% → 75%/month) and the break-even round-trip cost rises only from
1.27% to 1.75%.

**A8. When it should work, mechanically.** Scaling helps when (i) volatility is persistent — for momentum the AR(1)
of monthly realized variance is **0.70** with **OOS R² 57.82%** — and (ii) the conditional Sharpe ratio is
**negatively** related to lagged volatility. BSC document exactly that for momentum: **1.72 after calm months vs
0.28 after turbulent months**. The predictable part is the **strategy-specific** (residual) variance
(OOS R² **47.06%**), not the market-beta component (**20.87%**, with `beta^2` alone at **5.33%**); the market
component is only **23%** of total momentum risk. **Do not** implement this as a beta hedge.

### B. Cederburg et al. — the constraints you must respect

**B1. Do not extrapolate the momentum result to other factors.** Across 103 strategies the hit rate is a coin flip:
**53 positive / 50 negative**, and **77/103 positive spanning alphas but only 45/103 positive real-time Sharpe
differences.** The reliable exceptions are **MOM (all 9/9 improve), ROE, BAB**, plus the enterprise multiple.
Four portfolios get significantly *worse*. **Caveat: this is the single most important practical finding — the
in-sample alpha is a much weaker signal than it looks.**

**B2. Never treat a positive spanning-regression alpha as evidence of a better portfolio.** Because `rho_hat`
between scaled and unscaled is only **0.48–0.70** (0.48–0.80 across the 103), *"volatility scaling could lead to a
30% or larger drop in Sharpe ratio and at the same time produce a positive spanning regression intercept."*
The correct test is a **direct Sharpe comparison with a Jobson–Korkie (Memmel-corrected) test**, plus a real-time CER
comparison — not the regression intercept.

**B3. Do not build combination portfolios with fixed weights on the scaled and unscaled versions.** The weights
`x*_sigma = alpha_hat/(gamma*sigma_hat^2_f*(1−rho_hat^2))` are ex-post optimal and *"not implementable in real
time."* If you do build one, estimate `c_t`, `mu_hat_t`, `Sigma_hat_t` from an **expanding window with an initial
10-year (K = 120) training period**, set `gamma = 5`, cap `|y_t| ≤ 5`, and re-balance monthly — that is exactly the
design that produced **45/103 positive on Sharpe and 31/103 on CER**. **Caveat: BSC never build a combination
portfolio; they invest 100% in the scaled strategy. These are different rules and only the BSC pure-scaled rule is
supported by BSC's evidence.**

**B4. Assume the relationship breaks.** Spanning-regression parameters are unstable: **0 of 103** volatility-managed
spanning regressions has zero structural breaks, mean **2.37** breaks, and **41/103 have ≥3**, versus a mean of
**1.44** for ordinary CAPM anomaly regressions. Use a rolling window if you believe breaks are large — though note
the rolling-window variant made CER results *worse* (17 positive / 86 negative). Refit and monitor.

**B5. Check the sign of the risk-return relation for your strategy before deploying.** Volatility management helps
only when the risk-return relation is **flat or negative**; a positive relation makes it *counterproductive*. BSC
document a negative relation for momentum (1.72 vs 0.28 Sharpe); Cederburg et al. find the cross-section of 103
strategies on average exhibits a **positive** relation (observed 53 positive vs 66 expected under the no-relation
null; p = 0.01). Test this first.

**B6. Do not implement the naive 1-month version for anything other than momentum.** Cederburg's own rule is
`f_{sigma,t} = (c*/sigma_hat^2_{t-1})*f_t` with `sigma_hat^2_t = (22/J_t)*SUM_j (f^j_t)^2` over **month t−1 only**,
and `c*` calibrated to match the original's full-sample volatility. That rule gives a real-time momentum Sharpe of
**0.92 vs 0.49** and CER **8.47% vs 2.29%**, but loses for MKT (−0.04 / −0.19), SMB, HML and CMA, and only 32/103
survive once FF3 is in the opportunity set.

**B7. What neither paper supports.** (i) Any claim that transaction costs explain the failure of volatility-managed
portfolios — **Cederburg et al. never test costs**; BSC find costs are *not* the binding constraint (turnover 74% →
75%, break-even cost 1.27% → 1.75%). (ii) Any claim that Cederburg et al. showed scaling by the strategy's own
volatility is more (or less) robust than scaling by market volatility — **they run no such comparison**; the
distinction appears only as descriptive taxonomy in their footnote 7, alongside SD-vs-variance, short-vs-long
windows, and target-vol-vs-match-vol. (iii) Any claim that Cederburg et al. showed the alpha is driven by "a small
number of extreme observations" **in general** — that specific diagnosis is offered only for the **market factor**,
where the in-sample gain is *"concentrated in the period surrounding the Great Depression."* Their general diagnosis
is **structural instability**, not outliers.

### C. If you must build one thing
For **momentum only**, use **BSC's pure constant-volatility rule (A1–A5)** rather than a scaled/unscaled
combination portfolio. It is the configuration with the strongest and most replicated evidence: BSC's US 1927–2011
(SR 0.53 → 0.97), BSC's international sample (SR improves in all four reported markets — France, Germany, Japan, UK —
and, per the paper, in all 17 of the remaining countries in Chaves' 21-country set), and
Cederburg et al.'s independent out-of-sample test (real-time combination SR 0.92 vs 0.49, CER 8.47% vs 2.29%, p = 0.00,
plus 9/9 momentum strategies improving and 5/9 significant). Everything outside momentum should be treated as
unsupported.
