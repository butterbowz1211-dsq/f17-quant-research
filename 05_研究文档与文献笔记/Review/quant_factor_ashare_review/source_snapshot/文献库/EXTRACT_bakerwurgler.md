<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# Methodology-level extraction — Baker & Wurgler, "Investor Sentiment and the Cross-Section of Stock Returns"

**Source actually read:** `workspace/source_workspace\papers\bw_nber.txt` (4,298 lines), the full extracted text of
**NBER Working Paper No. 10449, April 2004** (Baker & Wurgler, "Investor Sentiment and the Cross-Section of Stock Returns"),
converted from https://www.nber.org/system/files/working_papers/w10449/w10449.pdf

**Published version:** Journal of Finance 61(4), 2006, 1645–1680, DOI 10.1111/j.1540-6261.2006.00885.x. Not readable here (Wiley 403);
a secondary mirror could not be retrieved (search engine rate-limited). **Therefore every number, quote and line number below is from the
NBER WP text on disk, not from the published JF version.** Where the WP is internally inconsistent or silent, that is stated explicitly.

---

## 0. Provenance / version notes (read this before using any number)

1. **The WP has no Appendix.** `grep -i "Appendi"` over the whole file returns **no matches**. All index-construction detail in this paper
   is in Section IV.C "Investor sentiment" (starts line 485) plus the Table 2 / Figure 1 notes.
2. **The WP never mentions varimax, rotation, or any orthogonal/oblique rotation.** `grep -i "varimax|rotation|orthogonal rotation"`
   returns no matches. Do **not** attribute a varimax-rotated index to this paper on the basis of this source.
3. **Internal inconsistency in the macro-control list** (see §2.2): the body text omits *growth in employment*; the Table 2 note and the
   Figure 1 note include it.
4. **Frequency of the orthogonalization regression is never stated** (see §2.2). The proxies are annual and the regression is described in
   annual terms, so annual is the strong implication, but the WP does not say "annual", does not report the number of observations, and does
   not give an equation for this step.
5. **The exact turnover detrending window alignment is never stated** (see §2.1): the only wording is "the five-year moving average" /
   "the past five-year average". Whether the window includes year *t* and whether it is backward-looking or centered is **not** specified.
6. The published JF version is known in the literature to use a *different* (updated) index with different loadings and a different
   variance-explained figure. **I could not verify any published-version number from a readable source, so none is quoted here.**
   Treat the loadings in Eq. (2) below as the **WP-10449 (2004)** values only.

---

## 1. (15) THE SIX SENTIMENT PROXIES

Sample sentence (lines 497–498): *"The sentiment proxies are measured annually from 1962 through 2001."*
So all six are **annual** series; the index is an **annual** object. 40 annual observations for the index years 1962–2001.

### 1.1 Summary table

| # | Symbol (paper's exact usage) | What it measures / how computed | Frequency & source | High value = high or low sentiment? | Loadings sign | Timing **inside** the index (year *t*) | Timing **matched to monthly returns** of calendar year *t* |
|---|---|---|---|---|---|---|---|
| 1 | **CEFD** (`CEFD_t`) | "the average difference between the NAV of closed-end stock fund shares and their market prices"; year-end, **value-weighted** average discount on closed-end **stock** funds | annual, year-end; Neal & Wheatley (1998) 1962–1993; CDA/Wiesenberger 1994–1998; turn-of-the-year WSJ issues 1999–2001 | **High = LOW sentiment** ("the discount is inversely related to sentiment") | **−0.358** | `CEFD_t` (contemporaneous) | `CEFD_{t−1}` |
| 2 | **TURN** (`TURN_{t-1}`) | NYSE share turnover = reported share volume ÷ average shares listed, **detrended**: "the log of the ratio is detrended by its moving average" (5-year). See §2.1 | annual; NYSE Fact Book | **High = HIGH sentiment** (optimistic irrational investors add liquidity) | **+0.402** | `TURN_{t−1}` (lagged 1 yr) | `TURN_{t−2}` |
| 3 | **NIPO** (`NIPO_t`) | "the annual number of initial public offerings" (a **count**, not scaled) | annual; Jay Ritter's website, updating Ibbotson, Sindelar & Ritter (1994) | **High = HIGH sentiment** (IPO market sensitive to sentiment; firm-supply response) | **+0.414** | `NIPO_t` (contemporaneous) | `NIPO_{t−1}` |
| 4 | **RIPO** (`RIPO_{t-1}`) | "the average annual first-day returns of initial public offerings" (average underpricing, %) | annual; Jay Ritter's website (same source as NIPO) | **High = HIGH sentiment** ("high first-day returns on IPOs may also be a measure of investor enthusiasm") | **+0.464** | `RIPO_{t−1}` (lagged 1 yr) | `RIPO_{t−2}` |
| 5 | **S** (`S_t`) | "gross equity issuance divided by gross equity plus gross long-term debt issuance" — "using data from the Federal Reserve Bulletin"; i.e. S = GE / (GE + GLD) | annual; Federal Reserve Bulletin (series from Baker & Wurgler 2000) | **High = HIGH sentiment** (equity-heavy financing composition = market timing on sentiment-driven mispricing) | **+0.371** | `S_t` (contemporaneous) | `S_{t−1}` |
| 6 | **PD-ND** (`P^{D-ND}_{t-1}`) | "the year-end log difference of the average market-to-book ratios of payers and nonpayers" — value-weighted average M/B of dividend payers **minus** value-weighted average M/B of nonpayers, in logs | annual, year-end; Baker & Wurgler (2003) data | **High = LOW sentiment** (established by the negative index loading and by line 882: "As sentiment increases (lower CEFD_{t-1} or PD-ND_{t-2} …)". *My* mechanistic gloss, not the paper's words: a high premium means payers are relatively expensive ⇒ relative demand for the safe category ⇒ low propensity to speculate; the paper only says PD-ND is "a proxy for relative investor demand for dividend payers", line 554) | **−0.431** | `PD-ND_{t−1}` (lagged 1 yr) | `PD-ND_{t−2}` |

Significance/notes: **all six loadings have the sign the paper calls "the expected sign"** — quoted at lines 629–630:
*"First, each individual sentiment proxy enters with the expected sign."*

Equation (2) re-derived from the PDF text: **`SENTIMENT_t = −0.358·CEFD_t + 0.402·TURN_{t−1} + 0.414·NIPO_t + 0.464·RIPO_{t−1} + 0.371·S_t − 0.431·(PD−ND)_{t−1}`** (see §2.6 for the fragment-level verification of this reconstruction).

### 1.2 Per-proxy detail with the paper's own wording and line numbers

**(1) CEFD — closed-end fund discount** (lines 507–514)
> "The closed-end fund discount CEFD is the average difference between the NAV of closed-end stock fund shares and their market prices.
> Prior work suggests that the discount is inversely related to sentiment. … We take the value-weighted average discount on closed-end
> stock funds for 1962 through 1993 from Neal and Wheatley (1998), for 1994 through 1998 from CDA/Wiesenberger, and for 1999 through 2001
> from turn-of-the-year issues of the Wall Street Journal." (lines 507–514)

Role the paper assigns it (lines 678–684) — this is the single most important sentence for the feasibility question:
> "it is worth showing results for the index component CEFD alongside the overall index, however, because CEFD is perhaps our cleanest
> 'general' indicator of investor sentiment – it is not mechanically connected to sentiment for any one segment of the cross-section.
> In contrast, the volume of IPOs and the dividend premium could in principle be driven by a concentrated demand for firms with IPO
> characteristics or dividends, respectively." (lines 678–683)

**(2) TURN — NYSE share turnover, detrended** (lines 515–529)
> "NYSE share turnover is based on the ratio of reported share volume to average shares listed from the NYSE Fact Book. Baker and Stein
> (2002) suggest that turnover, and more generally liquidity, can serve as a sentiment index. In a market with short-sales constraints,
> irrational investors participate and thus add liquidity only when they are optimistic, and hence high liquidity tends to coincide with
> overvaluation." (lines 515–519)

Detrending (lines 524–529) — exact wording:
> "Turnover displays an exponential positive trend over our period, however, and the May 1975 elimination of fixed commissions have a
> visible effect. As a partial solution, the raw turnover ratio TURN is detrended by the five-year moving average (specifically, the log
> of the ratio is detrended by its moving average). This helps the variable to identify sharp changes in turnover, but it will clearly
> still pick up the sharp changes caused by market structure." (lines 524–529)

Table 2 note restatement (lines 2021–2023): *"The second measure (TURN) is detrended log turnover. Turnover is the ratio of reported share
volume to average shares listed from the NYSE Fact Book. We detrend using the past five-year average."*

**(3) NIPO and (4) RIPO — IPO volume and first-day returns** (lines 530–536)
> "The IPO market is often viewed as sensitive to sentiment, and high first-day returns on IPOs may also be a measure of investor
> enthusiasm. Likewise, the low idiosyncratic returns on IPOs are often interpreted in terms of market timing (Stigler (1964),
> Ritter (1991)). We take the number of IPOs NIPO and the average first-day returns RIPO in a given year from Jay Ritter's website,
> which updates the sample in Ibbotson, Sindelar, and Ritter (1994). Of course, there are many non-sentiment reasons for why IPO volume
> and so-called underpricing vary over time, but they do not have direct implications for cross-sectional return predictability."
> (lines 530–536)

**(5) S — equity share in new issues** (lines 537–543 and footnote 7)
> "The share of equity issues in total equity and debt issues is another measure of financing activity that may capture an aspect of
> sentiment. Baker and Wurgler (2000) find that when the equity share is in its bottom (top) historical quartile, the next year's
> equal-weighted market return averages 27% (−8%), and argue that part of this predictability reflects market timing by firms made
> possible by sentiment-driven mispricing. The equity share is defined as gross equity issuance divided by gross equity plus gross
> long-term debt issuance using data from the Federal Reserve Bulletin." (lines 537–543)

Footnote 7 (lines 545–548), which matters for replication:
> "While they both reflect equity issues, the number of IPOs and the equity share have important differences. The equity share includes
> seasoned offerings, is known to predict market returns, and scales by total external finance to isolate the composition of finance from
> the level. On the other hand, the IPO variables may better reflect demand for certain IPO-like regions of the cross-section that theory
> and history suggest are most sensitive to sentiment."

**(6) PD-ND — dividend premium** (lines 553–557)
> "The dividend premium PD-ND is the log difference of the average market-to-book ratios of payers and nonpayers. Baker and Wurgler (2002)
> motivate the dividend premium as a proxy for relative investor demand for dividend payers. Because Fama and French (2001) find that
> payers are larger, more profitable, and have weaker growth opportunities, the dividend premium may also pick up the relative investor
> demand for this correlated bundle of characteristics." (lines 553–557)

Table 2 note version (lines 2025–2026): *"The sixth measure (PD-ND) is the year-end log ratio of the value-weighted average market-to-book
ratios of payers and nonpayers from Baker and Wurgler (2003)."*

### 1.3 Why the timing differs across proxies — the paper's stated logic (lines 558–566)

> "Intuitively, to detect the hypothesized cross-sectional patterns in predictability, we want to condition on turning points in investor
> sentiment. But the sentiment proxies themselves have certain lead-lag relationships, raising the possibility that different variables
> may reflect the same sentiment factor at different times. For instance, Ibbotson and Jaffe (1975), Lowry and Schwert (2002), and
> Benveniste, Wilhelm, Ljungqvist, and Yu (2003) find that IPO volume lags the first-day returns on IPOs. One interpretation of this fact
> is that sentiment is in part behind the high first-day returns, and this attracts additional IPO volume with a lag. More generally,
> proxies that involve firm supply responses (S and NIPO) are likely to lag proxies that are based on investor demand or investor
> behavior (RIPO, PD-ND, TURN, and CEFD)." (lines 558–566)

**Timing verdict (inside the index):** the three *price / investor-behavior* proxies **TURN, RIPO, PD-ND** enter the index **lagged one year**
(`t−1`), while the two *firm-supply* proxies **NIPO, S** and the behavior proxy **CEFD** enter **contemporaneously** (`t`). The paper calls the
CEFD case the exception (lines 630–631):
> "Second, all but one of the proxies enter with the expected timing – with the exception of CEFD, price and investor behavior variables
> lead firm supply variables." (lines 629–631)

Table 2 note (lines 2026–2027): *"Turnover, the average annual first-day return, and the dividend premium are lagged one year relative to
the other three measures."* Figure 1 note (lines 1404–1406): *"In the index, turnover, the average annual first-day return, and the dividend
premium are lagged one year relative to the other three measures."*

**Timing verdict (matching to monthly cross-sectional returns) — one extra year of lag on everything.** The paper conditions returns of
calendar year *t* on the **December of year t−1** value of the index:
> "We report sorts on CEFD in Table 3a and SENTIMENT in Table 3b. As discussed above, we condition returns from calendar year t on the
> December of year t-1 values of CEFD and SENTIMENT." (lines 667–669)

And, decisively, for the individual proxies (lines 882–884):
> "As sentiment increases (lower CEFDt-1 or PD-ND t-2 or higher TURN t-2, NIPO t-1, RIPO t-2, S t-1, or SENTIMENT t-1), expected returns
> tend to decrease on small firms, young firms, firms with volatile returns, unprofitable firms, and non-dividend-paying firms." (lines 882–885)

Which matches the sentence at lines 632–634:
> "We will make use of this pattern in the subsequent analysis, i.e. we will condition on the first predetermined value of CEFD, NIPO, S,
> and SENTIMENT, and the lag of that value for TURN, RIPO, and PD-ND." (lines 632–634)

and the table notes (lines 3177–3179, repeated 3496–3498 and 3828–3830):
> "Coefficients are matched to the closed-end fund discount (CEFD), the number of IPOs (NIPO), and the equity share (S) for the calendar
> year one year before t. Coefficeints [sic] are matched to detrended log turnover (TURN), the average annual first-day return (RIPO), and
> the dividend premium (PD-ND) for the calendar year two years before t. All sentiment measures are orthogonalized to macroeconomic
> conditions." (lines 3177–3180)

**⇒ Effective lag from a raw data observation to the return month (the thing a replicator must get right):**
- `CEFD`, `NIPO`, `S` are matched to monthly returns of year *t* using the value from calendar year **t−1** (≈ 1 year, up to 18 months at the start of year *t*).
- `TURN`, `RIPO`, `PD-ND` are matched to monthly returns of year *t* using the value from calendar year **t−2** (≈ 2 years).
- `SENTIMENT_{t−1}` — i.e. the index formed at/for year *t−1* — conditions all monthly returns in calendar year *t*. Because the index already embeds the one-year lag on TURN/RIPO/PD-ND, using `SENTIMENT_{t−1}` reproduces the same effective lags as the individual proxies. The Table 3b header is literally `SENTIMENTt-1` (line 2583).

---

## 2. HOW THE INDEX IS BUILT, STEP BY STEP

All of this is Section IV.C, lines 485–657.

### Step 0 — Sample and frequency
- Annual proxies, 1962–2001 (line 498). Index plotted from 1962 in the last panel of Figure 1 — labeled, with a typo in the original,
  **"Panel E. Sentiment index (SENTIMENT)"** at line 1557 (the earlier panels run A–F, so this is the 7th/G panel); its x-axis ticks start
  at "1962" (lines 1565–1572).
- Because TURN, RIPO, PD-ND enter the index lagged one year, the **underlying** TURN/RIPO/PD-ND series must extend back to **1961** even though the table labels them `TURN_{t-1}` etc. (Table 2's row labels `TURNt-1`, `RIPOt-1`, `Pt-1 D-ND` are relative to the index year.)
- Monthly returns run 1963–2001 (line 106: *"We use monthly stock returns between 1963 and 2001."*; Table 1 title, line 1848: *"Summary Statistics, 1963-2001"*). So the first index value used to condition returns is the end-1962 index, for the 1963 monthly returns.

### Step 1 — Detrending (applies to TURN only)
Raw `TURN` = reported share volume ÷ average shares listed (NYSE Fact Book), then **detrended in logs by a five-year moving average**
(lines 526–528; restated line 2022). Exact wording: *"the raw turnover ratio TURN is detrended by the five-year moving average
(specifically, the log of the ratio is detrended by its moving average)"* and *"We detrend using the past five-year average."*
**Not stated in the source:** whether the 5-year average includes year *t* itself or is the mean of *t−1…t−5*, and whether it is backward or
centered. A replicator must choose; "past five-year average" points to a backward-looking window. Table 2 Panel A shows the detrended log
turnover already (mean 5.21, SD 7.94, min −11.60, max 18.66) with **no stated units** — the paper never says whether these are log × 100.

### Step 2 — Orthogonalize / residualize each raw proxy on macro variables FIRST (yes — this precedes the PCA)
Lines 499–505, verbatim:
> "To isolate the sentiment component of the proxies from business cycle components, we orthogonalize each proxy with respect to several
> macroeconomic variables. Specifically, we regress each of the raw proxies on growth in the industrial production index (Federal Reserve
> Statistical Release G.17), growth in consumer durables, nondurables, and services (all from BEA National Income Accounts Table 2.10),
> and a dummy variable for NBER recessions. We take the residuals from these regressions as a cleaner proxy that is independent of major
> business cycle effects. All of our results are based on these cleaned proxies." (lines 499–505)

**Exact macro variable list — TWO DIFFERENT LISTS EXIST IN THIS WP:**
- **Body text (lines 501–503)** — three regressors: (i) growth in the industrial production index (Fed G.17); (ii) growth in consumer
  durables, nondurables, and services (BEA NIPA Table 2.10); (iii) a dummy variable for NBER recessions.
- **Table 2 note (line 2028) and Figure 1 note (lines 1402–1403)** — four regressors: *"the growth in industrial production, the growth in
  durable, nondurable and services consumption, **the growth in employment** and a flag for NBER recessions."*

The only difference is **growth in employment**, which appears **only** in the two exhibit notes, never in the body text, and for which
**no source is given**. A replicator should either use the 4-variable list (the exhibit notes) or test both. Second verbatim instance:
*"We regress each measure on the growth in industrial production, the growth in durable, nondurable and services consumption, the growth in
employment and a flag for NBER recessions. The solid line (right axis) is the residuals from this regression."* (lines 1401–1404)

**Frequency of this regression: NOT STATED.** The paper does not say annual or monthly, does not give an equation, and reports no
coefficients, R², or observation counts. Strong implication is **annual** (the dependent variables are annual series measured 1962–2001,
i.e. 40 observations), but this is an inference, not a quotation. A monthly version would require applying an annual proxy to monthly macro
data, which the text does not describe.

**What is carried forward:** the **residual** — lines 504–505 ("We take the residuals from these regressions as a cleaner proxy… All of our
results are based on these cleaned proxies"; cf. line 656: *"we consider only the cleaned proxies in our subsequent empirical work, in order
to isolate the effect of sentiment from macroeconomic conditions."*).

Result of cleaning: the paper notes it barely changes the proxies' time-series behaviour — *"cleaning the proxies of macro conditions has
little effect on their time-series properties. Indeed, Table 2 suggests that the cleaned proxies are equally if not more correlated with each
other than are the raw proxies, suggesting that investor sentiment can be distinguished from macroeconomic conditions."* (lines 652–655)

### Step 3 — Standardize
Stated for equation (2): *"where for Eq. (2) each of the six components was first cleaned of macroeconomic conditions and standardized."*
(lines 624–625). Standardization is also implied by the use of the **correlation** matrix (line 572–573 below) and by line 879
(*"The sentiment proxies are all standardized."*). The exact standardization (z-score, sample vs population SD) is not spelled out but is the
standard z-score; the "clean" residuals are already mean-zero (Table 2 Panel B shows mean 0.00 for every cleaned proxy).

### Step 4 — First stage: PCA on **12** terms (six proxies + their lags), twice the number used in the final index
Verbatim (lines 567–574), the complete description of the two-stage procedure:
> "We also form a novel composite index SENTIMENT in order to capture the common factor in the six proxies, and to help us identify their
> best relative timing. The procedure is as follows. We start by estimating the first principal component of the six proxies and their lags.
> This gives us a first-stage index with twelve loadings, one for each of the current and lagged proxies. We then compute the correlation
> between the first-stage index and the current and lagged values of each of the proxies. Finally, we construct SENTIMENT as the first
> principal component of the correlation matrix of six variables – each proxy's lead or lag, whichever has the higher correlation with the
> first-stage index." (lines 567–574)

Reading this literally for a rebuild:
- **First-stage index** = first PC of the 6 cleaned+standardized proxies **at year *t*** and the same 6 at year ***t−1*** → **12 loadings**.
  (The WP does not say the words "at t and t−1"; it says "the six proxies and their lags", and 12 = 6 × 2, so one lag each is the only
  reading consistent with "twelve loadings, one for each of the current and lagged proxies".)
- Ambiguity to flag: the WP never restates that the first-stage inputs are the *cleaned* rather than the *raw* series. Given line 505
  ("All of our results are based on these cleaned proxies"), line 624 ("each of the six components was first cleaned … and standardized"),
  and Table 2 Panel B (which reports the cleaned series as the objects being correlated with SENTIMENT), **the defensible rebuild uses the
  cleaned, standardized series in both stages.**
- **Selection rule:** for each proxy, compute its correlation with the first-stage index at the lead (current, *t*) and at the lag (*t−1*),
  and keep whichever has the **higher correlation with the first-stage index**. Six winners survive.
- The WP states the selection criterion but does **not** enumerate which proxy won which slot. The winners are inferable from the printed
  equation (2) and are confirmed by the paper's own statements: **CEFD_t, TURN_{t−1}, NIPO_t, RIPO_{t−1}, S_t, PD-ND_{t−1}.**

### Step 5 — Final index
**SENTIMENT = first principal component of the correlation matrix of the six selected lead/lag series** (line 572–573: *"the first principal
component of the correlation matrix of six variables"*). Note the words **correlation matrix** — the inputs are standardized; the WP does
**not** say the loadings are rescaled to sum to one, nor that scores are normalized to unit variance in the construction step (but they are
used as standardized series afterwards — Table 3b note: *"the standardized SENTIMENT (orthogonalized to macroeconomic conditions) from the
previous year end is positive, negative"*, lines 2576–2580; Table 5 note: *"each standardized to have unit variance"*, line 3162).

### Step 6 — Reported properties of the final index
> "The first principal component explains 53% of the (standardized) sample variance, and only the first eigenvalue is above 1.00, so we
> conclude that one factor captures the common variation. The correlation between the twelve-term first-stage index and the SENTIMENT index
> is 0.96, suggesting that little information is lost in dropping six terms." (lines 625–628)

- **Variance explained by PC1: 53%** (of the standardized sample variance).
- **Eigenvalue criterion:** only the first eigenvalue exceeds **1.00** (Kaiser-type rule) → one common factor.
- **Correlation between the 12-term first-stage index and the final 6-term SENTIMENT: 0.96.**

### Step 7 — Sign, scaling, updating, and use
- **Sign convention:** the index is signed so that **higher = higher sentiment** (i.e. a positive SENTIMENT means optimistic), which is why
  CEFD and PD-ND load negatively. The paper's own reading of a high vs low index: sorts are run on "months where the standardized SENTIMENT
  … from the previous year end is positive, negative" (lines 2576–2580), and Table 3a's analogous split uses "when CEFD is positive or
  SENTIMENT is negative" to denote **low** sentiment (lines 686–688: *"the cross-sectional size effect of Banz (1981) exists in
  low-sentiment conditions only, i.e. when CEFD is positive or SENTIMENT is negative"*).
- **No further normalization is described.** There is no statement that the loadings are divided by their sum, that the index is rescaled to
  a particular mean/SD, or that it is expressed in a "0–1" or "−1 to 1" range. The only stated transformation is standardization
  (z-scoring) of the six components before PCA, and the later use of the index as a standardized variable.
- **Updating frequency: annual.** All proxies are annual; the index is formed once per calendar year and is used as the conditioning variable
  for the following calendar year and, in the tables, is indexed `SENTIMENT_{t-1}`.
- **Matching to monthly cross-sectional returns:** `SENTIMENT_{t−1}` (end of the previous calendar year) conditions every monthly return in
  calendar year *t* (lines 667–669, 882–884, and Table 3b note/header lines 2576–2583).
- **Rotation:** **none is mentioned anywhere in the WP** (no varimax, no orthogonal/oblique rotation, no "rotated index" language). The index
  is the unrotated first principal component of the correlation matrix.
- **Index data range:** 1962–2001, i.e. **40 annual values**; the regression tests use the 39 values `SENTIMENT_1962 … SENTIMENT_2000` to
  condition the 1963–2001 monthly returns.

### 2.6 Equation (2) reconstructed, with evidence (because the PDF text is scrambled)

The raw extraction of equation (2) is at lines 580–623 and is badly reordered by the PDF→text conversion. Here is the raw fragment dump,
and the reconstruction with the fragment count that verifies it.

First block (lines 580–596), fragments in extraction order:
`SENTIMENTt =` · `t` · `t` · `t` · `NIPO` · `TURN` · `CEFD` · `414` · `.0` · `402` · `.0` · `358` · `.0` · `1 +` · `+` · `−` · `−`

Second block (lines 599–622), fragments in extraction order:
`ND` · `D` · `t` · `t` · `t` · `P` · `S` · `RIPO` · `−` · `−` · `−` · `−` · `+` · `+` · `1` · `1` · `.` · `431` · `.0` · `371` · `.0` · `464` · `0`

**Reconstruction:**
```
SENTIMENT_t = − 0.358 · CEFD_t
              + 0.402 · TURN_{t−1}
              + 0.414 · NIPO_t
              + 0.464 · RIPO_{t−1}
              + 0.371 · S_t
              − 0.431 · (PD − ND)_{t−1}
```
**Verification by counting fragments** (this is a consistency check, not a guess):
- Magnitudes, in order: 0.414 → NIPO, 0.402 → TURN, 0.358 → CEFD (stacked directly under their variables); 0.464 → RIPO, 0.371 → S, 0.431 → (PD−ND).
- Subscript letters: exactly **6** `t` fragments (one per term) ✔.
- Subscript `"1"` fragments: **3** (block 1 has one, block 2 has two) ⇒ exactly **three** terms are lagged `t−1`: TURN, RIPO, PD-ND ✔ — matches the Table 2 note ("Turnover, the average annual first-day return, and the dividend premium are lagged one year relative to the other three measures") and the row labels `CEFDt / TURNt-1 / NIPOt / RIPOt-1 / St / Pt-1 D-ND` (lines 2051–2099).
- Minus signs: **2** in block 1 (one coefficient minus for CEFD, one subscript minus for `t−1`) and **4** in block 2 (coefficient minus for PD-ND, the `D−ND` hyphen, and two subscript minuses) ✔.
- Plus signs: **2** per block (two signs linking three terms) ⇒ total 4 positive terms, 2 negative terms ✔ — matches "each individual sentiment proxy enters with the expected sign" given CEFD and PD-ND are the two inverse-sentiment proxies.
- The `"0"` and `"."` stragglers are the split leading digits/points of "0.464" and ".431".

**Independent corroboration of the timing pattern** (Table 2 Panel B row labels, lines 2113–2172):
`CEFDt` · `TURNt-1` · `NIPOt` · `RIPOt-1` · `St` · `Pt-1 D-ND` — identical ordering and lags.

**Loadings are not equal-weighted:** they range from 0.358 to 0.464 in absolute value, largest on RIPO (0.464) and smallest on CEFD (0.358).

---

## 3. ADDITIONAL REQUIRED DETAIL

### 3.1 Sample period of the index
**1962–2001, annual** (line 498: *"The sentiment proxies are measured annually from 1962 through 2001."*). Figure 1's final panel plots the
index from 1962 (axis label lines 1565–1572). Monthly returns are 1963–2001 (lines 106, 1848), conditioned on `SENTIMENT_{t−1}`.
The anecdotal history section covers 1961–2002 (line 266) but that is not the estimation sample.

### 3.2 Individual proxy correlations with the final SENTIMENT index (Table 2, lines 2019–2172)

Header order of the correlation columns (line 2042–2048): **SENTIMENT, CEFD, TURN, NIPO, RIPO, S, PD-ND**. Significance: a = 1%, b = 5%, c = 10% (lines 2029–2030).

**Panel A — raw data (correlation with SENTIMENT in the first correlation column):**

| Proxy (row label as printed) | Mean | SD | Min | Max | Corr. with SENTIMENT |
|---|---|---|---|---|---|
| `CEFDt` | 8.83 | 8.11 | −10.41 | 23.70 | **−0.60ᵃ** |
| `TURNt-1` | 5.21 | 7.94 | −11.60 | 18.66 | **0.69ᵃ** |
| `NIPOt` | 358.41 | 262.76 | 9.00 | 953.00 | **0.66ᵃ** |
| `RIPOt-1` | 16.94 | 14.93 | −1.67 | 69.53 | **0.80ᵃ** |
| `St` | 19.53 | 8.34 | 7.83 | 43.00 | **0.43ᵃ** |
| `Pt-1 D-ND` | 0.20 | 18.67 | −33.17 | 36.06 | **−0.76ᵃ** |

**Panel B — controlling for macroeconomic conditions (the cleaned series; these residuals are what feed the index):**

| Proxy | Mean | SD | Min | Max | Corr. with SENTIMENT | CEFD | TURN | NIPO | RIPO | S | PD-ND |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `CEFDt` | 0.00 | 6.15 | −18.33 | 9.16 | **−0.64ᵃ** | 1.00 | | | | | |
| `TURNt-1` | 0.00 | 6.73 | −11.31 | 11.45 | **0.72ᵃ** | −0.29ᶜ | 1.00 | | | | |
| `NIPOt` | 0.00 | 226.30 | −435.98 | 484.15 | **0.74ᵃ** | −0.46ᵃ | 0.39ᵇ | 1.00 | | | |
| `RIPOt-1` | 0.00 | 14.31 | −23.55 | 46.54 | **0.83ᵃ** | −0.45ᵃ | 0.53ᵃ | 0.44ᵃ | 1.00 | | |
| `St` | 0.00 | 6.15 | −12.17 | 14.29 | **0.66ᵃ** | −0.40ᵇ | 0.32ᵇ | 0.50ᵃ | 0.47ᵃ | 1.00 | |
| `Pt-1 D-ND` | 0.00 | 16.89 | −43.20 | 35.96 | **−0.77ᵃ** | 0.28ᶜ | −0.60ᵃ | −0.46ᵃ | −0.68ᵃ | −0.28ᶜ | 1.00 |

**The two weakest individual correlations with the composite** (the item requested):
- **Cleaned Panel B (the relevant one): `CEFD` at |−0.64| and `S` (equity share) at |0.66| are the two lowest.**
- **Raw Panel A: `S` at 0.43 and `CEFD` at −0.60 are the two lowest.**
- So **S (equity share) and CEFD are consistently the weakest**, and RIPO (0.83) and PD-ND (−0.77) the strongest, in the WP.
- **Caveat a replicator must know:** these are *in-sample* correlations between each proxy and an index **that contains that same proxy**, so
  they are mechanically inflated and are not a clean measure of a proxy's marginal value. The WP does not report leave-one-out
  (index-without-that-proxy) correlations anywhere. Also, I could **not** verify whether the published JF version reports different values.

### 3.3 The paper's main result — which stocks are most affected ("sentiment-prone")

Conclusion, verbatim (lines 1195–1205):
> "To summarize, the main empirical finding is that the cross-section of future stock returns is conditional upon beginning-of-period proxies
> for investor sentiment. … Specifically, when sentiment appears to be high, stocks that are likely to be relatively attractive to optimists
> and speculators and at the same time unattractive candidates for arbitrage – **young stocks, small stocks, unprofitable stocks,
> non-dividend paying stocks, high-volatility stocks, extreme-growth stocks, and distressed stocks** – experience low future returns relative
> to other stocks. Conditional on low sentiment, on the other hand, these cross-sectional patterns attenuate or reverse. Several aspects of
> the results cast doubt on the hypothesis that they reflect rational compensation for systematic risk." (lines 1195–1205)

Abstract (lines 36–39) and intro (lines 108–120) give the same list and add the mechanism: *"When sentiment is low, subsequent returns are
relatively high on smaller stocks, high volatility stocks, unprofitable stocks, non-dividend-paying stocks, extreme-growth stocks, and
distressed stocks, consistent with an initial underpricing of these stocks. When sentiment is high, on the other hand, these patterns
attenuate or fully reverse."*

Selected magnitudes actually printed in the WP text (Table 3b, conditions on `SENTIMENT_{t-1}` divided at zero):
- **Size:** *"when SENTIMENT is negative, returns average 2.33 percent per month for the bottom ME decile and 0.91 for the top decile"*
  (lines 688–690).
- **Age:** *"When SENTIMENT is pessimistic, top-decile Age firms return 0.47 percent less than bottom-decile Age firms, but return
  0.72 percent more when SENTIMENT is optimistic."* Age *"shows no unconditional effect at all"* (lines 705–711).
- **Volatility (Sigma):** *"high Sigma stocks appear to be out of favor when SENTIMENT is low – they earn returns of 2.25 percent per
  month"*, and the effect fully reverses in high sentiment (lines 712–715).
- **Profitability / dividends:** *"when SENTIMENT is positive, monthly returns are 0.32 percent higher on profitable than unprofitable firms
  and 0.45 percent higher on payers than nonpayers. When it is negative, however, returns are 0.85 percent per month lower on profitable
  firms and 0.77 percent lower on payers."* Patterns are *"driven mostly by conditional variation in the returns of unprofitable and
  nonpaying firms"* (lines 716–725).
- **Extreme growth & distress (BE/ME, EF/A, GS):** a **U-shaped** conditional pattern — *"when sentiment is low, the subsequent returns on
  stocks at both extremes are especially high relative to their unconditional average, while stocks in middle deciles are less affected by
  sentiment"* (lines 121–130, 741–768).
- **Regression magnitude example:** *"the 0.1 coefficient of SENTIMENT on ME means that a two-SD increase in the sentiment index raises the
  size coefficient by 0.2 percentage points per month"* (lines 897–899).
- **Risk explanation rejected:** betas with market returns / consumption growth show no matching time variation (lines 133–148, 1055+), and
  earnings-announcement patterns suggest expectational errors (lines 146–147).
- **Which individual proxy is strongest in the regressions:** *"those using IPO volume as the sentiment proxy are particularly strong"*
  (lines 892–893).
- **On low sentiment predicting higher market-wide returns:** *"subsequent returns tend to be higher across the board when sentiment is low.
  This is consistent with prior results that the equity share and turnover, for example, forecast market returns."* (lines 696–699)

---

## 4. (16) FEASIBILITY ASSESSMENT FOR A NON-US MARKET

### 4.1 What the user has, mapped onto BW's six proxies

| BW proxy | User's available item | Verdict | What is required to actually replicate BW's construction |
|---|---|---|---|
| **CEFD** (closed-end fund discount) | — none — | **NOT reproducible / NOT approximable** | Needs listed **closed-end funds** with both market price and published **NAV** per share, a **value-weighted average discount** across closed-end **stock** funds, at **year-end**. None of the six data items the user holds substitutes for it. |
| **TURN** (detrended log turnover) | "market turnover/volume" | **Reproducible in principle**, with caveats | Needs turnover = share volume ÷ average shares listed (or free float) for the *same* market, an **annual** series back to ~5 years before the index start (detrending) **plus 2 more years** (5-yr MA + the extra index lag + the return-matching lag). Then take **logs and subtract a 5-year moving average**. Watch for market-structure breaks (BW explicitly flag the May 1975 US commission deregulation as a contaminating event, lines 525–529). |
| **NIPO** (number of IPOs) | "number of IPOs" | **Reproducible** | Annual **count** of IPOs. BW use raw counts (mean 358/yr, max 953), not scaled by market size. |
| **RIPO** (average first-day return) | "IPO first-day returns" | **Reproducible** | **Average** first-day return across that year's IPOs (BW's mean 16.94%, SD 14.93%). Must be the *average across IPOs*, not an index of IPO performance. |
| **S** (equity share in new issues) | "equity share in new issues" | **Reproducible only if it is BW's definition** | BW: **gross equity issuance ÷ (gross equity + gross long-term debt issuance)** — i.e. a *composition* measure built from total external finance (footnote 7, lines 545–548). It **includes seasoned offerings**, not just IPOs. If the user's series is IPO-proceeds-only, or net rather than gross, or excludes debt, it is a different variable. |
| **PD-ND** (dividend premium) | "dividend premium" | **Reproducible only under BW's definition** | BW: **year-end log( value-weighted average M/B of dividend payers ) − log( value-weighted average M/B of nonpayers )**. A "dividend premium" defined as a yield/payout spread or as a raw ratio (not log, not value-weighted, not M/B based) is **not** BW's PD-ND. |

**Net verdict: 5 of the 6 BW proxies are approximable from the user's data; the 6th (CEFD) is not available at all.**
Additionally, all six user data items are **market-level annual inputs** and all six BW proxies are **annual** — the annual frequency is not a
constraint here. The binding constraints are (a) the missing CEFD and (b) the exact definitions of S and PD-ND.

### 4.2 Minimum data span required
For an index covering years `T0 … T1` and returns matched as BW do:
- CEFD, NIPO, S: need calendar years `T0 … T1` (matched to returns with a 1-year lag).
- TURN, RIPO, PD-ND: need calendar years `T0−1 … T1−1` **inside the index** (matched to returns with a 2-year lag), and TURN's 5-year
  detrending window pushes its raw series back another ~5 years, to about `T0−6`.
- ⇒ Building a BW-style index from, say, 1995 onward requires turnover data from ~1989 and all other series from 1994.

### 4.3 Is CEFD essential? What is its role? (what the paper actually says)
1. **Mechanically it is one of six PC inputs.** Dropping it changes every loading (the index is a *first principal component*, so removing a
   variable re-solves the whole eigenproblem). The WP does **not** report any "without-CEFD" robustness index, so there is no evidence in
   this source about how much the index changes if CEFD is dropped.
2. **Conceptually it is the paper's flagship "general" indicator** (lines 678–683): *"CEFD is perhaps our cleanest 'general' indicator of
   investor sentiment – it is not mechanically connected to sentiment for any one segment of the cross-section."* The paper deliberately
   re-runs its headline sorts on **CEFD alone** (Table 3a) precisely so it can show a *general* sentiment effect that is not an artifact of
   one corner of the cross-section — the same reason it distrusts NIPO and PD-ND (**"the volume of IPOs and the dividend premium could in
   principle be driven by a concentrated demand for firms with IPO characteristics or dividends, respectively"**, lines 681–682).
3. **It is the timing "exception"** in the index: it is the one price/investor-behavior proxy that enters contemporaneously rather than
   lagged (lines 629–631).
4. **It links to a classic result:** *"a link between the size effect and closed-end fund discounts was previously noted by Swaminathan
   (1996)"* (lines 690–692).
5. **Assessment (mine, not the paper's):** the index can be *computed* without CEFD, but the resulting object is no longer BW's SENTIMENT —
   it is a 5-proxy variant whose loadings, timing choices and variance-explained share will all differ, and it loses the only component the
   authors describe as free of single-segment mechanical ties. In a non-US market the honest options are (i) find closed-end funds with NAV
   data in that market and build CEFD, (ii) substitute another *general* sentiment proxy (e.g. a fund-flow or retail-participation measure)
   and document the substitution, or (iii) run the 5-proxy index and label it explicitly as a BW-style variant rather than BW's index.
   Note that **the paper itself never substitutes a proxy**, so any substitution is outside the paper's evidence base.

### 4.4 Which proxies carry the weakest signal (for prioritisation when data are scarce)
From the WP's own Table 2 (see §3.2): the **equity share S** and the **closed-end fund discount CEFD** are the two with the lowest
absolute correlation with the composite (S: 0.43 raw / 0.66 cleaned; CEFD: 0.60 raw / 0.64 cleaned), while RIPO and PD-ND are highest
(0.83 and −0.77 cleaned). Two important qualifications from the paper itself:
- These correlations are with an index that *includes* the proxy (in-sample inflation), so they understate nothing and overstate everything.
- Which proxies work best in the **regression tests** is a different question: *"those using IPO volume as the sentiment proxy are
  particularly strong"* (lines 892–893).
So on the WP's own numbers, the user's weakest links are **S** (and the missing CEFD), while **NIPO, RIPO and TURN** carry the most weight.

### 4.5 What is NOT stated in the source read (do not fill these in from memory)
- The orthogonalization regression's **frequency** (annual vs monthly), its **functional form**, its **sample**, and **whether employment
  growth is a regressor** (body text says no, exhibit notes say yes).
- The exact **turnover detrending window alignment** ("past five-year average" — includes year *t*? centered?).
- The **units of TURN** in Table 2 (log × 100? raw log?), and the units of the S residual.
- The **first-stage loadings** (only the final six are printed) and any explicit statement of which proxies won the lead/lag contest, other
  than what can be inferred from Eq. (2) and the row labels.
- **No rotation** of any kind; **no alternative index** without CEFD; **no leave-one-out correlations**.
- Anything about the **published JF (2006) version**'s numbers (different sample end, different loadings, different variance-explained) —
  Wiley returned 403 and no mirror could be retrieved, so **no published-version number is asserted here**.

---

## 5. REBUILD CHECKLIST (pseudocode, annual, index years 1962–2001)

```
for each year y in 1962..2001:
    CEFD[y]   = year-end value-weighted avg discount of closed-end STOCK funds      # %
    TURN[y]   = log(volume[y] / avg_shares_listed[y]) - mean(log(volume/avg_shares) over 5-yr window)
    NIPO[y]   = count of IPOs in year y
    RIPO[y]   = average first-day return of IPOs in year y                          # %
    S[y]      = gross equity issuance[y] / (gross equity[y] + gross LT debt[y])
    PDND[y]   = log(vw avg M/B of payers[y]) - log(vw avg M/B of nonpayers[y])     # year-end

# (1) residualize each raw proxy on macro controls (annual regressions), keep residuals
for p in {CEFD,TURN,NIPO,RIPO,S,PDND}:
    resid_p = residual of  p ~ [growth IP, growth cons.durables, growth cons.nondurables,
                                growth cons.services, (growth employment?), NBER recession dummy]

# (2) standardize residuals (z-scores); call them z_p

# (3) FIRST STAGE: first PC of the 12 series {z_p(t), z_p(t-1) for all six p} -> F1  (12 loadings)

# (4) For each proxy p: corr(F1, z_p(t))  vs  corr(F1, z_p(t-1)); keep the winner w_p (call it z*_p)

# (5) FINAL: SENTIMENT = first PC of the correlation matrix of the six z*_p
#     WP result (1962-2001): PC1 explains 53%; only eigenvalue > 1.00; corr(F1, SENTIMENT) = 0.96
#     WP loadings: -0.358*CEFD(t) + 0.402*TURN(t-1) + 0.414*NIPO(t) + 0.464*RIPO(t-1)
#                  + 0.371*S(t) - 0.431*PDND(t-1)

# (6) USE: condition monthly returns of calendar year t on SENTIMENT(t-1)
#     (equivalently on CEFD(t-1), NIPO(t-1), S(t-1), TURN(t-2), RIPO(t-2), PDND(t-2))
#     Sign: positive SENTIMENT = optimistic; split samples at 0 (or at the sample median, per line 108-110).
```
