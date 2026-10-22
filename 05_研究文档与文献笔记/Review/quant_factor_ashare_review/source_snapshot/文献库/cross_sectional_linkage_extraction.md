<!-- Research project author: Gilmour Jia. Cited works retain their original authors. -->

# Deep Extraction: Cross-Sectional / Economic-Linkage Return Predictability

**Prepared for:** quantitative factor construction, A-share adaptation
**Standard applied:** body-text / methodology-level reading (Data, Methodology, Portfolio Construction, Empirical Results, Appendix). Abstracts were not used as a source.

---

## 0. SOURCE INVENTORY — WHAT WAS ACTUALLY READ, AND WHAT WAS NOT

| Paper | What I read | Full text? | URL actually read |
|---|---|---|---|
| **Cohen & Frazzini (2008)** | **Published JF version**, 35 pp, incl. Tables II–VII and Section VI | **YES — full published text** | `https://people.stern.nyu.edu/afrazzin/pdf/Economic%20Links%20and%20Predictable%20Returns%20-%20Cohen%20and%20Frazzini.pdf` |
| Cohen & Frazzini — **Appendix** | 13 pp, incl. **Table A1 "Supplier momentum"**, Table A2 cross-industry, Table A3 out-of-sample | **YES** | `https://pages.stern.nyu.edu/~afrazzin/pdf/Economic%20Links%20and%20Predictable%20Returns%20-%20Appendix%20-%20Cohen%20and%20Frazzini.pdf` |
| Cohen & Frazzini — working paper | 41 pp, Feb 2006 draft (used for cross-checks where the published version is terser) | YES | `http://www.econ.yale.edu/~shiller/behfin/2006-04/cohen-frazzini.pdf` |
| **Lee, Sun, Wang & Zhang (2019)** | **Working-paper version dated 23 Oct 2017**, 61 pp (ABFER copy), incl. Section 3 Data/Variables, Table 2, Table 3, Section 5 robustness, Section 6 mechanism, Appendix Table A2 variable definitions | **YES (WP version)** | `https://abfer.org/component/edocman/senior-fellows-fellows/technological-links-and-predictable-returns/download` |
| **Parsons, Sabbatucci & Titman (2020)** | **AEA 2020 conference version**, 65 pp, incl. Sections 2–5, Tables 1–6, Appendices A1–A9 | **YES (conference version)** | `https://www.aeaweb.org/conference/2020/preliminary/paper/SfTyaRaf` |
| **Menzly & Ozbas (2010)** | **Actual full SSRN text, 65 pp**, "Journal of Finance forthcoming" version, incl. Sections I–IV, Tables I–VII, and the full trading-strategy section | **YES — full text** | `https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID1496974_code84406.pdf?abstractid=989080` |
| Menzly & Ozbas — Internet Appendix | 28 pp official supplement (limited-information model + Tables IA.I–IA.VIII) | YES | `https://afajof.org/wp-content/uploads/files/supplements/Internet_Appendix_to_Market_.pdf` |

### Explicit gaps / version caveats — read these before citing numbers

1. **Lee et al.: I read the Oct-2017 working paper, not the final JFE 132(3) 76–96 text.** The published version is hard-paywalled — Unpaywall returns `is_oa: false`, `oa_status: "closed"`, zero OA locations (`https://api.unpaywall.org/v2/10.1016/j.jfineco.2018.11.008`). The WP's headline is **117 bps/month EW alpha**, and the published abstract repeats the "technological closeness" language; I could not verify whether the final published tables renumber or revise magnitudes. **Treat the 117 bps as the WP-version number.**
2. **Parsons et al.: I read the AEA 2020 conference version, not the final RFS 33(10) 4721–4770 text.** Also closed (`10.1093/rfs/hhz145` → `is_oa: false`). The conference version's L/S figure is **42 bps/month raw (t=3.65)**, i.e. ~5%/yr, which is consistent with the published abstract's "5%–6% annually," so I am reasonably confident the numbers carried over — but table numbering may differ.
3. **Menzly & Ozbas: full main text obtained**, so no gap. Note that the version I read is the "JF forthcoming" SSRN manuscript; page/table numbers may differ slightly from the printed JF article (65:1555–1580).
4. **Papers I could NOT read at all:** none. All four were obtained at body-text level.

### ⚠️ CORRECTION TO THE BRIEF: Menzly & Ozbas are NOT "analyst-coverage-based links"

The task brief described Paper 4 as *"(analyst-coverage-based links)."* **That is wrong about the linkage definition.** In Menzly & Ozbas (2010):

- The **linkage network** is **BEA Benchmark Input-Output Use Table commodity flows between INDUSTRIES** (upstream/downstream vertical links). It is an **industry-level**, not firm-level, network.
- **Analyst coverage and institutional ownership are the cross-sectional MODERATOR** used to test the mechanism (the effect should be weaker where more informed investors are), **not the link**.

Direct quote (p. 8 of the SSRN text): *"BEA Surveys provide a detailed picture of the interdependent structure of the U.S. economy by assigning gross output to industry accounts and by reporting the amount of inter-industry flow of goods and services in the Use Table… In calculating monthly returns for these portfolios, we use data on the flow of goods and services to and from the industries in question as portfolio weights."*

This matters for replication: you cannot build Mozbas links from analyst data, and you cannot build them from OHLCV. You need an input-output table.

---

# PAPER 1 — Cohen & Frazzini (2008), *Economic Links and Predictable Returns*, JF 63(4), 1977–2011

## 1.1 THE EXACT FACTOR CONSTRUCTION RECIPE

### Data vintage and sample
- **Links:** Compustat segment files (principal customers). Regulation **SFAS No. 131** (effective for fiscal years beginning after 15 Dec 1997) and its predecessor **SFAS No. 14**. Firms must disclose **the identity of any customer representing more than 10% of total reported sales**.
- **Link sample:** 1980–2004. **30,622 distinct firm-year relationships**, **11,484 unique supplier–customer relationships**.
- **Return tests:** **1981–2004 = 288 months.**
- Sample = CRSP/Compustat firms with non-missing BE and ME at fiscal-year end, for which the customer is *also* a traded CRSP/Compustat firm. **Common stocks only (CRSP share codes 10 and 11).**
- **Name matching:** pre-1998 customers are listed as abbreviations that vary across firms/time. They use a **"soundex" phonetic string-matching algorithm** to generate candidate matches, then **hand-match** to PERMNO by inspecting firm name, segments and industry. *"We are deliberately conservative in assigning customer names and firm identifiers."* **Non-unique matches are dropped.**
- **Look-ahead control:** a **six-month gap** is imposed between fiscal year-end and the stock returns the link is used to explain (mimicking the standard accounting-to-returns gap).

### The linked-portfolio signal ("customer return")
> *"Since some firms in our sample have multiple principal customers over many periods, we construct an equally weighted portfolio of the corresponding customers using the last available supplier–customer link. We rebalance these portfolios every calendar month. Hereafter, we refer to the monthly return of this portfolio as the customer return."*

- **Base specification: EQUAL-WEIGHTED** average of the principal customers' monthly returns. (Footnote 10: results are identical using sales weights = percentage of total sales to each customer; they chose EW because the dollar sales split is **missing in ~19% of firm-year observations**.)
- **Window: one month (t−1).** A 12-month variant is also run (see below).
- **"Last available link"** — i.e., the most recently disclosed customer relationship, held until a new one is disclosed.
- **Signal type: a RAW RETURN.** It is *not* a residual return and *not* a characteristic. The paper is explicit: *"In our base specification, we use the monthly customer return as a proxy for news about customers. We believe that a return-driven news sort is appropriate because it closely mimics the underreaction hypothesis at hand."*

### Sorting and portfolio formation
> *"At the beginning of calendar month t, we rank stocks in ascending order based on the customer returns in month t−1 and we assign them to one of five quintile portfolios. All stocks are value (equally) weighted within a given portfolio, and the portfolios are rebalanced every calendar month to maintain value (equal) weights."*

- **Sort: 5 quintiles, UNIVARIATE (dependent) sort** on lagged customer return. No independent/double sorts in the baseline.
  - *(Independent sorts into low-30 / mid-40 / high-30 on `COMOWN` are used only in Section VI, and the L/S is then computed **within** each ownership group.)*
- **Weights: VALUE-WEIGHTED is the headline panel**; EQUAL-WEIGHTED is reported as a parallel panel.
- **Holding period: 1 month. Rebalanced monthly.**
- **L/S = top 20% (high customer return) minus bottom 20% (low customer return).**
- **Eligibility filters:** non-missing customer return and non-missing stock price at the end of the previous month; **price > $5** at portfolio formation (*"This ensures that portfolio returns are not driven by microcapitalization illiquid securities"*). Relaxing the $5 cutoff leaves all results robust.
- **12-month variant:** sort on the 1-year customer return, **skipping month t−1** between the ranking period and portfolio formation.

### Pseudocode

```
# ---------- STEP 1: build the link table (Compustat segment files) ----------
for each fiscal_year FY, each firm S with a disclosed principal customer:
    raw_name  = customer name string as reported in the segment file
    C         = soundex_match(raw_name) -> CRSP PERMNO   # hand-verified, unique only
    if C is None or not unique: DROP
    store (S, C, FY, pct_sales_to_C)

# ---------- STEP 2: enforce the 6-month information gap ----------
link_effective_date(S, C) = FY_end(S) + 6 months
# the link is "known" only from that date onward

# ---------- STEP 3: monthly "customer return" for each supplier ----------
for each month t, each supplier S:
    links = last_available_link_per_customer(S, as_of = t)
    # NB: "last available" = most recent disclosure, not the first
    custs = [C for (S, C, ...) in links]
    if custs is empty: skip S
    CUST_RET[S, t-1] = mean( RET[C, t-1] for C in custs )       # EQUAL-weighted (base)
    # alternative: sum(w_C * RET[C, t-1]), w_C = pct_sales_to_C / sum(pct_sales)

# ---------- STEP 4: sort and form portfolios at the start of month t ----------
universe = { S : CUST_RET[S, t-1] exists
               and PRICE[S, t-1] exists
               and PRICE[S, t-1] > 5 }
rank universe ascending on CUST_RET[S, t-1] -> quintiles Q1..Q5
Q1 = lowest customer return ... Q5 = highest customer return

for q in 1..5:
    R_q[t] = value_weighted_mean( RET[S, t] for S in Qq )   # VW headline
    # EW panel: equal_weighted_mean(...)
L S[t] = R_5[t] - R_1[t]

# ---------- STEP 5: alphas ----------
alpha = intercept of OLS( (R_q[t] - Rf[t]) ~ MKT + SMB + HML [+ MOM] [+ PS_liquidity] )
```

### SIGN CONVENTION — get this right
**LONG the supplier whose CUSTOMERS had HIGH past-month returns; SHORT the supplier whose CUSTOMERS had LOW past-month returns.**
The sort variable is the *customer's* return. The traded asset is the *supplier*. The predictability runs **customer → supplier**, i.e. a shock to the customer predicts the supplier's subsequent return.

---

## 1.2 HOW THE NETWORK IS BUILT, AND WHICH DIRECTION THE PREDICTABILITY RUNS

**Source:** Compustat segment files (principal-customer disclosures under SFAS 131 / SFAS 14). **10% of consolidated sales** disclosure threshold.

**Direction — stated explicitly in the published paper (p. 1986):**
> *"The current financial regulation, however, requires firms to report major customers (and not major suppliers). Given the presence of the 10% cutoff, our sample has more information about customers who are major stakeholders, and not the reverse. **Thus, our main tests are in the direction of suppliers' stock price response to customers' shocks.**"*

So: the published headline is **customer → supplier**. The paper's own label is **"customer momentum,"** and it is the *supplier's* stock that is bought/sold.

**Cross-industry character (critical for the A-share proxy):**
- 78% of firm–customer relations are between firms in **different** industries (48 Fama–French industries). Table I: only **22.7%** of firm–customer pairs are in the same industry on average.
- *"Thus, the stock return predictability we analyze is mostly related to assets in different industries as opposed to securities within the same industry."*

**The reverse direction (supplier → customer) — Appendix Table A1.** This is a genuine and important nuance, and it is **not** the headline:
> *"A natural extension is to test whether suppliers' stock returns forecast customers' stock returns. Unfortunately, since regulation SFAS No. 131 require firms to report major customers with a 10% sales cutoff but not major suppliers, inverting the links generates only an incomplete sample of suppliers."*

They handle this with a conditioning variable:
- `CSALE` = supplier's sales to a given customer ÷ the customer's total market value.
- `IMPSUPP` = dummy = 1 if `CSALE` at the end of the previous month is **above the 75th percentile** of all firms that calendar month.
- Result: *"the results in Table A1 show evidence of supplier momentum: past returns of important suppliers forecast subsequent stock returns of the corresponding customers."*

**Bottom line for replication:** both directions exist, but (i) the **main, published, un-conditional** effect is **customer → supplier**, and (ii) the reverse **supplier → customer** effect only appears after conditioning on *important* suppliers by relative size of the sales relationship.

**A second caveat Menzly & Ozbas raise about the same data (Mozbas p. 5):**
> *"using the customer information database to identify supplier firms (in effect, by using reported relationships in the reverse direction) and then testing for cross-predictability effects from supplier firms yield no significant results. This is because the customer firms reported in COMPUSTAT are typically much larger than the reporting firm."*
Also: *"only the smallest stocks exhibit cross-predictability effects based on lagged COMPUSTAT customer returns."*

---

## 1.3 SLOW-DIFFUSION REQUIREMENT AND THE STATED MECHANISM

**Hypothesis LA (Limited Attention)** — verbatim from the published text:
> *"HYPOTHESIS LA (Limited Attention): Stock prices underreact to firm-specific information that induces changes in valuation of related firms, generating return predictability across assets. In particular, stock prices underreact to negative (positive) news involving related firms, and in turn generate negative (positive) subsequent price drift."*

> *"In a world where investors have limited ability to collect and gather information, and market participants are unable to perform the rational expectations exercise to extract information from prices, returns across securities are predictable. **News travels slowly across assets as investors with limited attention overlook the impact of specific information on economically related firms.** These investors tend to hamper the transmission of information, generating return predictability across related assets."*

**Two necessary conditions they state explicitly:**
1. The information must be **publicly available before prices evolve**.
2. The information must be **salient** — something investors should reasonably be expected to gather.

They argue customer–supplier links satisfy both: links are in public financial statements, **average customer accounts for ~20% of supplier sales** (mean `PERCSALE` = 19.80%, median 14.68%), and many links are longstanding (mean link duration 2.7 years).

**Real-effects validation (Section on real activity, Table X in the WP):** during years when two firms are linked, the correlation of customer→supplier operating income rises by **38.7% (t = 3.88)** and customer→supplier sales by **51.4% (t = 8.55)**, relative to non-linked years for the *same* firm pairs, with industry-pair × date fixed effects.

---

## 1.4 REPORTED MAGNITUDES — EXACT NUMBERS

### Table III — Customer Momentum Strategy, abnormal returns, **1981–2004 (288 months)**, price > $5

**Panel A: VALUE weights** (published JF version)

| | Q1 (Low) | Q2 | Q3 | Q4 | Q5 (High) | **L/S** |
|---|---|---|---|---|---|---|
| Excess return | −0.596 | −0.157 | 0.125 | 0.313 | 0.982\* | **1.578\*** |
| | [−1.42] | [−0.41] | [0.32] | [0.79] | [2.14] | **[3.79]** |
| 3-factor alpha | −1.062\* | −0.796\* | −0.541\* | −0.227 | 0.493\* | **1.555\*** |
| | [−3.78] | [−3.61] | [−2.15] | [−0.87] | [1.98] | **[3.60]** |
| 4-factor alpha | −0.821\* | −0.741\* | −0.488\* | −0.193 | 0.556\* | **1.376\*** |
| | [−2.93] | [−3.28] | [−1.89] | [−0.72] | [1.99] | **[3.13]** |

**Panel B: EQUAL weights** (published JF version — verified directly from the published table)

| | Q1 | Q2 | Q3 | Q4 | Q5 | **L/S** |
|---|---|---|---|---|---|---|
| Excess return | −0.457 | 0.148 | 0.385 | 0.391 | 0.854\* | **1.311\*** |
| | [−1.03] | [0.38] | [1.01] | [1.01] | [2.04] | **[4.93]** |
| 3-factor alpha | −1.166\* | −0.661\* | −0.446\* | −0.304 | 0.140 | **1.306\*** |
| | [−5.27] | [−3.89] | [−2.74] | [−1.76] | [0.71] | **[4.67]** |
| 4-factor alpha | −0.897\* | −0.482\* | −0.272\* | −0.224 | 0.315 | **1.212\*** |
| | [−4.20] | [−2.89] | [−1.70] | [−1.28] | [1.61] | **[4.24]** |
| 5-factor alpha | −0.939\* | −0.549\* | −0.239 | −0.041 | 0.420\* | **1.359\*** |
| | [−4.61] | [−3.27] | [−1.38] | [−0.23] | [2.11] | **[4.79]** |

**Five-factor alphas (adding the Pastor–Stambaugh (2003) traded liquidity factor to FF-3 + Carhart MOM):**
- **VALUE-weighted L/S = 1.237% (t = 2.99)**
- **EQUAL-weighted L/S = 1.359% (t = 4.79)**
- Published text: *"The liquidity adjustment has little effect on the result: Subsequent to portfolio formation, the baseline zero-cost portfolio earns abnormal returns of 1.24% per month (t-statistic = 2.99). The results show that even after controlling for past returns or a reversal measure of liquidity, high (low) customer momentum stocks earn high (low) subsequent returns."*
- Note the asymmetry: the liquidity factor **costs** the value-weighted portfolio 14 bps (1.376 → 1.237) but **adds** 15 bps to the equal-weighted portfolio (1.212 → 1.359).

**Headline statements in the published text:**
- **"The customer momentum strategy that is long the top 20% good customer news stocks and short the bottom 20% bad customer news stocks delivers Fama and French (1993) abnormal returns of 1.45% per month (t-statistic = 3.61), or approximately 18.4% per year."** *(Note: this sentence cites the 3-factor FM-1993 alpha of 1.45% — a formulation carried over from the WP; the table's own 3-factor VW L/S figure is 1.555 with t = 3.60.)*
- **"Subsequent to portfolio formation, the baseline long–short portfolio earns abnormal returns of 1.37% per month (t-statistic = 3.12)"** — this is the 4-factor (Carhart momentum added) alpha.
- Abstract: **"A long–short equity strategy based on this effect yields monthly alphas of over 150 basis points."**
- Equal-weighted baseline alpha: **"1.3% (t-statistic = 4.93)."**
- Alphas **rise monotonically** across quintiles.

### How much survives own-firm momentum and industry momentum?

**(a) Own-firm momentum (Carhart MOM added as a 4th factor):**
- VW 3-factor 1.555 → **4-factor 1.376** (t = 3.13). Loss ≈ **18 bps, ~11.5%**. Still highly significant.
- EW 3-factor 1.306 → **4-factor 1.212** (t = 4.24). Loss ≈ **9 bps, ~7%**.
- Text: *"Adjusting returns for the stock's own price momentum by augmenting the factor model with Carhart's (1997) momentum factor has a negligible effect on the results."*

**(a′) Adding the Pastor–Stambaugh (2003) liquidity factor on top (5-factor):** VW 1.376 → **1.237 (t = 2.99)**; EW 1.212 → **1.359 (t = 4.79)**.

**(b) Industry momentum + cross-industry momentum + reversals + size/BM — Fama–MacBeth hedged returns (Table V in the published version / Table VII–VIII in the WP).**
The dependent variable is the supplier's monthly return. Regressors: `CRET(t−1)`, `CRET(t−12, t−2)`, `RET(t−1)` (own reversal), `RET(t−12, t−2)` (own momentum), `INDRET(t−1)` (industry momentum, Moskowitz–Grinblatt), `INDRET(t−12, t−1)`, `CINDRET(t−1)` (customer-industry momentum, Menzly–Ozbas), `CINDRET(t−12, t−1)`, plus **Hou-style size-sorted industry portfolios** (P1 bottom 30% / P2 mid 40% / P3 top 30%, formed on end-of-June market cap, EW) for both own and customer industries, plus unreported size and book-to-market.

Coefficients on **`CRET(t−1)`** (in units of monthly %):

| | EW (1) baseline | EW (3) | EW (5) full controls | VW (6) baseline | VW (8) | VW (10) full controls |
|---|---|---|---|---|---|---|
| `CRET(t−1)` | **0.895\*** | 0.724\* | **0.730\*** | **1.170\*** | 1.178\* | **0.876\*** |
| | [4.03] | [3.01] | **[2.68]** | [3.57] | [3.26] | **[3.22]** |

Other coefficients (EW baseline): `RET(t−1)` = **−0.862\*** [−2.69]; `INDRET(t−1)` = **0.791\*** [3.04] → collapses to **0.098** [0.30] with full controls; `CINDRET(t−1)` = **1.407\*** [4.92]. 1-year `CRET(t−12,t−2)` = 0.529\* [2.88] EW (0.529\* [2.44] with controls), but **insignificant for VW** (−0.136, [−0.43]).

**Key bottom line, in the paper's own words (published p. 1997):**
> *"The results in Table VI give an unambiguous answer: Past customer returns forecast subsequent supplier stock returns. The effect is large, robust, and largely unrelated to other documented predictability effects. **Using the full set of controls and value-weighted portfolios, the average net effect in Table VI (after hedging) is around 88 basis points per month.**"*

So: **~155 bps gross → ~88 bps after hedging out size, BM, 1-month reversal, 12-month momentum, industry momentum, cross-industry momentum, and intra-industry size-based lead-lag.** That is roughly **57% survival**.

### Underreaction magnitude
- `URC = RET / (RET + CAR)` where CAR is the cumulative 6-month subsequent return. URC = 1 means no underreaction.
- **Suppliers: ~60% of the price gap closes in month t, the remaining ~40% closes over the next six months.** Supplier CAR = **2.8% (t = 3.74)**.
- **Customers: URC = 0.94, not statistically different from 1** — customer prices impound their own news essentially fully.
- By size: large firms cover **69%** in the initial month (31% left); small firms cover only **35%** (65% left).
- L/S cumulative return over the subsequent year = **4.73%**. Predictability persists for about a year.

---

## 1.5 STATED ECONOMIC MECHANISMS AND THEORETICAL BASIS

- **Primary theory: limited investor attention.**
  - **Kahneman (1973)** — *"Attention is a scarce cognitive resource and attention to one task necessarily requires a substitution of cognitive resources from other tasks."*
  - **Merton (1987)** — the first theoretical approach to segmented markets and investor inattention: investors obtain information on only a small number of stocks and trade only those, so less-followed stocks sell at a discount.
  - **Hong & Stein (1999)** — *"a model with multiple investor types, in which information diffuses slowly across markets and agents do not extract information from prices, generating return predictability."* ← **the core theoretical engine.**
  - **Hirshleifer & Teoh (2003)**; **Peng & Xiong (2006)** — additional inattention models.
- **Supporting empirical literature cited:** Huberman & Regev (2001) (re-release of old news moves price); Ramnath (2002) (intra-industry earnings-surprise spillover); Hong, Lim & Stein (2000) (gradual diffusion, especially of negative news); Hou & Moskowitz (2005) (price delay); DellaVigna & Pollet (2006).
- **Asymmetry:** *"customer momentum returns are asymmetric: the returns of the long short portfolio are largely driven by slow diffusion of negative news. This pattern is consistent with market frictions (such as short sale constraints) exacerbating the delayed response of stock prices to new information when bad news arrives."*
- **Why inattention, not risk:** none of the factor loadings on the L/S portfolio is significant; results survive DGTW characteristic adjustment and industry adjustment; results are stronger precisely where inattention should bind.

---

## 1.6 ROBUSTNESS TESTS — AND WHAT KILLS OR WEAKENS THE EFFECT

| Test | VW L/S | EW L/S | Verdict |
|---|---|---|---|
| Baseline 1-month | **1.578** [3.79] | **1.311** [4.93] | — |
| **Skip a week** | 1.464 [3.55] | 0.932 [3.28] | Survives (nonsynchronous trading ruled out) |
| **Skip a month** | 1.377 [3.16] | 1.046 [3.14] | Survives |
| **DGTW characteristic-adjusted** (125 ME/BM/mom portfolios) | 1.121 [3.23] | 0.839 [3.23] | **Weakens ~29%**, still significant |
| **Industry-adjusted** (48 FF industries) | **0.975 [2.89]** | **0.508 [2.14]** | **Weakens ~38% (VW) / ~61% (EW)**; EW only marginal at 5% |
| Smaller firms | 1.487 [3.95] | 1.071 [3.06] | Both legs strong |
| Larger firms | 1.475 [3.70] | 1.336 [4.21] | Strong |
| **1-year customer return, VW** | **~0.70 [1.85]** | — | **DIES — cannot reject zero** |
| 1-year customer return, EW | — | **1.13 [4.16]** | Strong |
| Sub-period **1981–1992** | **1.963 [4.39]** | 1.391 [4.28] | Strong |
| Sub-period **1993–2004** | **1.266 [1.99]** | 0.698 [1.66] | **Weakens sharply; EW insignificant** |
| Different-industry links | 1.157 [4.83] | 1.162 [2.84] | Strong |
| Same-industry links | 1.288 [2.49] | 1.192 [2.90] | Weaker t-stats |
| Positive-volume-every-day filter | little effect | little effect | Survives |
| Out-of-sample 1966–1980 (links proxied from first 1980–85 link) | holds | holds | *"informative but obviously imperfect"* |

**Inattention cross-section (COMOWN = # funds holding both customer and supplier / # funds holding the supplier), published Table VII, large caps above NYSE median with ≥20 mutual funds:**
- Low/zero common ownership (**high inattention**): **2.70% per month (t = 3.49), EW**
- High common ownership (**low inattention**): **0.61% per month (t = 1.05)** — **not different from zero**
- Spread (high-inattention minus low-inattention): **2.09% per month (t = 2.42)**
- (WP 2006 version, value-weighted, 30/40/30 independent sorts: **3.02% (t = 2.70)** vs **0.55% (t = 0.58)**; spread **2.48% (t = 1.98)**.)
- Footnote: *"the correlation between COMOWN and total mutual fund ownership is 7%"* — so COMOWN is not just a size/liquidity proxy.

### What KILLS the effect
1. **Value-weighted, 1-year-horizon sorts** → ~70 bps, statistically indistinguishable from zero.
2. **High common-ownership subsample** → 0.61% (t = 1.05), i.e. zero.
3. **The second half of the sample (1993–2004)** → VW t-stat falls to 1.99 and EW becomes insignificant (t = 1.66).

### What merely WEAKENS it
- DGTW characteristics adjustment (−29%).
- Industry adjustment (−38% VW / −61% EW) — the single largest legitimate control.
- Full Fama–MacBeth hedging: 155 → 88 bps (−43%), still t ≈ 3.2.

---

# PAPER 2 — Lee, Sun, Wang & Zhang (2019), *Technological Links and Predictable Returns*, JFE 132(3), 76–96

> **Version read: the 23-Oct-2017 working paper (61 pp).** See §0 for the caveat.

## 2.1 THE EXACT FACTOR CONSTRUCTION RECIPE

### Data and sample
- **Patent data:** Google Patents data of **Kogan, Papanikolaou, Seru & Stoffman (2017)** — **1.9 million CRSP-matched USPTO patents granted 1926–2010**, assignees mapped to CRSP PERMNO. Source: `https://iu.app.box.com/patents`.
- Sample = intersection of Google patent data, CRSP and Compustat. Common stocks only (**share codes 10, 11**). **Exclude financial firms (1-digit SIC = 6).**
- Require non-missing ME and SIC from CRSP, **non-negative book equity** at the end of the prior fiscal year.
- **Restrict to firms with at least one patent granted in the rolling past 5 years.**
- **Exclude stocks priced below $1** at the beginning of the holding period (robustness: <$5 or below the 10th NYSE ME percentile — unchanged).
- Shumway (1997) delisting-bias correction applied (no effect).
- **Accounting-to-return gap:** Google patent data for grant year *t* is matched to Compustat fiscal year *t*, then matched to CRSP returns **from July of year t+1 to June of year t+2** (Fama–French 1992 convention).
- **Final sample: 561,989 firm-month observations, July 1963 – June 2012 (588 months).** 189 firms in Jul-1963 → 1,363 in Jun-2012. Covers ~53% of the CRSP common-stock universe by market cap.

### The linkage measure — patent CLASS-distribution overlap (Jaffe 1986), **not** patent-text overlap

> ⚠️ **The brief described this as "patent-text-overlap links." That is not what the paper does.** The measure is the **uncentered correlation between the two firms' distributions of patents across 427 USPTO technology classes** — a *class-count* measure in the Jaffe (1986) tradition, following Bloom, Schankerman & Van Reenen (2013). The paper explicitly distinguishes itself from text-based approaches: *"BK apply textual analysis to patent documents to determine technological affinity, while our paper measures pairwise distance using patent class distribution (Jaffe, …)."*

**Technology closeness** (verbatim from the paper):

```
TECH_ij = (T_i · T_j') / [ (T_i · T_i')^(1/2) * (T_j · T_j')^(1/2) ]
```

where `T_i = (s_1, s_2, …, s_τ, …, s_427)` and `s_τ` is **the average share of the number of patents in USPTO technology class τ out of firm i's total number of patents over the rolling past five years**.

- **427 USPTO technology classes.**
- `TECH_ij ∈ [0, 1]`, **symmetric** (`TECH_ij = TECH_ji`).
- *"by construction the TECH essentially acts as a weight in calculating the average stock return of technology-linked firms and is biased toward firms more technologically close to the focal firm."*
- **TECH is computed at the end of each year t** using patent **grant dates** that are publicly available, then **mapped to return data from July of year t+1 to June of year t+2.**

**Technology-linked return:**

```
TECHRET_it = [ Σ_{j≠i} TECH_ij · RET_jk ] / [ Σ_{j≠i} TECH_ij ]
```

where **k ∈ {1, 3, 6, 12} = the number of past month returns of linked firm j; base specification k = 1.**

- **Signal type: a RAW RETURN**, weighted by an economic-proximity measure. **Not** a residual return, **not** a firm characteristic.
- **Weighting is NOT equal and NOT value.** This is a key design choice and it is tested: see §2.4.

### Sorting and portfolio formation
> *"At the beginning of each month, we sort all firms into deciles based on the return of their technology-linked portfolios in the previous month. The decile portfolios are then rebalanced at the beginning of each month to maintain either equal or value weights. The bottom lines show the returns of a zero-cost portfolio that hold the top 10% high technology-linked firm return stocks and sells short the bottom 10% low technology-linked firm return stocks."*

- **Sort: 10 DECILES**, univariate, on lagged `TECHRET`.
- **Both EW and VW** portfolios reported.
- **Holding period: 1 month, rebalanced monthly.**
- **L/S = top decile minus bottom decile.**
- Long-horizon tests: `(L, H)` strategies following Moskowitz–Grinblatt, L ∈ {1,3,6,12}, H ∈ {1,6,12,24,36}, rebalanced monthly.

### Pseudocode

```
# ---------- STEP 1: build firm technology vectors (annual, rolling 5y) ----------
for each firm i, at end of year t:
    counts = {}   # USPTO class -> number of patents GRANTED in (t-4 .. t)
    for each patent p granted to i in (t-4 .. t):
        counts[class(p)] += 1
    total = sum(counts.values())
    if total == 0: firm i not in sample this year
    T_i = vector of length 427
    for tau in 1..427:
        # "average share ... over the rolling past five years"
        T_i[tau] = mean over years y in (t-4..t) of ( patents_y(i, tau) / patents_y(i) )
        # == counts[tau] / total if you compute the 5-year pooled share

# ---------- STEP 2: pairwise technology closeness (annual matrix) ----------
for each ordered pair (i, j), i != j:
    TECH[i][j] = dot(T_i, T_j) / ( sqrt(dot(T_i,T_i)) * sqrt(dot(T_j,T_j)) )
    # uncentered correlation; in [0,1]; symmetric

# ---------- STEP 3: map to a return month ----------
# TECH computed at end of year t applies to return months Jul(t+1) .. Jun(t+2)

# ---------- STEP 4: technology-linked return for month t-1 ----------
for each firm i, month t:
    J = { j != i : TECH[i][j] > 0 }
    num = sum( TECH[i][j] * RET[j][t-1] for j in J )
    den = sum( TECH[i][j]          for j in J )
    TECHRET[i][t-1] = num / den
    # k=1 base. For k=3/6/12 use the k-month cumulative return of j ending t-1.

# ---------- STEP 5: sorts ----------
universe = { i : TECHRET[i][t-1] exists, PRICE[i][t-1] >= 1, SIC != 6x }
deciles D1..D10 on TECHRET[i][t-1] ascending
for d in 1..10:
    R_d[t] = mean(RET[i][t] for i in Dd)          # EW
    R_d_VW[t] = value_weighted_mean(...)          # VW
L S[t] = R_10[t] - R_1[t]

# ---------- STEP 6: alphas ----------
alpha = intercept of OLS( L S[t] ~ MKT + SMB + HML [+ MOM] )     # or FF5 [+ MOM]
```

### SIGN CONVENTION
**LONG the focal firm whose technology-linked firms performed BEST in the prior month; SHORT the focal firm whose technology-linked firms performed WORST.** Positive sign, "technology momentum," same convention as Cohen–Frazzini: the *linked* firms' returns forecast the *focal* firm's return.

### Network statistics (important for feasibility)
- **Average number of linked firms in the technology space: 280** per focal firm.
- **Mean pairwise TECH = 0.11, SD = 0.16** — *"indicating that the technological link is quite pervasive and sparse."*
- Diversification check: *"an average (median) patent technology class contains firms from 10 (10) different [2-digit SIC industries]"*; HHI of patents by industry within a tech class = 0.37 (2-digit) / 0.21 (4-digit) — i.e., **technology classes cut across industries**, which is why the effect is distinct from industry momentum.

---

## 2.2 HOW THE NETWORK IS BUILT AND WHY GRANT (NOT APPLICATION) DATES

- Two dates matter for a patent: the **application date** and the **grant date**; the lag averages **two to three years**.
- They use the **grant date** because *"the USPTO issues patents every Tuesday, and its publication, Official Gazette, lists detail information of the patents granted on that day. Thus the investors are able to get the patent information freely from the patent offices on the grant date."* Kogan et al. (2017) document abnormal turnover around the grant date.
- **AIPA caveat they state:** after the American Inventors Protection Act (effective 30 Nov 2000), applications are published 18 months after filing, so for patents filed after that date the market may know earlier (publication date or grant date, whichever is first). They nevertheless use the grant date throughout.

---

## 2.3 SLOW-DIFFUSION REQUIREMENT AND THE STATED MECHANISM

**Stated economic story:** *"sluggish price adjustment to more nuanced technological news."* The abstract's own framing.

Three posited drivers of the delayed reaction, quoted from Section 6:
> *"Ex ante, we posit that the magnitude of the delayed price reaction will be an increasing function of: **(a) the relevance of technology-linked firm news to the focal firm, (b) the extent to which investors are inattentive, and (c) the relative costs of arbitrage.**"*

On innovative firms specifically:
> *"compared with traditional firms, more innovative firms are harder to value and thus the information diffusion could be slower (Hirshleifer, Hsu, and Li, 2013); second, comparing a firm that takes innovation more seriously with a firm that barely does innovation, the former is more likely to be affected by innovation shocks and shocks to its technology-linked peers in particular. Thus, if our return effect is truly driven by investors' limited capacity and resources, combined with the valuation difficulty of more innovative firms, we would expect that the more innovative the firm, the more severe the lag."*

**Theoretical basis:** Hong & Stein (1999); Hirshleifer, Hsu & Li (2013); Hong, Lim & Stein (2000); Cohen & Frazzini (2008); Menzly & Ozbas (2010); Cohen & Lou (2012); and for the real-effects channel, Acemoglu, Carvalho, Ozdaglar & Tahbaz-Salehi (2012) (network propagation of shocks) and Bloom, Schankerman & Van Reenen (2013) (technology spillovers).

**No-reversal evidence (important — rules out overreaction):** in Figure 1 they track the hedge portfolio for 6 months and extend to 12 and 24 months: *"we also observe modest additional upward drift through month six… we see no reversal over the long-run, suggesting that we are capturing a mechanism of delayed updating of focal firm prices to reflect information important to their fundamental values."*

---

## 2.4 REPORTED MAGNITUDES — EXACT NUMBERS

### Baseline portfolio results (Table 2)

| Portfolio | Raw return | CAPM α | FF-3 α | FF-3+MOM α | FF-5 α | FF-5+MOM α |
|---|---|---|---|---|---|---|
| **Equal-weighted L/S** | **1.17% / mo (t = 5.47)** ≈ 14.1%/yr | 1.22% | 1.26% | **1.08%** | 1.37% | 1.21% |
| **Value-weighted L/S** | **0.69% / mo (t = 3.19)** ≈ 8.3%/yr | 0.74% | 0.80% | **0.65%** | 0.86% | 0.73% |

**Own-momentum control:** adding Carhart's momentum factor has *"the only negative but negligible effect"* — EW 1.26% → **1.08%** (loss ≈ 18 bps, ~14%); VW 0.80% → **0.65%** (loss ≈ 15 bps, ~19%). Still significant.

### Weighting-scheme horse race (this is the important design result)
WP footnote 20, Online Appendix Table A.1 Panel D:

| Scheme for constructing `TECHRET` | L/S raw return (EW portfolio) | L/S raw return (VW portfolio) |
|---|---|---|
| **TECH-weighted (baseline)** | **1.17%** | **0.69%** |
| Equal-weighted across linked firms | 1.07% | 0.64% |
| Value-weighted (by linked firms' market cap) | **0.39%** | **0.36%** |

> *"These portfolio returns are smaller than the TECH-weighted … baseline results in Table 2, supporting the value of technology closeness (TECH) weighting scheme."*

So the economic-proximity weighting contributes roughly **+10 bps over equal weighting** and roughly **+78 bps over value weighting**. The link *measure* is doing real work — it is not merely a sector proxy.

### Fama–MacBeth regressions (Table 3)
Dependent variable = focal firm return in month t. Key regressor = `TECHRET(t−1)`.

- **Column 1 (no controls): coefficient = 0.643, t = 4.18** → *"the average monthly return spread of the focal firms in the top and bottom technology-linked return deciles is 64.3 basis point."*
- **Column 2 (adding size, BM, gross profitability, asset growth, R&D intensity, REV = own t−1 return, MOM = own t−12..t−2):** *"The magnitude and significance of the coefficient of TECHRET(t−1) are almost the same as in column 1."*
- **Column 3 (adding `INDRET(t−1)`, the value-weighted own-industry return, following Cohen & Lou 2012 and Moskowitz–Grinblatt 1999):** *"Both the magnitude and the significance … are similarly pronounced … which indicates that the technology momentum effect we documented cannot be explained by the industry momentum effect."*
- **Column 4 — industry-adjusted dependent variable `RET(t) − INDRET(t)`:** *"the magnitude and significance of the coefficient for TECHRET(t−1) are virtually the same when we use industry-adjusted returns."* And, critically: **`INDRET(t−1)` becomes indistinguishable from zero**, which is the Cohen–Lou delayed-information signature. *"By subtracting the industry return from focal firm return, we purge out stock return continuation that arises from industry wide return auto-correlations."*
- **Table 4 controls for supplier and customer returns (Menzly–Ozbas), pseudo-conglomerate returns (Cohen–Lou), and stock turnover:** *"the magnitude and significance of the coefficient for TECHRET(t−1) are qualitatively the same as our main results, indicating that the information diffused along the technological link cannot be explained by the information shocks from supply-chain, business segment, or the illiquidity of stocks."*

### Horizon / formation-period results (Online Appendix Table A.4)
- **Most profitable strategy is (1,1)**: L=1 month formation, H=1 month holding.
- EW raw: (1,1) = **1.17%**; (12,1) = **1.11%** → *"the profitability of short-term strategy is less sensitive to the length of ranking period L."*
- DGTW-adjusted (1,1) = **0.79%**; industry-adjusted (1,1) = **0.99%**.
- EW raw (1,12) = **0.32% (t = 3.90)** — decays but remains significant.
- *"value-weighted returns are smaller than the equal-weighted returns in all the strategies, indicating that the speed of information diffusion is quicker in firms with big size."*
- *"the return predictability diminished to insignificant in the longer holding period, and the decay is much quicker for value-weighted portfolios."*

### Stickiness of technology closeness (Online Appendix Table A.3)
- corr(`TECHRET`, 1-year-lagged `TECHRET_L1`) = **0.843**; with 3-year-lagged `TECHRET_L3` = **0.610**.
- `TECHRET_L1` EW L/S = **88 bps/month (t = 4.22)** ≈ 10.6%/yr — still strong with year-old patent data.

---

## 2.5 STATED MECHANISMS AND CONDITIONING EVIDENCE

All interaction coefficients are from Fama–MacBeth regressions with the full control set (Table 5):

**(a) Innovation intensity — the effect should be stronger for more innovative firms:**
- `R&D dummy × TECHRET(t−1)` = **0.520 (t = 3.67)**
- `Patent dummy × TECHRET(t−1)` = **0.416 (t = 2.92)**
- *"the technology momentum effect for more innovative firms is over twice as large as that of other firms."*

**(b) Investor inattention — the effect should be stronger where attention is lower:**
Three dummies (all = 1 if above sample median):
- **size dummy × TECHRET** → **negative and significant**
- **analyst coverage dummy × TECHRET** → **negative and significant**
- **institutional ownership dummy × TECHRET** → **negative and significant**

**(c) Cost of arbitrage — the effect should be stronger where arbitrage is costly:**
- **IdioVol dummy × TECHRET** → **positive and significant**. `IdioVol` = SD of residuals from a regression of **daily** stock returns in the previous month on the FF-3 factors (≥10 daily returns required).
- **Bad News × TECHRET** = **0.948 (t = 2.10)**, where `Bad News` = 1 if `TECHRET(t−1)` is in the worst-performing 30% (Hong–Lim–Stein convention). Compare to the **unconditional coefficient of 0.643** from Table 3 — i.e. **the effect is ~48% larger on the downside.**

**(d) Real-effects validation (Section 7) — establishes that this is not pure risk:**
- **`TECHSUE` (technology-linked standardized unexpected earnings) predicts the focal firm's future `SUE`**, with firm and time fixed effects, controlling for up to 4 lags of own `SUE`.
- **Technology-linked patent counts (`TECHPNUM`) predict the focal firm's future patent applications; technology-linked citation counts (`TECHCNUM`) predict future citations** (with industry/firm fixed effects). Citation counts use Kogan et al. (2017) truncation adjustments.

---

## 2.6 ROBUSTNESS TESTS — AND WHAT KILLS OR WEAKENS THE EFFECT

| Test | Result |
|---|---|
| **Sub-periods 1963–79 / 1980–89 / 1990–99 / 2000–12** | `TECHRET(t−1)` **positive and significant in all four**. **Industry momentum `INDRET(t−1)` becomes insignificant in 2000–2012**, while technology momentum does not. |
| Require **≥2 or ≥3 years** with granted patents in the past 5 years | Robust |
| Exclude price < $5, or ME below the 10th NYSE percentile | Unchanged |
| Require `TECH > 0.01` or linked firms ranked in the **top 50 by TECH** | Unchanged |
| **Value-weighted portfolios throughout** | **Roughly halves the effect** (117 → 69 bps) |
| **Long holding periods (H = 12, 24, 36)** | **Decays; VW decays to insignificance much faster than EW** |
| Value-weighting the *linked-firm* returns inside `TECHRET` | **Guts the signal** (1.17% → 0.39% EW) |
| Industry adjustment of the dependent variable | Coefficient "virtually the same"; industry momentum itself becomes insignificant |
| Controls for supply-chain, pseudo-conglomerate, turnover | "Qualitatively the same" |

**What kills it:** nothing outright. **What weakens it materially:** (i) value weighting, (ii) long holding horizons, (iii) replacing TECH weights with market-cap weights. **The effect is concentrated in smaller, more innovative, less-covered, higher-idiosyncratic-volatility stocks.**

---

# PAPER 3 — Parsons, Sabbatucci & Titman (2020), *Geographic Lead-Lag Effects*, RFS 33(10), 4721–4770

> **Version read: the AEA 2020 conference version (65 pp).** See §0.

## 3.1 THE EXACT FACTOR CONSTRUCTION RECIPE

### Sample and data
- **Universe:** domestic common stocks (**CRSP share codes 10 and 11**) on NYSE, NASDAQ, AMEX.
- **Period: 1970–2013 = 527 months.**
- **Restriction: stocks headquartered in, or immediately proximate to, the TWENTY LARGEST URBAN CENTERS in the United States.**
- **Location:** the **zip code of the firm's headquarters from Compustat (`ADDZIP`)**, grouped into **Bureau of Economic Analysis Economic Areas (EAs)**. EAs *"are intended to capture local nodes of economic activity, and typically involve a main metropolitan area, along with smaller surrounding regions from which workers may commute."* Examples given: San Jose–San Francisco–Oakland (CA); Atlanta–Sandy Springs–Gainesville (GA-AL); Houston–Baytown–Huntsville (TX). Definition reference: `http://www.bea.gov/newsreleases/regional/rea/rea1104.htm`.
- **Industry:** each firm is linked **monthly** to a single **Fama–French 12 industry** (non-durables, durables, manufacturing, energy, chemicals, business equipment, telecommunications, utilities, shops, healthcare, finance, other). Deliberately **broad**: *"We intentionally select such relatively broad groupings in order to reduce the extent of overlap between firms classified in different industries."*
- **Minimum group size: at least 10 firms** required to form a lead (predictor) portfolio — *"the same threshold required throughout the paper."*

### The signal — TWO portfolios, and the crucial distinction
The benchmark regression (their equation 4):

```
r_(i,c,j,t+1) = α + β1 · r_(c,∉j,t) + β2 · r_(∉c,j,t) + β3 · X_t + ε
```

- **`r_(c,∉j,t)` — the AREA portfolio (the paper's contribution):** the **equally-weighted lagged return of firms headquartered in the same city c but operating OUTSIDE firm i's industry (∉j).**
- **`r_(∉c,j,t)` — the INDUSTRY portfolio (the known control):** the **equally-weighted lagged return of firms in the same industry j but OUTSIDE city c.**
- **`X_t` controls:** firm's own 1-month lagged return, 12-month momentum, firm size, book-to-market, trading volume, gross profitability, asset growth, and institutional ownership (following Harvey–Liu–Zhu 2016 and Novy-Marx 2013).

> *"The first is r_(c,∉j,t), the equally-weighted, lagged return of firms headquartered in city c, but operating outside firm i's industry (∉j). Coefficient β1 thus estimates the lead-lag effect within cities, but across industrial sectors."*

**Worked example given in the paper:** *"if the dependent variable is the July 2007 return of Coca-Cola (NYSE:KO), the city portfolio would include the June 2007 returns of such Atlanta peers as Home-Depot, and the industry portfolio would include the June 2007 return of non-local bottlers such as Pepsi-Cola, headquartered in New York City."*

### Sorting and portfolio formation
> *"Every month, we rank each firm i not by its own lagged return (as we would in a simple momentum strategy), but by r_(c,∉j,t), the average lagged return of firms headquartered in the same region, but operating in different sectors. We use a one month horizon both for the sorting criterion (i.e., area-level stock returns are measured over a month) as well as the holding period (i.e., portfolios are reformed at the end of every month). **Based on these rankings, we form value-weighted portfolios.**"*

- **Sort: 5 quintiles** (deciles also reported in Appendix Table A2, "nearly identical results").
- **⚠️ IMPORTANT ASYMMETRY: the SIGNAL portfolio is EQUALLY-WEIGHTED, but the TRADED portfolio is VALUE-WEIGHTED.** This is the opposite of what a naive reading would assume, and it matters — see §3.6.
- **Holding period: 1 month. Reformed at the end of every month.**
- **L/S = top 20% (highest lagged area return) minus bottom 20%.**

### Pseudocode

```
# ---------- STEP 1: firm -> (city, industry) ----------
for each firm i, each month t:
    zip     = Compustat ADDZIP(i)                 # STATIC: only the CURRENT HQ zip
    city_c  = BEA_EconomicArea(zip)               # ~179 EAs; sample keeps the 20 largest
    j       = FamaFrench12( SIC(i, t) )           # assigned monthly

# ---------- STEP 2: the AREA portfolio return (EQUALLY weighted) ----------
for each firm i in (city c, industry j), at month t:
    peers = { k : city(k) == c  AND  industry(k) != j  AND  k != i }
    if count(peers) < 10: skip i                  # min-10-firm rule
    AREA_RET[i, t] = mean( RET[k, t] for k in peers )        # EQUAL weights

# ---------- STEP 3: the INDUSTRY portfolio return (EQUALLY weighted) ----------
    ind_peers = { k : industry(k) == j  AND  city(k) != c }
    IND_RET[i, t] = mean( RET[k, t] for k in ind_peers )     # EQUAL weights

# ---------- STEP 4: predictive regression (the "lead-lag" object) ----------
run Fama-MacBeth monthly cross-sectional OLS:
    RET[i, t+1] ~ AREA_RET[i, t] + IND_RET[i, t] + X[i, t]
    # report time-series mean of coefficients and FM t-statistics

# ---------- STEP 5: the TRADING strategy (VALUE weighted portfolios) ----------
at end of each month t:
    rank all firms ascending on AREA_RET[i, t]
    assign to quintiles Q1..Q5  (Q1 = lowest area return)
    for q in 1..5:
        R_q[t+1] = VALUE_weighted_mean( RET[i, t+1] for i in Qq )
    L S[t+1] = R_5[t+1] - R_1[t+1]        # 42 bps/month raw (t = 3.65)

# ---------- STEP 6: alphas ----------
alpha = intercept of OLS( L S[t] ~ MKT + SMB + HML [+ MOM] )      *  and *  ~ FF5
```

### SIGN CONVENTION
**LONG firms whose same-city, DIFFERENT-industry neighbours had HIGH past-month returns; SHORT firms whose same-city, different-industry neighbours had LOW past-month returns.** Note this is *"co-headquartered firms operating in different sectors"* — the same-industry city peers are deliberately excluded, otherwise you would just be re-measuring industry momentum.

---

## 3.2 HOW THE NETWORK IS BUILT — AND THE MEASUREMENT-ERROR PROBLEM

**Source:** Compustat `ADDZIP` → BEA Economic Areas. Industry from CRSP/Compustat SIC → FF-12.

**The look-ahead problem, stated plainly by the authors:**
> *"Because COMPUSTAT lists only the zip code of the firm's **current** headquarters, we will misclassify firms that have relocated, such as Boeing, which moved its HQ from Seattle to Chicago in 2001. Though this introduces measurement error into our analysis, **this works against us, i.e., the effects we estimate will be closer to zero than they would be absent headquarter misclassification.**"*

Example: General Dynamics moved St. Louis → Washington DC in 1992, but `ADDZIP` = 22042 (Falls Church, VA) for **all** years, before and after.

**Quantified error rate:** Pirinsky & Wang (2006) estimate 2–3% of firms moved over 1992–1997 (~0.5%/yr). Over 40 years, ~80.6% still correctly classified; because the reported location is "correct" on average half the time, the implied **firm-year error rate is ~9.7%**, and they argue this is conservative because the panel over-weights large, long-lived firms that move less.

**Extending beyond headquarters (Section 5.2).** Because a single HQ measure ignores dispersed operations (Wal-Mart, Home Depot vs. DTE Energy), they adopt **Garcia & Norli (2012)**: a text-parsing count of **the number of unique state names mentioned in the annual report** (1994–2008), time-series averaged per firm and extrapolated to all years. Firms below the median mention **nine** states on average; above the median, **three**. Cutoff = **5.46 states**.

---

## 3.3 SLOW-DIFFUSION REQUIREMENT AND THE STATED MECHANISM

**The mechanism is analyst SPECIALIZATION and OVERLAP — this is the paper's central theoretical contribution.**

> *"The key concept of the model in Section 3 is that when two firms are simultaneously covered by many of the same analysts, their stock prices are more likely to react synchronously to common information shocks, and consequently, less likely to exhibit lead-lag relationships. Importantly, what the model emphasizes it is not analyst coverage per se that matter; **rather, the lead-lag relationship is determined by the extent to which analysts overlap between two (or more) firms. Indeed, this concept of analyst overlap is the main contribution of the paper, both theoretically and empirically.**"*

**The economic logic:** sell-side analysts specialize **by industry**, **not by geography**. Therefore industry shocks reach all industry peers' prices simultaneously, but a city-wide shock reaches only the firms whose analysts happen to be looking at that city — which is almost nobody.

> *"analysts tend to specialize by industry, which means that a firm covered by lots of analysts will almost certainly also have substantial overlap in coverage with its industry peers. However, if our focus is on lead-lags between geographic peers, this is not the case, because analysts do not tend to focus on geographic segments."*

**Quantified incidence of zero analyst overlap** (IBES Recommendations History, from 1993; excluding single-firm analysts and firm-years with no coverage):

| Analyst following | **Zero overlap with same-CITY, different-industry peers** | **Zero overlap with same-INDUSTRY, different-city peers** |
|---|---|---|
| Full sample | **57.93%** | **7.39%** |
| ≤3 analysts | >66% | 12.67% |
| 4–9 analysts | 49.37% | 2.32% |
| ≥10 analysts | (ratio ~80× the industry figure) | — |

Network view: firms are connected via analyst overlap to only **2** geographic peers (different industries) on average, but to **over 15** industry peers (different cities). For firms with ≥10 analysts, ~5.55 geographic overlaps vs. ~49 industry overlaps.

**Formal model (Section 3).** Two-firm limited-information model with parameters `p1` (probability an *industry* shock is recognized and impounded) and `p3` (probability an *area* shock is), with `x, y, z` the per-analyst probabilities of covering the various firm pairs, and `N` the number of analysts:
- `p1` rises steeply with `N`; e.g. if `y = .15`, `p1(10) = 56%` and `p1(20) = 96%`.
- **`∂p3(N)/∂N = −log(1−x−z)·N·(1−x−z)^N ≈ 0`.** *"The lack of sensitivity to N implies that geographical lead-lags may remain significant, even for firms covered by a large number of analysts."* E.g. if `x+z = .01`, then with five analysts `p3 < 5%`, and with ten analysts `p3 < 10%`.

**Underlying assumption validated:** geographic co-movement in **fundamentals**, not just returns. Table 2 (monthly): sensitivity of a firm's return to the area portfolio = **0.23 (t = 16.59)**; to the industry portfolio = **0.927 (t = 88.04)**. At the annual level, local sensitivities are ~23–31% of industry sensitivities across ΔEPS, sales, employees and external capital. *(They note this contradicts Pirinsky & Wang's finding of negatively correlated local earnings changes; they attribute it to PW's firm-by-firm time-series approach being outlier-driven.)*

**Theoretical basis:** Hong & Stein (1999); Hong, Torous & Valkanov (2007); Merton (1987); Pirinsky & Wang (2006) (local return comovement); Korniotis & Kumar (2013).

---

## 3.4 REPORTED MAGNITUDES — EXACT NUMBERS

### Fama–MacBeth predictive regressions (Table 4, Panel A, 1-month predictors, full sample 1970–2013)
- **β1 (city / area) = 0.061** → *"six basis points (t=5.11) for the same change in a firm's lagged city portfolio."* (i.e. a 1% increase in the lagged city portfolio → +6 bps next month)
- **β2 (industry) = 0.243** → *"twenty-four basis points the following month (t=11.71)"*
- *"with lead-lags within cities being around one-fourth as strong as those within industry groups."*

**Panel B (city-industry-month portfolios, ~100,000 obs):** industry lead-lags ~15% smaller, **area lead-lags ~50% larger**; geographic coefficients end up one-third to one-half the size of industry coefficients.

### Trading profits (Table 5, value-weighted quintiles)
| | Raw monthly return | CAPM α | FF-3 α | FF-5 α | Sharpe |
|---|---|---|---|---|---|
| **Q1 (lowest lagged area return)** | 0.74% | **−0.26% (t = −3.23)** | −0.24% (t = −3.01) | −0.20% (t = −2.33) | ~0.2 |
| Middle three | ~market | small, insignificant | — | — | ~market |
| **Q5 (highest lagged area return)** | **1.16%** | t = 3.20 | — | t = 2.71 | **0.53** |
| **L/S (Q5 − Q1)** | **0.42% (42 bps, t = 3.65)** | **0.47% (t = 4.16)** | — | **0.42% (t = 3.32)** | — |

> *"the difference in raw returns between the first and fifth quintiles (42 basis points, t = 3.65) is nearly identical to the intercept estimated from either a regression against the market (47 basis points, t = 4.16) or against the FF-5 factors (42 basis points, t = 3.32)."*

- ≈ **5.0% per year**; the published abstract describes **"risk-adjusted returns of 5%–6% annually"** — consistent.
- **Adding Carhart momentum "likewise makes no difference."**
- **Deciles (Appendix Table A2): nearly identical results.**
- **Largest 20% of firms only (Appendix Table A3): almost identical results.** ← *This is a distinguishing feature: unlike Cohen–Frazzini and Lee et al., the geographic effect does NOT live only in small caps.*
- **Portfolio characteristics show no trends** in firm-specific volatility, size, or book-to-market that could explain the return pattern; size is hump-shaped with the largest average market cap in the middle group.

---

## 3.5 STATED MECHANISMS AND THEORETICAL BASIS

- **Primary:** analyst specialization + informational segmentation of markets. Information producers are organized along **industry** lines, so **geographic** common shocks are not simultaneously impounded.
- **Path:** analyst overlap → synchronous price reaction; lack of overlap → lead-lag.
- **Theoretical basis:** Hong & Stein (1999) gradual information diffusion; Hong, Torous & Valkanov (2007) (industries lead the market); Merton (1987) (investor recognition); Pirinsky & Wang (2006) (local comovement); Moretti (2012) (urban knowledge spillovers as the reason headquarters is the right location proxy).
- *"One contribution of our study is to provide an institutional rationale for limited attention, i.e., drawing on the structure of the investment analyst business to motivate why analysts and investors organize their information gathering efforts as they do."*

---

## 3.6 ROBUSTNESS TESTS — AND WHAT KILLS OR WEAKENS THE EFFECT

| Test | Result | Verdict |
|---|---|---|
| **Panel regressions with date FE, double-clustered by firm and date** (Table A5 Panel A) | Area coefficient **0.081 (t = 4.87)** vs FM 0.061 (t = 5.11) — **STRONGER**. **Industry momentum weakens** — profitable only within 3 months. Firm-clustering is irrelevant; **clustering by time drives the difference.** | Geography survives and strengthens |
| **Skip a month** (Table A5 Panel B) | Area 0.061 → **0.040 (t = 3.71)**; industry 0.243 → **0.136 (t = 6.03)** | Survives, but ~35% weaker |
| **Value-weighted city and industry portfolios** (Table A5 Panel C) | *"the magnitudes are **40-50% smaller**, depending on the horizon"* — but still significant | **Materially weakens** |
| **Controlling for other lead-lag sources** — extra portfolios formed on size, analyst coverage, trading volume, institutional ownership (quintile + a pooled "higher-ranked" portfolio = 8 controls), Table A6 | **STRONGER**: 0.061 → **0.079 (p < 0.01)** | Survives and strengthens |
| **Excluding RHS firms that are larger / more-covered than the LHS firm** (Tables A7, A8) | Still statistically significant, even when **every** forecasting firm is smaller and less-covered | Survives — rules out size/coverage lead-lag |
| **Controlling for customer/supplier lagged returns (Cohen–Frazzini)** (Table A9) | Costs **>98% of firm-month observations** | Sample-destroying |
| **Headquarter misclassification** (Table A4): scramble 1% / 5% / 10% / 20% / 50% of locations | 1% and 5%: trivial. 10%: slight. **20%: magnitude cut by one-third** (still significant full-sample). **50%: result vanishes entirely.** | 20%+ is fatal; ~5–10% is the realistic rate |
| **Analyst-coverage stratification** (Table 6 Panel A) | **Industry lead-lag collapses with coverage** (0.28, t=8.79 at 0 analysts → 0.14, t=4.64 at 5–9 → **0.098** at 10+; Q1 vs Q4 difference rejected at 1%). **Geographic lead-lag is FLAT** (0.060 t=3.27; 0.090 t=4.12; 0.060 t=2.76; **p = 0.877** for the Q1−Q4 difference) | **A key differentiator** — geography does **not** weaken in high-coverage stocks |
| **Size / volume stratification** (Panels B, C) | Industry lead-lags weaken by ~1/3 (volume) to ~2/3 (size); **geographic lead-lags show no clear relation** (one exception: p = 0.03 that the smallest quartile has higher profits; with volume, geographic lead-lags are **largest** in the most heavily traded quartile) | Survives |
| **Sub-periods 1970–1990 vs 1991–2013** | *"lead-lags between geographic peers appear to have **weakened somewhat over the last two decades**, whereas industry momentum has maintained relatively constant."* | **Materially weakens in recent decades** |

### What KILLS the effect
- **Misclassifying ≥50% of headquarter locations** (not a real-world concern).
- Effectively: **controlling for customer/supplier links** destroys the sample rather than the effect.

### What materially WEAKENS it
1. **Value-weighting the SIGNAL portfolios** (−40 to −50%). *The authors' own explanation:* *"If the goal is to measure local economic fundamentals using portfolio returns, an equally-weighted basket is more likely to be informative, compared to one that puts disproportionate weight on a few large firms (e.g., Dallas's ExxonMobil, Seattle's Amazon, etc.), especially given that they are less likely to be regionally concentrated."*
2. **The post-1990 sub-period** — geographic lead-lags have decayed.
3. **Skipping a month** (−35%).
4. **20% HQ misclassification** (−33%).

---

# PAPER 4 — Menzly & Ozbas (2010), *Market Segmentation and Cross-Predictability of Returns*, JF 65(4), 1555–1580

> Full main text obtained (65-page SSRN "JF forthcoming" version). See §0 for the correction about the link definition.

## 4.1 THE EXACT FACTOR CONSTRUCTION RECIPE

### Sample
- **CRSP monthly returns, July 1963 – June 2005.**
- NYSE, AMEX, Nasdaq. **Exclude** closed-end funds, REITs, ADRs, foreign companies, primes and scores.
- Financial statement data from the Merged Compustat–CRSP database.
- I/B/E/S (detail + summary history) for analyst coverage; **Thomson Financial 13F Holdings** for institutional ownership.

### The linkage network — BEA Benchmark Input-Output Use Table
> *"We use a series of Benchmark Input-Output Surveys of the Bureau of Economic Analysis to identify supplier and customer industries for a given stock or industry… BEA Surveys provide a detailed picture of the interdependent structure of the U.S. economy by assigning gross output to industry accounts and by reporting the amount of **inter-industry flow of goods and services in the Use Table**."*

- **11 surveys used on a rolling basis:** 2002, 1997, 1992, 1987, 1982, 1977, 1972, 1967, 1963, 1958, 1947.
- **Two different timing conventions — this is a subtle but important implementation detail:**
  - **For the "underlying economics" tests:** use a given survey from its survey year until the next survey. *"we use data from the 1963 Survey between 1963 and 1966, the 1967 Survey between 1967 and 1971, and so on."*
  - **For the TRADING STRATEGIES (look-ahead control):** *"we **delay using any data from a given survey until the end of the year in which the survey is publicly released**."* Release years: 2007, 2002, 1997, 1994, 1991, 1984, 1979, 1974, 1969, 1964, 1952.
- **Industry-account mergers:**
  - 1997 and 2002 surveys (NAICS): merge **2301 (new nonresidential construction) + 2302 (new residential construction) + 2303 (maintenance and repair construction)** into a single account.
  - Pre-1997 surveys (SIC): merge pairs **1-2** (livestock & agricultural products), **5-6** (metallic ores mining), **9-10** (nonmetallic minerals mining), **11-12** (construction), **20-21** (lumber and wood products), **33-34** (footwear, leather & leather products).
  - **Drop** miscellaneous accounts related to government, import and inventory adjustments.
- **Firm → industry mapping:** the firm's reported **SIC or NAICS code in Compustat** (falling back to CRSP if missing). BEA publishes SIC and NAICS dictionaries (their Table IA.VIII). **Applies to multi-segment firms too** — *"the reported SIC or NAICS code in COMPUSTAT, which has the desired property of representing the firm's main business and thus its main economic exposure."*
- **Industry returns are VALUE-WEIGHTED monthly.**
- **Inclusion rule:** a stock must have a market capitalization at the **end of June** of a given year to be included for the subsequent 12 months — *"primarily to stabilize the composition of industry portfolios."*

### The related-industry portfolios (the signal)
> *"After calculating industry portfolio returns, we further form **two separate portfolios for each industry**, one composed of supplier industries and another composed of customer industries. In calculating monthly returns for these portfolios, **we use data on the flow of goods and services to and from the industries in question as portfolio weights**. These portfolio weights are **shares of an industry's total purchases from other industries in calculating supplier industry returns**, and **its total sales to other industries in calculating customer industry returns**."*

So for industry *i*:
```
r_supplier(i, t) = Σ_k  w^S_ik · R_k(t)     where w^S_ik = (i's purchases from k) / (i's total purchases)
r_customer(i, t) = Σ_k  w^C_ik · R_k(t)     where w^C_ik = (i's sales to k)     / (i's total sales)
r_composite(i, t) = 0.5 · r_supplier(i, t) + 0.5 · r_customer(i, t)
```
> *"this portfolio weighting scheme has the desired properties of (i) identifying the set of economically related supplier and customer industries for a given industry, and (ii) reflecting the relative economic importance of related industries as proxied by the amount of inter-industry trade."*

**Signal type: a RAW RETURN**, aggregated with **input-output flow weights**. Not a residual, not a characteristic.

### Sorting and portfolio formation (the trading strategy)
> *"At the beginning of each month, we sort **industries** into five bins based on returns in supplier and customer industries in the previous month. We allocate industries with previous month related industry returns in the bottom quintile to the first bin, industries with previous month related industry returns in the second quintile to the second bin, and so on. After sorting industries in this fashion, we form **value-weighted portfolios for each of the five bins**, and calculate returns on these portfolios for the ensuing **one-month** period. Self-financing trading strategies consist of buying the high portfolio (industries with previous month related industry returns in the top quintile) and selling the low portfolio (industries with previous month related industry returns in the bottom quintile)."*

- **Unit of analysis: INDUSTRIES, not stocks.** *"we exclusively report on trading strategies that involve the buying and selling of industries as opposed to trading strategies that involve the buying and selling of individual stocks. The use of industries (which are value-weighted portfolios of stocks) should address potential concerns that thin markets could be driving the results or that transactions costs could render the trading strategies unattractive."*
- **Sort: 5 quintiles of INDUSTRIES** on the prior-month related-industry return.
- **Weights: VALUE-WEIGHTED within each bin.**
- **Holding period: 1 month, reformed monthly.**
- **L/S = top-quintile industry portfolio minus bottom-quintile industry portfolio.**
- Three separate strategies: sorted on `r_supplier`, on `r_customer`, and on `r_composite`.
- **Long-horizon variant** (Internet Appendix Table IA.VI): *J/K* strategies — sort industries into quintiles on **previous J-month** related-industry returns, hold for **K months**, rebalanced monthly so that K overlapping portfolios are held.

### The stock-level Fama–MacBeth regression (the other main test)
```
r_(i,t) = α_t + λ^supplier_t · r^supplier_(i,t−1) + λ^customer_t · r^customer_(i,t−1) + Λ_t · Z_(i,t−1) + e_(i,t)
```
- `Z` controls: `r_(i,t−1)` (**short-term reversal**), `r_(i,t−2:t−12)/11` (**medium-term continuation**, divided by 11 "to maintain comparability"), and `r^industry_(i,t−1)` (**industry momentum**, Moskowitz–Grinblatt).
- *"Further including estimated return betas such as market, HML and SMB to control for risk as well as firm characteristics such as book-to-market and size do not affect the results, and so we omit them."*
- Runs also at the **industry** level (same form, only `r_(i,t−1)` as a control).
- Heteroskedasticity/autocorrelation-robust standard errors; in the Internet Appendix, year-quarter clustered standard errors and panel regressions with monthly fixed effects.

### Pseudocode

```
# ---------- STEP 1: build the industry linkage weights (BEA I-O Use Table) ----------
# Use Table: rows = commodities/industries (sources), columns = industries (users)
for each BEA survey year Y:
    USE = Use_Table(Y)            # USE[s][u] = flow of goods/services from s to u
    for each industry u:
        total_purchases(u) = Σ_s USE[s][u]
        total_sales(u)     = Σ_v USE[u][v]
        w_supplier[u][s] = USE[s][u] / total_purchases(u)   # who u BUYS FROM
        w_customer[u][v] = USE[u][v] / total_sales(u)       # who u SELLS TO
    # merge accounts per the SIC/NAICS rules above; drop government/import/inventory adj.

# effective dates:
#   economics tests : survey Y applies from year Y to year Y+1's survey
#   trading strategy: survey Y applies only from end of the year Y is RELEASED

# ---------- STEP 2: firm -> BEA industry; industry returns ----------
for each firm i, month t:
    bea_ind[i] = BEAdict( Compustat SIC_or_NAICS(i) )     # or CRSP if missing
require ME(i, end of June of year y) to include i for Jul(y) .. Jun(y+1)
for each industry k, month t:
    R[k][t] = VALUE_weighted_mean( RET[i][t] for i with bea_ind[i] == k )

# ---------- STEP 3: related-industry portfolios (the signal) ----------
for each industry u, month t-1:
    r_supplier[u] = Σ_s w_supplier[u][s] * R[s][t-1]
    r_customer[u] = Σ_v w_customer[u][v] * R[v][t-1]
    r_composite[u] = 0.5 * r_supplier[u] + 0.5 * r_customer[u]

# ---------- STEP 4: sort INDUSTRIES, form VW portfolios ----------
for each month t:
    for X in {supplier, customer, composite}:
        rank industries ascending on r_X[.][t-1] -> quintiles Q1..Q5
        for q in 1..5:
            R_q[t] = VALUE_weighted_mean( RET[i][t] for i in industries in Qq )
        LS_X[t] = R_5[t] - R_1[t]

# ---------- STEP 5: alphas ----------
alpha = intercept of OLS( LS_X[t] ~ MKT + SMB + HML + MOM )
```

### SIGN CONVENTION
**LONG the industry whose related (supplier and/or customer) industries had HIGH past-month returns; SHORT the industry whose related industries had LOW past-month returns.** Same positive cross-predictability convention as the other three papers. Because they always include **both** supplier and customer returns in the stock-level regression (and only omit one at a time in the Internet Appendix), the direction here is **symmetric by construction** — this is a *mutual* cross-predictability, not a directed lead-lag.

**However, magnitudes are asymmetric:** supplier effects are consistently **larger** than customer effects (0.114 vs 0.071 at the stock level; 0.113 vs 0.075 at the industry level). The paper's own reading: *"The weaker customer industry cross-predictability effects suggest that informed investors impound customer industry signals more efficiently into stock prices than signals from supplier industries."*

---

## 4.2 HOW THE NETWORK IS BUILT, AND WHY THEY PREFERRED IT TO COMPUSTAT CUSTOMER LINKS

**Source:** BEA Benchmark Input-Output Surveys, Use Table, inter-industry flows of goods and services. Freely available: `http://www.bea.gov/industry/index.htm#benchmark_io` (accessed 5 June 2009). Table IA.VII lists the industries in the 1987 survey; Table IA.VIII is the SIC↔BEA industry dictionary.

**Why they chose BEA over the Compustat customer database (a direct, quotable critique that matters for replication):**
> *"However, for the purposes of the present paper, we have ultimately decided for two reasons that the BEA Surveys represent the best empirical approach… **First, only the smallest stocks exhibit cross-predictability effects based on lagged COMPUSTAT customer returns.** This raises natural questions as to whether return cross-predictability is an economically important phenomenon worthy of study if it is limited to small stocks. **Second, using the customer information database to identify supplier firms (in effect, by using reported relationships in the reverse direction) and then testing for cross-predictability effects from supplier firms yield no significant results. This is because the customer firms reported in COMPUSTAT are typically much larger than the reporting firm.** In comparison, the BEA Surveys enable us to identify for each firm broad portfolios of supplier and customer firms whose returns contain economically important information regardless of the size of the firm in question. As a result, cross-predictability effects based on the BEA Surveys are robust even to the exclusion of small stocks."*

**⚠️ Note the terminology clash between papers:** Cohen & Frazzini's *main* result (customer → supplier) and Menzly & Ozbas's statement that "supplier → customer yields no significant results" are not contradictory — Mozbas are describing the *inverted Compustat link* approach, where the resulting "supplier" set is a biased, incomplete sample of much smaller firms. Cohen & Frazzini's Appendix A1 finds a supplier→customer effect only *after* conditioning on `IMPSUPP` (important suppliers). **Any A-share replication must not build a "supplier" set by inverting a customer-disclosure table.**

---

## 4.3 SLOW-DIFFUSION REQUIREMENT AND THE STATED MECHANISM

**Central hypothesis (abstract):**
> *"We present evidence supporting the hypothesis that due to **investor specialization and market segmentation, value-relevant information diffuses gradually in financial markets**."*

**The three modeling ingredients they enumerate (p. 1–2):**
> *"(i) information about fundamentals is dispersed (Hayek, 1945), (ii) at least some investors can process only a subset of publicly available information (Hong and Stein, 1999) because either they have limited information-processing capabilities (Simon, 1955, Jensen and Meckling, 1992) or searching over all possible forecasting models using publicly available information itself is costly (Hong, Stein and Yu, 2007), and (iii) there are limits to arbitrage (De Long, Shleifer, Summers and Waldmann, 1990, Shleifer and Summers, 1990, Shleifer and Vishny, 1997)."*

**The failure mechanism:**
> *"new informative signals get incorporated into prices partially because at least some investors do not adjust their demand by performing the **Grossman-Stiglitz (1980) rational expectations inference** of recovering informative signals from observed prices. As a result of this failure on the part of some investors, returns exhibit predictability."*

**Two additional assumptions needed specifically for CROSS-predictability:**
> *"(A1) firms in different industries or segments of the market have **correlated fundamentals**, and (A2) as in Hong, Torous and Valkanov (2007), **investors specialize along these boundaries to some degree** which renders markets informationally segmented."*

**Crucial conceptual distinction they draw (relevant to whether you should call this "lead-lag"):**
> *"note that this spillover effect is different from a lead-lag effect because **either market can cross-predict the other when there is a new signal, rather than one market always leading or always lagging another**."*

**Three testable predictions of the model:**
1. Cross-predictability across assets with correlated fundamentals, **of the same sign as the correlation of fundamentals**.
2. The magnitude of cross-predictability should be **lower where there are more informed investors**.
3. Informed investors' trades in fundamentally related markets should be **correlated** — *"while information acquisition activities may involve specialization, trading need not."*

**Theoretical basis:** Hong & Stein (1999) (dispersed information); **Hong, Torous & Valkanov (2007)** (markets with correlated fundamentals — the model is a formal extension of HTV in two directions: *(i)* introducing uninformed investors, *(ii)* relaxing the assumption that informed investors invest only in the market about which they acquire signals); Hong & Stein (2007) (disagreement models); Grossman & Stiglitz (1980).

---

## 4.4 REPORTED MAGNITUDES — EXACT NUMBERS

### Preliminary evidence: fundamentals ARE correlated along the supply chain (Table I)
```
ROA_it = α_i + θ_market·ROA_market_t + θ_supplier·ROA_supplier_it + θ_customer·ROA_customer_it + e_it
```
- **Firm level (col 1): θ_supplier = 0.471, θ_customer = 0.465** — both statistically significant.
- **Industry level (col 2): θ_supplier = 0.398, θ_customer = 0.164.**
- ROA = (earnings before extraordinary items + depreciation & amortization) / total assets (Compustat items 18 + 14, scaled by item 6). Industry/market ROA aggregated with **firm assets as portfolio weights**. Double-clustered SEs (by firm and year; by industry and year).

### Evidence of specialization (Figure 1)
- **Median analyst covers 6.60 stocks per month**, vs ~6,036 stocks that satisfy the sample requirements and could hypothetically be covered.
- **Median analyst covers 1.91 industries.**
- **Median analyst's coverage in the covered firm's SUPPLIER industries accounts for only 8.64% of that firm's inputs; CUSTOMER industries, 8.73%.**

### Stock-level Fama–MacBeth cross-predictability (Table II)
| Specification | λ_supplier | λ_customer | own r_(t−1) | r_(t−2:t−12) | r^industry_(t−1) |
|---|---|---|---|---|---|
| **(1) All stocks, stock-level** | **0.114** | **0.071** | **−0.062** | 0.072 | 0.134 |
| (2) Excl. ME < 20th NYSE pct | 0.105 | 0.058 | −0.047 | 0.099 | 0.117 |
| (3) Excl. months w/o closing price | 0.117 | 0.059 | — | — | — |
| **(4) Industry-level** | **0.113** | **0.075** | 0.032 | — | — |

> *"the combined premium on the two portfolios are comparable to known medium-term continuation effects as represented by the coefficients on r_(i,t−2:t−12) (0.072) and r^industry_(i,t−1) (0.134)."*

### Informed-investor moderation (Table III; mirrored in Internet Appendix Table IA.II Panel A)
Regression: `r_(i,t) = α_t + Σ_{j=1..5} λ^j_t · A^j_(t−1) · r^composite_(i,t−1) + e_(i,t)`, where `A^j` are coverage/ownership quintile dummies and `r^composite` = average of `r^supplier` and `r^customer`. Stocks below the 20th NYSE percentile are excluded.

Quintile interaction coefficients (three columns: analyst coverage measured over trailing 12 months; analyst coverage in the current month; institutional ownership %):

| Quintile | (1) | (2) | (3) |
|---|---|---|---|
| **Q1 (LOWEST coverage/ownership)** | **0.287\*\*** | **0.277\*\*** | **0.329\*\*\*** |
| Q2 | 0.246\*\* | 0.217\* | 0.283\*\*\* |
| Q3 | 0.167 | 0.171 | 0.206\* |
| Q4 | 0.083 | 0.107 | 0.128 |
| **Q5 (HIGHEST)** | **−0.003** | **−0.002** | **0.048** |
| R² | 0.119 | 0.119 | 0.091 |
| N | 967,217 | 967,217 | 1,544,198 |

> *"the magnitude of the cross-predictability effects declines monotonically across the analyst coverage quintiles… stocks in the lowest quintile of analyst coverage exhibit significant cross-predictability whereas **stocks in the highest quintile of analyst coverage exhibit no cross-predictability at conventional levels**."*

And from the abstract: *"stocks in the top quintile exhibit no cross-predictability whereas stocks in the bottom quintile exhibit **two to three times more cross-predictability than the average stock**."*

Analyst coverage definitions: (a) active if the analyst made an EPS forecast **within the last twelve months**; (b) active only if the analyst made a forecast **in that month**. Institutional ownership = total 13F holdings ÷ shares outstanding at the most recent quarter-end (alternative: number of institutional holders — qualitatively similar). Holdings data restricted to report dates ≥ 1980; IBES restricted to post-1982 for data-quality reasons.

### Institutional trading (Table IV)
- `ΔIO_(i,q)` on `ΔIO_supplier` = **0.073** (significant); on `ΔIO_customer` = **0.055** (significant).
- `ΔIO_(i,q)` on `r_supplier` = **0.009** (significant); on `r_customer` = **0.007** (significant).
- Panel regressions with **stock fixed effects** and **year-quarter fixed effects**; SEs clustered at the industry-year level.

### Trading strategies (Table V) — **annualized** mean and SD of monthly excess returns, value-weighted industry portfolios, 1963–2005

| Sort variable | High (Q5) portfolio | Low (Q1) portfolio | **L/S** | Sharpe |
|---|---|---|---|---|
| **`r_supplier` (Panel A)** | **10.0%** | **2.8%** | **7.3%** | **≈ 0.7** |
| **`r_customer` (Panel B)** | — | — | **7.0%** | — |
| **`r_composite` (Panel C)** | — | — | **8.7%** | — |

*(Abstract confirms: "buying industries with high returns in supplier industries over the previous month and simultaneously selling industries with low returns in supplier industries over the previous month yields an **annual premium of 7.3%**. A similar trading strategy based on previous-month customer industry returns yields an annual premium of **7.0%**.")*

### Factor exposure (Table VI)
- **MOM loadings: 0.102 (supplier, col 1) and 0.144 (composite, col 3) — positive and statistically significant.**
- Market, HML, SMB: **"fairly orthogonal"** — *"which is not surprising because of the long-short nature of the trading strategies."*
- **Annualized monthly intercepts: supplier 6.2%, customer 5.7%, composite 6.9%** — *"similar to the annualized returns reported earlier."*
- *"The significance of MOM in columns 1 and 3 suggests that the momentum factor may be proxying, at least in part, for sustained information diffusion from related industries."*

### Horizon decay (Internet Appendix Table IA.VI; Section III.B)
- **1/1 composite = 8.7%**; **1/12 composite = 2.1%**.
- The paper's own arithmetic, which is very useful: *"If there were no cross-predictability effects after the first month, the yield on the 1/12 strategy would have been about 0.7% (8.7% divided by twelve), and not 2.1% as observed."* → **cross-predictability continues for up to a year beyond the first month.**
- Table IA.VI formation/holding grid (annualized, J = 1/3/6/12, K = 1/3/6/12). Panel A (supplier): J=1 row = 0.073 (4.27), 0.035 (3.23), 0.016 (1.91), 0.017 (2.70) for K = 1, 3, 6, 12. So the supplier strategy at K=1 is the strongest; **K=6 drops to insignificance (t = 1.91)**.
- Table IA.V (excluding stocks below the 20th NYSE percentile): trading profits are **not** driven by small stocks.

### Hedge-fund corroboration (Table VII)
Credit Suisse/Tremont Long/Short Equity Hedge Fund Index regressed on the strategy returns plus MKT/SMB/HML/MOM, Jan-1994 to Jun-2005:
- **Supplier strategy loading = 0.071 (significant); customer = 0.060; composite = 0.080** — all positive and significant.
- Baseline regression: market beta ≈ 0.5, SMB ≈ 0.2, MOM ≈ 0.2, **net-of-fees alpha indistinguishable from zero.**

---

## 4.5 ROBUSTNESS TESTS — AND WHAT KILLS OR WEAKENS THE EFFECT

| Test | Result |
|---|---|
| **Exclude stocks below the 20th NYSE percentile** | Coefficients shrink (~8–18%): supplier 0.114 → 0.105, customer 0.071 → 0.058. Still significant. Trading profits not driven by small stocks (Table IA.V). |
| **Exclude monthly observations without a closing price** (non-synchronous prices) | supplier 0.117, customer 0.059 — still significant |
| **Single-segment vs. multi-segment firms** (Table IA.I, using Compustat segment files from 1979) | Both similar to the whole sample and to each other → industry misclassification of multi-segment firms is not driving results |
| **Quintile interaction spreads with direct quintile fixed effects** (Table IA.II Panel B), panel regressions with monthly FE | Similar to Panel A and to Table III |
| **Year-quarter clustered standard errors** for the quarterly-ownership specification (Table IA.IV) | t-statistics only slightly smaller; conclusions unaffected |
| **Excluding small caps from the trading strategies** (Table IA.V) | Profits not driven by small stocks |
| **Alternative J/K formation/holding periods** (Table IA.VI) | 1/1 best; **holding periods as long as 12 months still yield more than 2%**; K=6 for the supplier strategy drops to t = 1.91 |
| **Inclusion-probability check** | *"the maximum inclusion probability of an industry rarely exceeds three percent in any of the strategy-leg combinations… and when it does, the industry in question also appears in the opposite leg of the strategy with a similarly high probability."* → not driven by a few industries |
| **Business-cycle interaction with NBER recession dummies** | *"we find similar monthly alphas"* |
| **Panel regressions with time fixed effects instead of Fama–MacBeth** | Similar |

### What KILLS or materially WEAKENS the effect
1. **High analyst coverage / high institutional ownership** → **zero** cross-predictability (Q5 coefficients −0.003, −0.002, 0.048 — all insignificant). This is the single strongest conditioning result in the paper. **The effect lives entirely in the low-attention cross-section.**
2. **The customer direction is much weaker than the supplier direction** (0.071 vs 0.114 stock-level; 0.075 vs 0.113 industry-level).
3. **Long holding periods beyond ~6 months** for the supplier strategy (t falls to 1.91 at K=6).
4. **Positive and significant MOM loading** (0.102–0.144) means a meaningful part of the raw premium is compensation for/co-movement with the momentum factor — the *intercepts* (6.2%/5.7%/6.9% annualized) are the honest alphas, not the 7.0–8.7% raw figures.

---

# IMPLEMENTABLE TAKEAWAYS FOR A-SHARE MARKET WITH LIMITED DATA

## A.0 Available data (as specified) and what each paper actually needs

| Paper | Signal needs | Network needs | Feasible on OHLCV + 83–84 CSRC industries + float cap + turnover + ST? |
|---|---|---|---|
| **Cohen–Frazzini** | linked firms' past-month return | **firm-level customer-supplier disclosure** (Compustat segment / SFAS 131 10% rule) | **NO** — no firm-level link data. Structure is proxyable at *industry* level only. |
| **Lee et al.** | TECH-weighted linked return | **patent class distributions across 427 USPTO classes** | **NO** — but the *statistical* link can be re-estimated from returns. |
| **Parsons et al.** | same-city, different-industry EW return | **firm HQ zip → BEA Economic Area, + FF-12 industry** | **NO from OHLCV alone** — but the *firm→city* map is cheap to obtain (see A.5). Industry part is free. |
| **Menzly–Ozbas** | I-O-weighted supplier/customer industry return | **BEA Benchmark Input-Output Use Table** | **NO from OHLCV alone** — but the China I-O table exists (see A.4). |

**Blunt statement: none of the four economic-linkage networks can be reconstructed from daily OHLCV bars, industry codes, float cap, turnover and ST status.** Every one of them requires an *external economic graph*. What you can build are **statistical linkage proxies** that preserve each paper's *portfolio-construction geometry* while replacing the economic graph with a data-estimated one.

## A.1 The single most important structural constraint: the short leg is mostly unavailable

This is not a detail — it is the dominant translation loss, and it invalidates the naive implementation of all four factors.

- All four papers' profits are **concentrated in low-attention, small-cap, low-coverage, high-idiosyncratic-volatility stocks**:
  - Cohen–Frazzini: high-inattention subsample 2.70%/mo vs low-inattention 0.61% (t = 1.05).
  - Lee et al.: inattention interactions all negative and significant; VW L/S is 41% smaller than EW.
  - Parsons et al.: geographic effect is **flat** across size and coverage (the one exception — but the effect is only 42 bps/mo).
  - Menzly–Ozbas: **top analyst-coverage quintile has literally zero cross-predictability**; the entire effect is in Q1–Q2.
- In A-shares, short selling is restricted to the **融资融券 (margin/securities-lending) eligible list**, which is skewed toward **large, liquid, heavily-covered, high-turnover** names — precisely the subpopulation where the literature says the effect is weakest or absent.
- **Implication:** build these as **long-only tilts / index-enhancement overlays within a benchmark**, or benchmark them against a long-only portfolio formed on the *same universe you can actually trade*. **Do not report an L/S backtest with a short leg you cannot implement.** If you must report an L/S number, restrict **both** legs to the lendable universe and expect materially smaller alpha.

Two further mechanical frictions:
- **T+1 settlement** means a signal computed on month-end data cannot be traded until the next session, and the position cannot be reversed intraday.
- **±10% daily price limits (±5% for ST)** truncate the first-day price adjustment. Since **all of the profit in Cohen–Frazzini and ~90% in Lee et al. is in the first month**, and a large part in the first days, limit-truncation is a first-order concern for the highest-signal names.

## A.2 Industry granularity: 83–84 is the wrong number, and Mozbas tell you why

Menzly & Ozbas deliberately chose **broad** industry groups: *"We intentionally select such relatively broad groupings in order to reduce the extent of overlap between firms classified in different industries."* Parsons et al. use **FF-12** for the same reason. Cohen & Frazzini use 48 but find that **78% of links are cross-industry**.

- With **83–84 CSRC industries** and daily A-share coverage, many "related industry" portfolios become **1–3 stocks**, which is not a portfolio — it is a stock. Menzly–Ozbas sort *industries* into quintiles; with 83 industries you get ~16 per quintile, which is workable, but the **related-industry portfolios themselves** become noisy.
- Parsons et al. impose a **minimum of 10 firms** in a lead portfolio. Adopt the same rule.
- **Recommendation: build a two-layer industry map.** Keep the 83–84 CSRC codes for *identification*, but **aggregate to 12–20 coarse sectors** for the *linkage layer* (the analogue of FF-12 / BEA accounts). This is the single cheapest way to raise the signal-to-noise ratio of any related-industry factor.

## A.3 The general-purpose proxy: a data-estimated linkage graph

All four papers share this geometry:

```
1. For each firm i, define a SET of economically related firms/industries L(i), excluding i itself,
   and (in Parsons and Cohen-Frazzini) excluding i's own industry.
2. Compute a weighted average of the PAST-MONTH returns of L(i)  ->  the signal  S(i, t-1).
3. Sort the cross-section on S(i, t-1); go long high-S, short low-S.
4. Weight the TRADED portfolio by float market cap; hold 1 month; rebalance monthly.
5. Control for: own 1-month reversal, own 12-1 momentum, own-industry momentum,
   and a DGTW-style characteristic benchmark.
```

The only thing you cannot copy is step 1's **economic** graph. So replace it with a **statistically estimated** one, and be explicit that the economic identification is gone.

### Proxy graph construction (estimation window strictly before the trading month)

```
# ---------- estimate the graph on a rolling window, OUT OF SAMPLE ----------
estimation window: months [t-36, t-1]   (36 months; use >= 24 months minimum)
for each stock i:
    r_resid[i, s] = residual from OLS( r[i,s] - rf[s] ~ MKT[s] + IND_own[s] + SMB + HML )
        # residualising removes market + own-industry + style exposure.
        # THIS STEP IS NOT OPTIONAL: without it the "link" is just common-factor exposure
        # and the signal collapses into industry momentum, which is exactly what
        # Cohen-Frazzini/Lee et al. control for and what Menzly-Ozbas already document
        # as a separate 0.134 effect.
for each pair (i, j) with IND(i) != IND(j):
    w[i,j] = corr( r_resid[i, .], r_resid[j, .] )   on the estimation window only
sort j by w[i,j] descending; keep the TOP-K (K = 20..50)
L(i) = that top-K set, with weights w[i,j]

# ---------- the signal ----------
S[i, t-1] = Σ_{j in L(i)} w[i,j] * RET[j, t-1]  /  Σ_{j in L(i)} w[i,j]
    # This is a Cohen-Frazzini/Lee-et-al. style weighted linked return.
    # For the Parsons analogue, use EQUAL weights:
S_geo[i, t-1] = mean( RET[j, t-1] for j in L(i) )

# ---------- sort and trade ----------
universe = { i : S[i,t-1] exists, not ST, listed >= 12 months, float cap >= threshold }
deciles (or quintiles) on S ascending
long leg  = top decile, FLOAT-CAP weighted
short leg = bottom decile (ONLY within the 融资融券 lendable list, if reported at all)
hold 1 month, rebalance monthly
```

**Mandatory caveats to state alongside any result from this proxy:**
1. **The link is estimated from the same returns you are predicting.** Mitigate with (a) a strictly out-of-sample estimation window, (b) residualisation, (c) a pre-registered K, and (d) a placebo test using *randomly reassigned* links of the same size distribution — if the placebo earns a similar alpha, you have data-mined.
2. **The economic story is lost.** Parsons' story is *local* shocks; Cohen–Frazzini's is *contractual* customer–supplier dependence; Lee et al.'s is *technology-space* adjacency. A correlation graph recovers none of these — it recovers "stocks that move together," which is very close to what a **statistical risk model already knows**. Expect the alpha to be substantially absorbed by a good multi-factor risk model, and test it explicitly against an industry-momentum factor and a "co-movement"/crowding factor.
3. **Directionality is unrecoverable.** Cohen–Frazzini's customer→supplier asymmetry and their `IMPSUPP` "important supplier" conditioning require disclosure data. A symmetric correlation graph cannot express either.

## A.4 Proxy F-MO — "Input-output chain momentum" (the highest-fidelity A-share factor)

**This is the one I would build first, and it requires no OHLCV trickery — it requires one external table.**

- **China publishes the equivalent of the BEA Use Table:** the **国家统计局 投入产出表 (National Bureau of Statistics of China Input-Output Table)**, roughly every five years (e.g. 2017, 2018, 2020, 2023 extended tables), with ~149 sectors, plus **投入产出基本流量表** giving the inter-industry flow matrix. This maps cleanly onto CSRC industries.
- Replicating Menzly–Ozbas then becomes almost mechanical:
  1. Map each A-share to a CSRC industry; map CSRC industries to I-O sectors.
  2. Build `w_supplier[u][s] = USE[s][u] / total_purchases(u)` and `w_customer[u][v] = USE[u][v] / total_sales(u)`.
  3. Compute `r_supplier[u]`, `r_customer[u]`, `r_composite[u]` as flow-weighted averages of **float-cap-weighted** industry returns in month t−1.
  4. Sort **industries** into quintiles; form **float-cap-weighted** portfolios; L/S = Q5 − Q1; hold 1 month.
  5. Alpha vs. market + size + value + **momentum** (the MOM loading was 0.102–0.144 in Mozbas, so this control matters).
- **Honour the timing convention:** Mozbas delay use of a survey until the **end of the year it is publicly released**. Since China's tables are released with a long lag (often 1–2 years), applying the most recent available table with a **≥1-year publication lag** is both faithful and prudent.
- **What is lost vs. Mozbas:** (i) the table is a **national** table, so it misses regional supply chains (a bigger deal in China than in the US); (ii) Chinese listed firms are a non-random, state-influenced subset of the economy, so I-O flow weights computed on the whole economy misweight the listed supply chain; (iii) the **data frequency is 5-yearly** vs. Mozbas's rolling 11-survey design, so the weights are stale.
- **What is NOT lost:** the economic identification. This is the only proxy in this document whose "link" is an actual economic flow. **If the I-O table can be obtained, this dominates every OHLCV-only alternative.**

## A.5 Proxy F-GEO — "Non-industry peer momentum" (Parsons analogue)

Parsons' signal is `r_(c,∉j,t)`: the EW return of firms sharing the firm's **city** but not its **industry**.

- **From OHLCV alone: not implementable.** Geographic lead-lag needs a firm→location map. Period.
- **But the map is cheap.** Tushare's `stock_company` (上市公司基本信息) exposes `province` / `city` / `office` address for A-share listed companies; annual reports and exchange disclosures also carry it. **This is a one-off static join, not a data pipeline.**
- Available in this environment: the `tushare-data` skill is present in the session skill catalog. **Recommendation: if the parent wants a genuine geographic factor, obtain the firm→province/city map once and cache it.**
- **Mapping the "economic area":** the BEA EA concept (a metro core plus commuting hinterland) maps onto Chinese **城市群 / 都市圈** (Yangtze River Delta, Pearl River Delta, Beijing–Tianjin–Hebei, Chengdu–Chongqing, Middle Yangtze). Using **province** is the crude version; using a **city-cluster map** is the faithful version. A-share HQ concentration is extreme (Guangdong, Jiangsu, Zhejiang, Beijing, Shanghai dominate), which is *favourable* — each city-cluster × coarse-sector cell will have enough names.
- **Honour the exclusions:** same-city **but different industry (∉j)** for the signal; **minimum 10 firms** per lead portfolio; **equal-weight the signal portfolio** and **float-cap-weight the traded portfolio** (Parsons' own asymmetry — and they show value-weighting the signal costs 40–50%).
- **Without the map, the closest OHLCV-only substitute** is a "statistically nearest non-industry peers" graph (A.3). Say plainly that this **discards the geographic mechanism entirely** — what remains is a generic co-movement momentum, not Parsons' factor.

## A.6 Proxy F-TECH / F-CF — "Cross-industry peer momentum"

For each stock, take the top-K industries (excluding its own) by historical residual co-movement, and use the float-cap-weighted return of those industries over month t−1 as the signal. This is the OHLCV-only stand-in for both Cohen–Frazzini (cross-industry customer-supplier links, 78% cross-industry) and Lee et al. (technology-space adjacency, which also cuts across industries).

- **What survives translation:** the geometry — a related-set return, excluding own industry, used to sort the cross-section, with float-cap weighting, monthly rebalancing, and the full control set.
- **What is lost:** (i) that the relation is *contractual* (CF) or *technological* (LSYZ) rather than merely *statistical*; (ii) the customer→supplier **direction**; (iii) the economic-proximity **weighting**, which is the single largest contributor in Lee et al. — value-weighting the linked-firm returns instead of TECH-weighting cuts the effect from 1.17% to 0.39% (EW). **A correlation weight is a much weaker object than a patent-overlap weight**, so expect the proxy to sit closer to the equal-weighted (1.07%) than to the TECH-weighted (1.17%) result — and if you use market-cap weights, expect ~0.39%.
- **Where it will collide:** Lee et al. themselves show the (1,1) industry-adjusted return is 0.99% vs 1.17% raw, and that `INDRET(t−1)` retains no predictive power once `TECHRET(t−1)` is included — i.e. **technology momentum and industry momentum are correlated but distinguishable**. A statistical graph will **not** preserve that separation; it will load much more heavily on industry momentum. **Always report the incremental alpha over a pure industry-momentum factor.**

## A.7 The attention overlay — the one thing you can do *natively* and *faithfully*

Every paper in this set agrees the effect is a **low-attention, high-limits-to-arbitrage** phenomenon. Cohen–Frazzini and Menzly–Ozbas both find the effect **goes to zero** in the high-attention subsample. You have exactly the right variables for this:

| Available variable | Role | Literature anchor |
|---|---|---|
| **Turnover** (and **abnormal turnover** = turnover ÷ its 12-month median) | Primary attention/arbitrage-cost proxy | Lee et al. show the effect is stronger for high `IdioVol`; Parsons show industry lead-lags weaken in high-volume stocks but geographic ones do not |
| **Float market cap** | Size/attention; also the weighting variable | Lee et al. and Mozbas both show VW ≈ half of EW |
| **ST status** | **Use as an EXCLUSION filter, not a conditioning variable** | ±5% price limits, distressed, essentially unshortable, and the "bad news" leg of these factors would concentrate there |

**Recommended construction:** compute the linkage signal on the **full** cross-section, then run the strategy **within the lowest-turnover tercile (or lowest float-cap tercile)**, excluding ST. Report the monotonicity of the long-short spread **across turnover terciles** as the mechanism test — that is the closest available analogue to Cohen–Frazzini's `COMOWN` sort (2.70% high-inattention vs 0.61% low-inattention) and to Mozbas's analyst-coverage quintiles (0.287 at Q1 → −0.003 at Q5). If your spread does not decline monotonically in turnover, the effect you found is probably not the effect these papers document.

## A.8 What is lost in translation — a consolidated list

1. **Economic identification.** Three of the four networks (customer–supplier contracts, patent-class overlap, input-output flows, HQ geography) are external economic graphs. A statistical graph is a *co-movement* graph. It is unfalsifiable as economics and vulnerable to data-mining. **The only proxy that preserves identification is the China I-O table (A.4).**
2. **Direction.** Cohen–Frazzini's customer→supplier asymmetry, and their `IMPSUPP` conditioning for the reverse direction, are unrecoverable without disclosure data. A correlation graph is symmetric by construction. Menzly–Ozbas are symmetric *by design*, so this loss is smaller for the I-O proxy.
3. **The proximity weighting.** Lee et al.'s TECH weight is worth roughly +78 bps/month over market-cap weighting. Correlation weights will not reproduce that.
4. **Geography.** Parsons' factor requires a firm→city map; OHLCV alone cannot supply it. The map is cheap but must be sourced.
5. **Analyst overlap.** Parsons' moderator (analyst overlap, not analyst coverage) and Mozbas's analyst-coverage quintiles both require IBES-style coverage data, which is not in the available set. **Turnover is a weaker but usable substitute**, and — importantly — Parsons shows the geographic effect is *flat* in coverage anyway, so for F-GEO this loss is minor.
6. **The short leg.** As argued in A.1, this is the largest practical loss and it cannot be engineered away. It biases every reported long-short alpha upward relative to what is implementable.
7. **The decay in recent decades.** Cohen–Frazzini's VW effect drops to t ≈ 2.0 in 1993–2004; Parsons' geographic lead-lags *"weakened somewhat over the last two decades."* A-share data starts around 2000–2005 at the earliest for full coverage, so **you are testing these factors entirely in their weak regime.** Set expectations accordingly, and test sub-period stability from the outset.

## A.9 Recommended build order

1. **Obtain the firm→province/city map** (one join) and **the China I-O Use Table** (public). These two additions convert three of the four papers from "not implementable" to "implementable with caveats."
2. **Build F-MO (I-O chain momentum, industry-level, float-cap-weighted, 1-month hold)** and test the incremental alpha over market + size + value + momentum + own-industry momentum. This is the highest-fidelity factor available.
3. **Build F-GEO (city-cluster × coarse-sector, ∉ own industry, EW signal → float-cap-weighted traded portfolio, min 10 firms)** if the geography map is obtained.
4. **Build F-CF/F-TECH (residual-correlation cross-industry peer momentum)** last, with an out-of-sample graph, a random-link placebo, and mandatory incremental-alpha reporting over industry momentum.
5. **For every factor, run the turnover-tercile monotonicity test** and report the long-only, lendable-universe-restricted version as the headline implementable number.

## A.10 Lead worth pursuing (not read — flagged as unverified)

A Bing result surfaced a paper titled **"Is Technology-Linked Return an Effective Factor for A-Share Stock [Market]"** (Springer, `https://link.springer.com/content/pdf/10.1007/978-981-19-9741-9_22.pdf`), which appears to apply the Lee et al. technology-link idea directly to A-shares. **I did not read it**, and I am not citing it as evidence — but it is the obvious prior-work check before building F-CF/F-TECH, and it may report how the patent-linkage problem was handled for China (where the analogue would be CNIPA patent classification data, not USPTO classes).

---

## APPENDIX: Consolidated numeric cheat-sheet

| | Cohen–Frazzini (2008) | Lee et al. (2017 WP) | Parsons et al. (2020 AEA) | Menzly–Ozbas (2010) |
|---|---|---|---|---|
| **Link type** | firm-level customer–supplier (SFAS 131) | firm-level patent-class overlap (427 USPTO classes) | firm-level HQ city (BEA EA) × FF-12 industry | industry-level BEA input-output flows |
| **Unit sorted** | stock | stock | stock | **industry** |
| **Signal weights** | **equal** (customer returns) | **TECH closeness** | **equal** (area returns) | **I-O flow shares** |
| **Traded weights** | value (headline) / equal | both reported | **value** | **value** |
| **Window** | 1 month (12-month variant) | 1 month (k = 3/6/12 variants) | 1 month | 1 month (J/K variants) |
| **Sort** | quintiles | **deciles** | quintiles | quintiles |
| **Direction** | **customer → supplier** (main); supplier → customer only with `IMPSUPP` | symmetric adjacency | **same-city, different-industry → firm** | **symmetric** (supplier and customer both) |
| **Sample** | 1980/81–2004 | Jul-1963–Jun-2012 | 1970–2013 | Jul-1963–Jun-2005 |
| **L/S raw** | 1.578 VW (t=3.79) / 1.311 EW (t=4.93) | **1.17 EW (t=5.47)** / 0.69 VW (t=3.19) | **0.42 (t=3.65)** | supplier 7.3%/yr; customer 7.0%/yr; **composite 8.7%/yr** |
| **Headline alpha** | 3-factor **1.555 (t=3.60)** VW / 1.306 EW; 4-factor **1.376 (t=3.13)** VW / 1.212 EW; 5-factor **1.237 (t=2.99)** VW / **1.359 (t=4.79)** EW; abstract ">150 bps" | EW: FF3 1.26%, **FF3+MOM 1.08%**, FF5 1.37%, FF5+MOM 1.21% | CAPM **0.47 (t=4.16)**; FF5 **0.42 (t=3.32)** | annualized intercepts: supplier **6.2%**, customer 5.7%, composite **6.9%**; MOM loading 0.102–0.144 |
| **After own momentum** | 1.555 → 1.376 VW (−11.5%); 1.306 → 1.212 EW (−7%) | 1.26% → 1.08% (−14%) | "makes no difference" | MOM loading significant, intercepts reported |
| **After liquidity too** | VW **1.237 (t=2.99)**; EW **1.359 (t=4.79)** | not tested | not tested | not tested |
| **After industry + all controls** | **~88 bps/mo VW** (−43%), t ≈ 3.2 | coefficient "virtually the same"; industry momentum itself goes to zero | city coeff 0.061 (t=5.11) vs industry 0.243 (t=11.71) | supplier 0.114 / customer 0.071 vs industry momentum 0.134 |
| **Kills it** | 1-yr VW sort; high common ownership; 1993–2004 sub-period | nothing outright | ≥50% HQ misclassification | top analyst-coverage quintile (zero) |
| **Weakens it** | industry adjustment (−38%); DGTW (−29%) | value weighting (−41%); long horizons | value-weighting the signal (−40/50%); post-1990 decay | customer direction; long horizons |
| **Key mechanism quote** | *"News travels slowly across assets as investors with limited attention overlook the impact of specific information on economically related firms."* | *"sluggish price adjustment to more nuanced technological news"* + harder to value + limits to arbitrage | *"the lead-lag relationship is determined by the extent to which analysts overlap between two (or more) firms… the main contribution of the paper"* | *"due to investor specialization and market segmentation, value-relevant information diffuses gradually in financial markets"* |
| **Theory** | Merton (1987); **Hong & Stein (1999)**; Hirshleifer & Teoh (2003) | **Hong & Stein (1999)**; Hirshleifer, Hsu & Li (2013); Acemoglu et al. (2012) | **Hong & Stein (1999)**; Hong, Torous & Valkanov (2007); Merton (1987) | **Hong & Stein (1999)**; **Hong, Torous & Valkanov (2007)**; Grossman & Stiglitz (1980) |
