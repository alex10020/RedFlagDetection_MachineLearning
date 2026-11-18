# Business case sources

> Collected by Claude through web research on 18 November 2026, then **checked**: we fetched each page and searched it for the exact figure. **Verified** means the figure or quote appears on the cited page. **Not verified** means the page blocked automated access, so open it in a browser before using the figure. The report and slides use verified facts only.
>
> Reliability: **P** = primary (filing, law, regulator, survey publisher), **PR** = company press release, **MR** = market-research estimate.

## 1. Market size

| Fact | Source | Rel. | Check |
|---|---|---|---|
| The dating-app sector "generated more than $6 billion in revenue in 2024", projected to reach $8.9 billion by 2030 | Global Dating Insights, 27 Aug 2025, reporting Business of Apps: https://www.globaldatinginsights.com/featured/global-dating-app-revenues-top-6-billion-likely-primed-to-grow/ | MR, second-hand | Verified |
| Match Group 2025: "Total Revenue of $3.5 billion was flat year-over-year"; Payers down 5% to 14.2 million | Match Group FY2025 results, 3 Feb 2026: https://ir.mtch.com/investor-relations/news-events/news-events/news-details/2026/Match-Group-Announces-Fourth-Quarter-and-Full-Year-Results | P | Verified. **Correction:** the release says $3.5B, not $3.49B |
| Bumble app about 2.4 million paying users | Bumble FY2025 10-K (mirror of the SEC filing): https://www.boardroomalpha.com/sec/bmbl-10-k-2026-03-16-0001830043-26-000027 | P | Verified on the mirror; cite SEC EDGAR, accession 0001830043-26-000027 |
| "Three-in-ten U.S. adults say they have ever used a dating site or app" (survey of 5–17 July 2022) | Pew Research Center, 2 Feb 2023: https://www.pewresearch.org/internet/2023/02/02/from-looking-for-love-to-swiping-the-field-online-dating-in-the-u-s/ | P | Verified |
| Content moderation services USD 10.1B (2023), forecast USD 32.8B by 2033, 12.5% CAGR | Market.us: https://market.us/report/content-moderation-services-market | MR | Verified; other firms' estimates differ |

## 2. Harassment and abuse on dating apps

- **Pew (2023):** among people who have ever used a dating site or app, 38% received unwanted sexually explicit messages or images, 30% were contacted again after saying they were not interested, "about a quarter (24%) report being called an offensive name", 6% were threatened with physical harm, and "about half (48%) have experienced at least one of these four unwanted behaviors". Among women under 50: 56%, 43%, 37%, 11%. https://www.pewresearch.org/internet/2023/02/02/the-experiences-of-u-s-online-daters/ (P, **verified**)
- **Pew (2020):** among women under 35, 44% were called offensive names and 19% were threatened with physical harm; 57% of female users aged 18–34 received unwanted explicit messages. https://www.pewresearch.org/short-reads/2020/03/06/young-women-often-face-sexual-harassment-online-including-on-dating-sites-and-apps/ (P, **verified**)
- **Behavioural Insights Team for the Minderoo Foundation (2026):** survey of 1,008 heterosexual women across Australia who had used a dating platform in the past three months. https://www.bi.team/wp-content/uploads/2026/09/Understanding-tech-based-abuse-of-women-on-dating-platforms.pdf (P, non-probability sample, **all verified in the PDF text**)
  - "99% of women experienced at least one instance of tech-based abuse on dating platforms in the last three months."
  - Psychological or emotional abuse: 95%.
  - "**Gaslighting was experienced by 53% of women** ('How often have you had your concerns dismissed, been called crazy, or been blamed for…')."
  - "Subtler forms of harm, or precursors (such as gaslighting, negging, love-bombing) are commonly experienced."
  - Overt behaviours such as threats were more often reported to platforms, while "psychological and coercive behaviours were reported far less despite being highly prevalent".
  - "Over-reliance on automated moderation risks eroding confidence in platform safety" and "Platforms should combine automated detection with clear, accessible pathways to human review". **This is almost exactly our triage design.**
- **CU Boulder 2021 campus survey:** 67% of undergraduate women and 57% of men said a partner "manipulated you into doubting your own memory, perception, or judgment" ("gaslighting"). These are rates among students who reported intimate partner abuse, not all students. https://www.colorado.edu/oiec/data/survey-projects/2021-cu-boulder-sexual-misconduct-survey-results/survey-results-intimate (P, verified, **limited scope; not used in the report**)

## 3. Existing safety features

| Feature | What the source says | Source | Check |
|---|---|---|---|
| Tinder "Does This Bother You?" (23 Jan 2020) | Detects whether an offensive message has been sent; "powered by machine learning"; the recipient can report the sender | https://www.tinderpressroom.com/tinder-introduces-safety-updates (PR) | Verified |
| Hinge "Are You Sure?" (Feb 2025) | "Asks users to pause and reassess before sending any message that may not land right"; "22% of those tested who encountered Are You Sure? edited their message before sending" | https://hinge.co/press/Safer-Internet-Day-2025 (PR) | Verified |
| Bumble / Badoo (FY2025 10-K) | "Review before you send"; Badoo's "Rude Message Detector"; machine learning that identifies harassment, "then flagged to moderators" | Bumble 10-K (above), P | Verified on the mirror |
| Bumble Private Detector (2019, open-sourced 2022) | Blurs suspected nude images; Bumble claims 98% accuracy | https://bumble.com/the-buzz/bumble-open-source-private-detector-ai-cyberflashing-dick-pics (PR) | Verified |
| Match Group (FY2025 10-K) | Operates "shared services across brands, including trust and safety and moderation" | Match 10-K mirror: https://www.boardroomalpha.com/sec/mtch-10-k-2026-02-26-0000891103-26-000025 (P) | Verified on the mirror; cite EDGAR |
| Tinder "Are You Sure?" (May 2021): inappropriate language fell by more than 10%, and reports rose 46% | https://mtch.com/?p=1691 (PR) | | **Not verified** (mtch.com returns 403 to scripts). Not used |
| "Are You Sure?" across Match brands (Nov 2023): over 500M messages edited or deleted | https://mtch.com/?p=1332 (PR) | | **Not verified** (403). Not used |

**What this means for us:** platforms already detect explicit rudeness, which is the part our models do well. None of these features claims to detect gaslighting or coercion, which the Minderoo survey finds common and under-reported. That is our differentiation and our Phase 2.

## 4. Regulation (demand driver)

- **UK Online Safety Act 2023** (GOV.UK explainer, P, **verified**): the Act covers "dating services". "Companies can be fined up to £18 million or 10 percent of their qualifying worldwide revenue, whichever is greater." Illegal content includes "harassment, stalking, controlling or coercive behaviour". https://www.gov.uk/government/publications/online-safety-act-explainer/online-safety-act-explainer
- **Match Group FY2025 10-K** (P, **verified**): "the UK's Online Safety Act imposes broad and similar requirements to those provided in the DSA... this law places new requirements on social media companies, including online dating companies".
- **EU Digital Services Act**, Regulation (EU) 2022/2065 (P, **verified** on the Commission page): applies from 17 February 2024; fines up to 6% of global turnover. https://digital-strategy.ec.europa.eu/en/policies/dsa-enforcement
- **GDPR Art. 9** (**verified**): processing data on "sex life or sexual orientation" is prohibited unless an exception applies, such as explicit consent (9(2)(a)). Art. 6(1) lists the lawful bases. https://gdpr-info.eu/art-9-gdpr/ (unofficial copy; cite Regulation (EU) 2016/679)

## 5. Not used

- Match's "$125 million annually in trust and safety" appears only on an unofficial mirror (`mtch.dd-nyc.co`), and Match's 10-K does not state it. Do not quote it unless someone finds it on mtch.com.

## Numbers used in the report and slides

1. **Need:** 48% of US online daters have faced at least one form of harassment, including 24% called an offensive name (Pew 2023). 53% of women surveyed in Australia experienced gaslighting on dating platforms, and coercive behaviours are reported far less than overt ones (BIT/Minderoo 2026).
2. **Market:** dating apps earned more than $6B in 2024 (Business of Apps via GDI). Match Group alone had $3.5B of revenue in 2025 and runs trust and safety and moderation as a shared service across its brands.
3. **Why now:** the UK Online Safety Act covers dating services, with fines up to 10% of worldwide revenue, and lists controlling or coercive behaviour as illegal content.

## References (APA)

- Behavioural Insights Team. (2026). *Understanding tech-based abuse of women on dating platforms: Final report*. Minderoo Foundation. https://www.bi.team/wp-content/uploads/2026/09/Understanding-tech-based-abuse-of-women-on-dating-platforms.pdf
- Bumble Inc. (2026). *Annual report on Form 10-K for the fiscal year ended December 31, 2025*. U.S. Securities and Exchange Commission.
- Department for Science, Innovation and Technology. (2025). *Online Safety Act: Explainer*. GOV.UK. https://www.gov.uk/government/publications/online-safety-act-explainer/online-safety-act-explainer
- Global Dating Insights. (2025, August 27). *Global dating app revenues top $6 billion, likely primed to grow*. https://www.globaldatinginsights.com/featured/global-dating-app-revenues-top-6-billion-likely-primed-to-grow/
- Hinge. (2025, February 11). *Safer Internet Day 2025* [Press release]. https://hinge.co/press/Safer-Internet-Day-2025
- Market.us. (n.d.). *Content moderation services market*. https://market.us/report/content-moderation-services-market
- Match Group. (2026). *Annual report on Form 10-K for the fiscal year ended December 31, 2025*. U.S. Securities and Exchange Commission.
- Match Group. (2026, February 3). *Match Group announces fourth quarter and full-year results* [Press release].
- Pew Research Center. (2023, February 2). *The experiences of U.S. online daters*. https://www.pewresearch.org/internet/2023/02/02/the-experiences-of-u-s-online-daters/
- Regulation (EU) 2016/679 (General Data Protection Regulation).
- Regulation (EU) 2022/2065 (Digital Services Act).
- Tinder. (2020, January 23). *Tinder introduces safety updates* [Press release]. https://www.tinderpressroom.com/tinder-introduces-safety-updates
