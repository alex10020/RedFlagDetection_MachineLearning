# Business case sources

> Collected by Claude through web research on 18 November 2026. **Open every link before you quote it.** Some pages could only be read on a mirror or second-hand, and those are marked. Reliability: **P** = primary (filing, law, regulator, survey publisher), **PR** = company press release, **MR** = market-research estimate.

## 1. Market size

| Fact | Year | Source | Rel. |
|---|---|---|---|
| Global dating-app revenue "more than $6 billion" | 2024 | Business of Apps, *Dating App Report 2025*, as reported by Global Dating Insights (27 Aug 2025): https://www.globaldatinginsights.com/featured/global-dating-app-revenues-top-6-billion-likely-primed-to-grow/ | MR, second-hand |
| Match Group revenue $3.49B (flat year on year); 14.2M payers (−5%) | FY2025 | Match Group Q4/FY results, 3 Feb 2026: https://ir.mtch.com/investor-relations/news-events/news-events/news-details/2026/Match-Group-Announces-Fourth-Quarter-and-Full-Year-Results | P |
| Bumble app about 2.4M paying users; Badoo and other apps about 1.2M | 2025 | Bumble FY2025 10-K (16 Mar 2026), read on a mirror: https://www.boardroomalpha.com/sec/bmbl-10-k-2026-03-16-0001830043-26-000027. Cite from SEC EDGAR, accession 0001830043-26-000027 | P |
| 3 in 10 US adults have ever used a dating site or app | 2022 survey | Pew Research Center, 2 Feb 2023: https://www.pewresearch.org/internet/2023/02/02/from-looking-for-love-to-swiping-the-field-online-dating-in-the-u-s/ | P |
| Content moderation services USD 10.1B in 2023, forecast USD 32.8B by 2033 (12.5% CAGR) | 2023 | Market.us: https://market.us/report/content-moderation-services-market | MR; other firms' estimates differ widely |

Not found: Statista's worldwide figures (paywalled) and a verified global user count.

## 2. Harassment and abuse on dating apps

- **Pew (2023), 6,034 US adults surveyed in July 2022.** Among online daters, 38% received unwanted sexually explicit messages or images, 30% were contacted again after saying they were not interested, 24% were called offensive names and 6% were physically threatened. 48% experienced at least one of these. Among women under 50 the figures were 56%, 43%, 37% and 11%. https://www.pewresearch.org/internet/2023/02/02/the-experiences-of-u-s-online-daters/ (P)
- **Pew (2020), October 2019 survey.** Among women under 35, 44% were called offensive names and 19% were threatened with physical harm; 57% of female users aged 18–34 received unwanted explicit messages. https://www.pewresearch.org/short-reads/2020/03/06/young-women-often-face-sexual-harassment-online-including-on-dating-sites-and-apps/ (P)
- **Behavioural Insights Team for the Minderoo Foundation (2026)**, 1,008 heterosexual women in Australia who had used a dating platform in the past 3 months. https://www.bi.team/wp-content/uploads/2026/09/Understanding-tech-based-abuse-of-women-on-dating-platforms.pdf (P, non-probability sample)
  - 95% experienced psychological or emotional abuse in the last 3 months.
  - **53% experienced gaslighting** ("concerns dismissed, been called crazy, or been blamed for someone else's behaviour").
  - Psychological and coercive behaviours were "reported far less despite being highly prevalent".
  - It warns against "over-reliance on automated moderation". **This strongly supports our gaslighting focus and the human-in-the-loop design.**
- **CU Boulder 2021 campus survey.** 67% of undergraduate women and 57% of men said a partner "manipulated you into doubting your own memory, perception, or judgment". The denominator appears to be students who reported partner abuse, so check it before quoting. https://www.colorado.edu/oiec/data/survey-projects/2021-cu-boulder-sexual-misconduct-survey-results/survey-results-intimate (P, limited scope)

## 3. Existing safety features (competitors and benchmarks)

| Feature | What it does | Launched | Source | Rel. |
|---|---|---|---|---|
| Tinder "Does This Bother You?" | Machine learning flags a possibly offensive message the user received and offers to report the sender | 23 Jan 2020 | https://www.tinderpressroom.com/tinder-introduces-safety-updates | PR |
| Tinder "Are You Sure?" | Warns the sender before an offensive message is sent. Inappropriate language fell by more than 10%; reports rose 46% among users who had seen "Does This Bother You?" | 20 May 2021 | https://mtch.com/?p=1691 | PR |
| "Are You Sure?" across Match Group | Expanded to Match, Stir, OurTime and others, with Hinge in 2024. Over 500M messages edited or deleted after a prompt; 18 languages | 14 Nov 2023 | https://mtch.com/?p=1332 | PR |
| Hinge "Are You Sure?" | 22% of tested users who saw it edited their message | 11 Feb 2025 | https://hinge.co/press/Safer-Internet-Day-2025 | PR |
| Bumble Private Detector | Blurs suspected nude images; open-sourced in 2022 | 2019 | https://bumble.com/the-buzz/bumble-open-source-private-detector-ai-cyberflashing-dick-pics | PR |
| Bumble / Badoo tools | "Review before you send" prompt; Badoo's "Rude Message Detector" ("machine learning to detect any text that could be perceived as rude"); machine learning that identifies harassment and hate, then "flagged to moderators" | Dates not found | Bumble FY2025 10-K (above) | P |

**What this means for us:** the big platforms already detect explicit rudeness, which is the part our models do well. The gap is subtle harm and gaslighting, which the platforms do not claim to detect and which the Minderoo survey finds both common and under-reported. That is our differentiation and our Phase 2.

## 4. Regulation (demand driver)

- **EU Digital Services Act**, Regulation (EU) 2022/2065. It applies to all platforms from 17 Feb 2024; micro and small enterprises are exempt from some duties. Platforms must take user reports, explain removals, and offer appeals and out-of-court dispute settlement. Fines go up to 6% of global turnover. https://digital-strategy.ec.europa.eu/en/policies/dsa-enforcement (P). Cite the regulation itself from EUR-Lex.
- **UK Online Safety Act 2023.** Dating services are explicitly in scope. Illegal-content risk assessments were due by 16 Mar 2025 and are enforced from 17 Mar 2025. Fines go up to £18M or 10% of qualifying worldwide revenue. **"Controlling or coercive behaviour" is on the list of illegal content.** https://www.gov.uk/government/publications/online-safety-act-explainer/online-safety-act-explainer (P)
- **Match Group FY2025 10-K:** the DSA "imposes additional requirements… around moderation, transparency", and the OSA applies to "online dating companies". https://www.boardroomalpha.com/sec/mtch-10-k-2026-02-26-0000891103-26-000025 (P, mirror; cite EDGAR)
- **GDPR Art. 6(1):** lawful bases, including (a) consent and (f) legitimate interests. **Art. 9(1):** processing data on "sex life or sexual orientation" is prohibited unless an exception such as 9(2)(a) explicit consent applies. This matters for Phase 2 training on reported messages. https://gdpr-info.eu/art-9-gdpr/ (unofficial copy; cite Regulation (EU) 2016/679)

## 5. Trust and safety spending

- Match Group: "We invest $125 million annually in trust and safety teams, features and initiatives" (CEO letter, 29 Apr 2024). This was read only on a mirror, https://mtch.dd-nyc.co/kr/?p=2032, so **find it on mtch.com before citing it**.
- Match had about 2,200 full-time employees at 31 Dec 2025, with "shared services across brands, including trust and safety and moderation". Bumble had about 580 after about a 30% cut in 2025. Both are from the FY2025 10-Ks (P).
- Not found: moderator headcount or T&S spending for Bumble.

## Suggested numbers for the report and slides

1. **The need:** 48% of US online daters faced at least one form of harassment, including 24% called offensive names (Pew 2023). In addition, 53% of women surveyed in Australia experienced gaslighting on dating platforms (BIT/Minderoo 2026).
2. **The market:** dating apps had more than $6B in revenue in 2024 (Business of Apps). Match Group alone had $3.49B in 2025 and says it spends $125M a year on trust and safety.
3. **Why now:** the UK Online Safety Act puts dating services in scope and lists controlling or coercive behaviour as illegal content, and the EU DSA requires report handling with explanations and appeals.

## References (APA, check before use)

- Behavioural Insights Team. (2026). *Understanding tech-based abuse of women on dating platforms: Final report*. Minderoo Foundation. https://www.bi.team/wp-content/uploads/2026/09/Understanding-tech-based-abuse-of-women-on-dating-platforms.pdf
- Bumble Inc. (2026). *Annual report on Form 10-K for the fiscal year ended December 31, 2025*. U.S. Securities and Exchange Commission.
- Department for Science, Innovation and Technology. (2025). *Online Safety Act: Explainer*. GOV.UK. https://www.gov.uk/government/publications/online-safety-act-explainer/online-safety-act-explainer
- Match Group. (2023, November 14). *Match Group announces "Are You Sure?" to expand to millions of users across Match, Stir, OurTime & more* [Press release]. https://mtch.com/?p=1332
- Match Group. (2026, February 3). *Match Group announces fourth quarter and full-year results* [Press release].
- Pew Research Center. (2023, February 2). *The experiences of U.S. online daters*. https://www.pewresearch.org/internet/2023/02/02/the-experiences-of-u-s-online-daters/ (copy the authors from the page)
- Regulation (EU) 2016/679 (General Data Protection Regulation).
- Regulation (EU) 2022/2065 (Digital Services Act).
