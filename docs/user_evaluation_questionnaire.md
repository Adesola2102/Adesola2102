# Non-Expert User Evaluation Questionnaire

**Purpose:** Collect the perceived-credibility and decision-making-speed
evidence for Objective V ("Evaluate the readability, perceived credibility,
and impact on decision-making speed of the generated plain language
explanations by comparing them with raw SHAP/LIME explanations among
non-expert users") that cannot be produced automatically and requires real
participants, as described in Chapter Three's Demonstration/Evaluation
phases of the DSRM methodology.

**Suggested participants:** 8-12 non-expert participants matching the
target user profile named throughout the report - IT helpdesk staff, SOC
analysts without a statistics/data-science background, or general
employees. Recruit a mix if possible.

**How to run it:** For each participant, run the Streamlit app
(`streamlit run app/streamlit_app.py`) live, or use the pre-generated
example pairs in `reports/evaluation_examples.json`. Show each participant
BOTH the raw SHAP/LIME output (e.g. the bar chart + feature table) and the
plain-language sentence for the same 6-8 predictions (mix of phishing and
legitimate, mix of email and URL), in randomised order, and time how long it
takes them to reach a "quarantine / do not quarantine" decision from each
representation.

---

## Part A - Per-example ratings

For each example shown, ask the participant to rate the statements below on
a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree),
**once for the raw SHAP/LIME view and once for the plain-language view**.

| # | Statement |
|---|---|
| 1 | I immediately understood why the system reached this verdict. |
| 2 | I feel confident this verdict is correct. |
| 3 | I would know what action to take (quarantine / allow) based on this explanation alone. |
| 4 | This explanation used language I am comfortable with. |
| 5 | I trust this explanation more than a verdict with no explanation at all. |

Record, per example and per view:

- Time-to-decision (seconds, stopwatch from "explanation shown" to
  "participant states their action").
- The five Likert ratings above.

## Part B - Overall comparison (after all examples)

1. Which explanation style did you find easier to understand overall - the
   raw SHAP/LIME output, or the plain-language sentence? (forced choice)
2. Which style would you trust more in a real incident, under time
   pressure? (forced choice)
3. On a scale of 1-5, how much data-science/statistics background do you
   feel this task required with the raw SHAP/LIME view?
4. On a scale of 1-5, how much data-science/statistics background do you
   feel this task required with the plain-language view?
5. Open comment: what, if anything, felt misleading or oversimplified about
   the plain-language explanation?

## Part C - Participant background (collected once)

- Job role / years of IT or security experience.
- Prior familiarity with SHAP, LIME, or model explainability concepts
  (none / heard of it / used it before).

---

## Analysis plan

- Compare mean time-to-decision (raw vs. plain-language) with a paired
  t-test or Wilcoxon signed-rank test (n is small, so report both the mean
  difference and effect size, not just a p-value).
- Compare mean Likert ratings per statement (raw vs. plain-language) the
  same way.
- Report the forced-choice tallies from Part B directly (e.g. "9 of 11
  participants preferred the plain-language explanation").
- Combine with the automated readability metrics in
  `reports/evaluation_report.md` to give a complete answer to Objective V:
  readability formulas provide the objective linguistic-complexity evidence;
  this questionnaire provides the subjective credibility/speed evidence the
  formulas cannot capture.

> This instrument was designed but not administered within the scope of
> this build: assembling a panel of real IT-helpdesk/SOC-analyst
> participants requires recruitment and ethics/consent steps outside an
> automated development session. It is provided ready for the student to
> run with real participants and report the results in Chapter Five /
> Appendices, exactly as Objective V and the DSRM evaluation phase require.
