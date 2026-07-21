# Objective V Evaluation Report

Automated comparison of the LLM Translation Layer's plain-language sentences against the raw SHAP feature-importance output they were derived from, using standard readability formulas (Flesch Reading Ease: higher = easier; Flesch-Kincaid Grade: lower = easier).

## URL classifier sample (n=30)

- Sample accuracy on this evaluation batch: **86.7%**
- Flesch Reading Ease — raw SHAP text: **13.9** vs translated sentence: **34.6** (higher is easier to read)
- Flesch-Kincaid Grade Level — raw SHAP text: **11.9** vs translated sentence: **16.8** (lower is easier to read)
- Average length — raw SHAP text: **14.0 words** vs translated sentence: **33.8 words**

## Email classifier sample (n=30)

- Sample accuracy on this evaluation batch: **80.0%**
- Flesch Reading Ease — raw SHAP text: **-38.9** vs translated sentence: **37.3** (higher is easier to read)
- Flesch-Kincaid Grade Level — raw SHAP text: **19.3** vs translated sentence: **14.7** (lower is easier to read)
- Average length — raw SHAP text: **14.0 words** vs translated sentence: **26.8 words**

## Example side-by-side comparisons

### URL examples

**Input:** `http://nrgbinary.com/`  
**True label:** legitimate | **Predicted:** legitimate (93.4%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.934, base_value=0.500, top_features=(num_subdomains: -0.108, url_length: -0.070, domain_length: -0.058, domain_entropy: -0.055, num_dots: -0.052)  
**Plain-language translation:** This web link looks LEGITIMATE with 93% confidence: it has a simple, normal domain structure, has no unusual hyphens in the domain, and does not mix unusual digits into the web address.

**Input:** `http://cyclingnews.com/`  
**True label:** legitimate | **Predicted:** legitimate (84.4%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.844, base_value=0.500, top_features=(num_subdomains: -0.112, url_length: -0.056, num_dots: -0.047, domain_entropy: -0.034, num_digits: -0.023)  
**Plain-language translation:** This web link looks LEGITIMATE with 84% confidence: it has a simple, normal domain structure, uses a common, mainstream domain ending, and does not mix unusual digits into the web address.

**Input:** `https://glide.me/`  
**True label:** legitimate | **Predicted:** legitimate (93.0%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.930, base_value=0.500, top_features=(url_length: -0.100, domain_length: -0.090, num_subdomains: -0.088, num_dots: -0.039, domain_entropy: -0.037)  
**Plain-language translation:** This web link looks LEGITIMATE with 93% confidence: it has a simple, normal domain structure, uses a common, mainstream domain ending, and has no unusual hyphens in the domain.

**Input:** `https://epnnazri.org/`  
**True label:** legitimate | **Predicted:** legitimate (90.6%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.906, base_value=0.500, top_features=(num_subdomains: -0.108, url_length: -0.068, domain_length: -0.060, num_dots: -0.047, domain_entropy: -0.044)  
**Plain-language translation:** This web link looks LEGITIMATE with 91% confidence: it has a simple, normal domain structure, uses a common, mainstream domain ending, and has no unusual hyphens in the domain.

**Input:** `https://rtbhouse.com/`  
**True label:** legitimate | **Predicted:** legitimate (87.6%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.876, base_value=0.500, top_features=(num_subdomains: -0.100, url_length: -0.066, domain_length: -0.055, num_dots: -0.045, domain_entropy: -0.036)  
**Plain-language translation:** This web link looks LEGITIMATE with 88% confidence: it has a simple, normal domain structure, uses a common, mainstream domain ending, and does not mix unusual digits into the web address.

### Email examples

**Input:** `re : grades`  
**True label:** legitimate | **Predicted:** phishing (76.4%) — *MISCLASSIFIED*  
**Raw SHAP output:** [SHAP] predicted_class=phishing, confidence=0.764, base_value=0.500, top_features=(exclamation_count: +0.163, word_count: +0.115, non_alpha_ratio: -0.051, attachment_keyword: +0.050, urgent_keyword_count: -0.015)  
**Plain-language translation:** This email was flagged as PHISHING with 76% confidence because it uses excessive exclamation marks to create urgency.

**Input:** `wheelabrator deal 6 / 17 / 01 - 06 / 18 / 01`  
**True label:** legitimate | **Predicted:** legitimate (79.2%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.792, base_value=0.500, top_features=(exclamation_count: -0.157, word_count: -0.076, non_alpha_ratio: -0.048, attachment_keyword: +0.040, urgent_keyword_count: -0.030)  
**Plain-language translation:** This email looks LEGITIMATE with 79% confidence: it is written in a calm, measured tone, does not use urgent or pressuring language, and has a normal amount of message content.

**Input:** `darden case study on " the transformation of enron "`  
**True label:** legitimate | **Predicted:** legitimate (94.5%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.945, base_value=0.500, top_features=(attachment_keyword: -0.286, word_count: -0.111, non_alpha_ratio: -0.084, exclamation_count: +0.079, urgent_keyword_count: -0.019)  
**Plain-language translation:** This email looks LEGITIMATE with 94% confidence: it has a normal amount of message content, does not use urgent or pressuring language, and does not ask for a password or other credentials.

**Input:** `component var`  
**True label:** legitimate | **Predicted:** legitimate (58.4%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.584, base_value=0.500, top_features=(exclamation_count: -0.144, attachment_keyword: +0.049, non_alpha_ratio: +0.033, urgent_keyword_count: -0.030, word_count: +0.008)  
**Plain-language translation:** This email looks LEGITIMATE with 58% confidence: it is written in a calm, measured tone and does not use urgent or pressuring language.

**Input:** `asset sales chart`  
**True label:** legitimate | **Predicted:** legitimate (93.8%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.938, base_value=0.500, top_features=(attachment_keyword: -0.479, exclamation_count: +0.089, urgent_keyword_count: -0.027, reward_keyword_count: -0.013, financial_keyword_count: -0.009)  
**Plain-language translation:** This email looks LEGITIMATE with 94% confidence: it does not use urgent or pressuring language, does not promise a prize, refund, or reward, and does not ask for a password or other credentials.


> **Note on scope:** these readability metrics are an objective, reproducible proxy for surface-level linguistic complexity. They do not, by themselves, measure perceived credibility or decision-making speed - those dimensions of Objective V require human participants and are covered by the structured questionnaire in `docs/user_evaluation_questionnaire.md`, intended for administration with real non-expert participants (IT helpdesk staff / general users) as described in Chapter Three.