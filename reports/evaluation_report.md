# Objective V Evaluation Report

Automated comparison of the LLM Translation Layer's plain-language sentences against the raw SHAP feature-importance output they were derived from, using standard readability formulas (Flesch Reading Ease: higher = easier; Flesch-Kincaid Grade: lower = easier).

## URL classifier sample (n=30)

- Sample accuracy on this evaluation batch: **100.0%**
- Flesch Reading Ease — raw SHAP text: **8.7** vs translated sentence: **36.4** (higher is easier to read)
- Flesch-Kincaid Grade Level — raw SHAP text: **12.6** vs translated sentence: **16.2** (lower is easier to read)
- Average length — raw SHAP text: **14.0 words** vs translated sentence: **32.2 words**

## Email classifier sample (n=30)

- Sample accuracy on this evaluation batch: **93.3%**
- Flesch Reading Ease — raw SHAP text: **-51.3** vs translated sentence: **25.2** (higher is easier to read)
- Flesch-Kincaid Grade Level — raw SHAP text: **21.0** vs translated sentence: **17.4** (lower is easier to read)
- Average length — raw SHAP text: **14.0 words** vs translated sentence: **31.0 words**

## Example side-by-side comparisons

### URL examples

**Input:** `https://freberg.westnet.com/`  
**True label:** legitimate | **Predicted:** legitimate (86.5%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.865, base_value=0.499, top_features=(num_subdomains: +0.147, path_length: -0.087, digit_ratio: -0.078, url_length: -0.075, num_digits: -0.063)  
**Plain-language translation:** This web link looks LEGITIMATE with 87% confidence: it is not dominated by digits, uses a common, mainstream domain ending, and does not mix unusual digits into the web address.

**Input:** `https://www.uvm.edu/~ofabweb/`  
**True label:** legitimate | **Predicted:** legitimate (99.8%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.998, base_value=0.499, top_features=(num_dots: -0.088, num_subdomains: -0.065, digit_ratio: -0.061, domain_entropy: -0.061, url_length: -0.058)  
**Plain-language translation:** This web link looks LEGITIMATE with 100% confidence: it has a simple, normal domain structure, is not dominated by digits, and uses a common, mainstream domain ending.

**Input:** `http://www.geocities.com/kurisumasu_jbt/`  
**True label:** legitimate | **Predicted:** legitimate (99.6%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.996, base_value=0.499, top_features=(num_subdomains: -0.095, num_dots: -0.077, digit_ratio: -0.058, domain_length: -0.048, num_digits: -0.048)  
**Plain-language translation:** This web link looks LEGITIMATE with 100% confidence: it has a simple, normal domain structure, is not dominated by digits, and uses a common, mainstream domain ending.

**Input:** `http://www.jasonmeador.com/`  
**True label:** legitimate | **Predicted:** legitimate (99.2%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.992, base_value=0.499, top_features=(num_dots: -0.085, num_subdomains: -0.080, digit_ratio: -0.070, num_digits: -0.060, url_length: -0.054)  
**Plain-language translation:** This web link looks LEGITIMATE with 99% confidence: it has a simple, normal domain structure, is not dominated by digits, and does not mix unusual digits into the web address.

**Input:** `https://www.uvm.edu/~uvmsbf/`  
**True label:** legitimate | **Predicted:** legitimate (99.8%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.998, base_value=0.499, top_features=(num_dots: -0.088, num_subdomains: -0.065, domain_entropy: -0.062, digit_ratio: -0.061, url_length: -0.058)  
**Plain-language translation:** This web link looks LEGITIMATE with 100% confidence: it has a simple, normal domain structure, is not dominated by digits, and does not mix unusual digits into the web address.

### Email examples

**Input:** `re :`  
**True label:** legitimate | **Predicted:** legitimate (88.5%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.885, base_value=0.500, top_features=(all_caps_word_ratio: -0.309, financial_keyword_count: +0.098, urgent_keyword_count: -0.060, non_alpha_ratio: -0.049, link_to_word_ratio: -0.021)  
**Plain-language translation:** This email looks LEGITIMATE with 89% confidence: it does not rely on capital letters for emphasis, addresses the recipient normally rather than with a generic greeting, and does not use urgent or pressuring language.

**Input:** `re : project tracking database access`  
**True label:** legitimate | **Predicted:** legitimate (96.2%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.962, base_value=0.500, top_features=(all_caps_word_ratio: -0.263, non_alpha_ratio: -0.077, credential_keyword_count: +0.077, urgent_keyword_count: -0.070, word_count: -0.036)  
**Plain-language translation:** This email looks LEGITIMATE with 96% confidence: it does not rely on capital letters for emphasis, does not use urgent or pressuring language, and addresses the recipient normally rather than with a generic greeting.

**Input:** `fw : up - dated plane schedule`  
**True label:** legitimate | **Predicted:** legitimate (98.1%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.981, base_value=0.500, top_features=(all_caps_word_ratio: -0.244, urgent_keyword_count: -0.074, attachment_keyword: -0.039, financial_keyword_count: -0.025, num_urls_in_body: -0.021)  
**Plain-language translation:** This email looks LEGITIMATE with 98% confidence: it does not rely on capital letters for emphasis, addresses the recipient normally rather than with a generic greeting, and does not use urgent or pressuring language.

**Input:** `fw : interruptible gas forms`  
**True label:** legitimate | **Predicted:** legitimate (97.2%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.972, base_value=0.500, top_features=(all_caps_word_ratio: -0.306, urgent_keyword_count: +0.113, non_alpha_ratio: -0.085, attachment_keyword: -0.068, num_urls_in_body: -0.025)  
**Plain-language translation:** This email looks LEGITIMATE with 97% confidence: it does not rely on capital letters for emphasis, addresses the recipient normally rather than with a generic greeting, and is not unusually link-heavy.

**Input:** `f . o . m . nomination eff . june 1 , 2000`  
**True label:** legitimate | **Predicted:** legitimate (99.4%) — *correct*  
**Raw SHAP output:** [SHAP] predicted_class=legitimate, confidence=0.994, base_value=0.500, top_features=(all_caps_word_ratio: -0.257, urgent_keyword_count: -0.079, non_alpha_ratio: -0.051, financial_keyword_count: -0.028, num_urls_in_body: -0.021)  
**Plain-language translation:** This email looks LEGITIMATE with 99% confidence: it does not rely on capital letters for emphasis, does not use urgent or pressuring language, and addresses the recipient normally rather than with a generic greeting.


> **Note on scope:** these readability metrics are an objective, reproducible proxy for surface-level linguistic complexity. They do not, by themselves, measure perceived credibility or decision-making speed - those dimensions of Objective V require human participants and are covered by the structured questionnaire in `docs/user_evaluation_questionnaire.md`, intended for administration with real non-expert participants (IT helpdesk staff / general users) as described in Chapter Three.