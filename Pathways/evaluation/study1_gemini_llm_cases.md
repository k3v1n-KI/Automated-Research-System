# Study 1 Gemini LLM Cases

This appendix summarizes the 33 cases recorded by the Study 1 evaluator as LLM-involved in `study1_semantic_llm_gemini_report.json`.

## Interpretation

- **LLM-involved:** The evaluator entered the Gemini fallback path for the case because unresolved or ambiguous language was present.
- **Raw response:** The JSON returned by Gemini before ontology validation.
- **Validated proposal:** The response after `_validate_llm_suggestion` retained only values permitted by the Pathways ontology.
- **Accepted:** At least one permitted value remained after validation. Accepted means ontology-valid, not necessarily correct according to the gold label.
- **Rejected/abstained:** Gemini returned `{}`, no usable value survived validation, or the proposal was absent. In these cases, no LLM proposal was merged.
- **Final-result status:** Whether the final extracted filters matched the gold filter. A validated proposal can be correct, redundant, or incorrect.

Summary: 33 recorded LLM-involved cases, 22 accepted validated proposals, and 11 rejected or abstained responses. The accepted proposals were generally redundant with deterministic or semantic extraction; `p19` is the exception where the final population value remained incorrect.

## Case Table

| ID | Query | Raw Gemini response | Validated proposal | Status | Result and reason |
|---|---|---|---|---|---|
| c02 | substance use support Ottawa | `{"need": ["addiction_services"]}` | `{"need": ["addiction_services"]}` | Accepted | Final output matches gold; proposal supplies the supported addiction concept. |
| c05 | walk-in care Toronto | `{"modality": ["walk-in"]}` | `{"modality": ["walk-in"]}` | Accepted | Final output matches gold; proposal confirms the walk-in modality. |
| c06 | same-day addiction help Brampton | `{"modality": ["same-day"]}` | `{"modality": ["same-day"]}` | Accepted | Final output matches gold; proposal confirms same-day access. |
| c08 | Mandarin primary care Toronto | `{}` | `null` | Rejected/abstained | Gemini abstained on the unresolved word “primary”; deterministic extraction already produced the correct language and location. |
| c09 | Cantonese walk-in care Richmond Hill | `{}` | `null` | Rejected/abstained | Gemini abstained; deterministic aliases already produced the correct walk-in, language, and location filters. |
| c10 | Ojibwe hospital services Thunder Bay | `{"language": ["Ojibwe"], "need": ["hospitals"]}` | `{"need": ["hospitals"], "language": ["Ojibwe"]}` | Accepted | Final output matches gold; both returned values are ontology-valid. |
| c12 | youth mental health Scarborough | `{}` | `null` | Rejected/abstained | Gemini abstained on “mental health”; the explicit youth alias already produced the correct result. |
| c15 | OHIP walk-in care Toronto | `{"modality": ["walk-in"]}` | `{"modality": ["walk-in"]}` | Accepted | Final output matches gold; proposal confirms the modality. |
| c17 | same day French care Ottawa | `{"modality": ["same-day"]}` | `{"modality": ["same-day"]}` | Accepted | Final output matches gold; proposal confirms same-day access. |
| c18 | in person Mandarin services Toronto | `{}` | `null` | Rejected/abstained | Gemini abstained on the residual “person”; the deterministic in-person alias already resolved the field. |
| c19 | walk in adolescent care Vaughan | `{"modality": "walk-in"}` | `{"modality": ["walk-in"]}` | Accepted | Final output matches gold; scalar output was normalized to an accepted array. |
| c20 | French OHIP primary care Toronto | `{}` | `null` | Rejected/abstained | Gemini abstained on “primary”; deterministic language, coverage, and location extraction was sufficient. |
| c21 | hospital for seniors Ottawa | `{"population": ["senior"]}` | `{"population": ["senior"]}` | Accepted | Final output matches gold; proposal resolves the plural “seniors” to the ontology value. |
| c25 | same-day hospital care Mississauga | `{"modality": ["same-day"]}` | `{"modality": ["same-day"]}` | Accepted | Final output matches gold; proposal confirms same-day access. |
| p02 | Where can I find help for drug use near Ottawa? | `{"need": ["addiction_services"]}` | `{"need": ["addiction_services"]}` | Accepted | Final output matches gold; proposal maps drug use to addiction services. |
| p05 | Is there a clinic that accepts walk in patients in Toronto? | `{"modality": ["walk-in"]}` | `{"modality": ["walk-in"]}` | Accepted | Final output matches gold; proposal identifies walk-in access. |
| p06 | I need help today with substance use in Brampton | `{"need": ["addiction_services"]}` | `{"need": ["addiction_services"]}` | Accepted | Final output matches gold; alias and proposal identify addiction services. |
| p07 | French-speaking support I can access online in Ottawa | `{"language": ["French"]}` | `{"language": ["French"]}` | Accepted | Final output matches gold; proposal confirms the language. |
| p08 | My parent needs care in Mandarin in Toronto | `{}` | `null` | Rejected/abstained | Gemini abstained on the residual “parent”; the deterministic “my parent” alias already mapped to senior. |
| p09 | Cantonese help for a walk-in appointment in Richmond Hill | `{"modality": ["walk-in"]}` | `{"modality": ["walk-in"]}` | Accepted | Final output matches gold; proposal identifies walk-in access. |
| p10 | A hospital serving Indigenous language speakers in Thunder Bay | `{"language": ["Ojibwe"]}` | `{"language": ["Ojibwe"]}` | Accepted | Final output matches gold; proposal maps the fixture’s Indigenous-language case to Ojibwe. |
| p11 | Support for an older adult in Toronto | `{"population": ["senior"]}` | `{"population": ["senior"]}` | Accepted | Final output matches gold; proposal confirms the senior population. |
| p12 | Mental health help for a teenager in Scarborough | `{}` | `null` | Rejected/abstained | Gemini abstained; the deterministic teenager alias already produced youth. |
| p15 | A walk-in provider in Toronto that takes OHIP | `{"modality": ["walk-in"]}` | `{"modality": ["walk-in"]}` | Accepted | Final output matches gold; proposal confirms walk-in access. |
| p16 | Medication help in Windsor even without OHIP | `{"coverage": ["no-ohip"]}` | `{"coverage": ["no-ohip"]}` | Accepted | Final output matches gold; proposal resolves the residual coverage wording. |
| p17 | French care needed today in Ottawa | `{}` | `null` | Rejected/abstained | Gemini abstained on “needed”; deterministic language and today-to-same-day extraction was sufficient. |
| p18 | Mandarin services delivered in person in Toronto | `{"modality": ["in-person"]}` | `{"modality": ["in-person"]}` | Accepted | Final output matches gold; proposal confirms in-person access. |
| p19 | A walk-in clinic for a teenager in Vaughan | `{"modality": ["walk-in"]}` | `{"modality": ["walk-in"]}` | Accepted | Walk-in result matches, but the final population remains youth while gold expects adolescent. The accepted proposal did not address that population ambiguity. |
| p20 | French primary care that accepts OHIP in Toronto | `{}` | `null` | Rejected/abstained | Gemini abstained on residual wording; deterministic language, coverage, and location extraction was sufficient. |
| p21 | A hospital for an older person in Ottawa | `{"population": ["senior"]}` | `{"population": ["senior"]}` | Accepted | Final output matches gold; proposal confirms the senior population. |
| p22 | Online help for a young person in Brampton | `{"population": ["youth"]}` | `{"population": ["youth"]}` | Accepted | Final output matches gold; proposal confirms youth. |
| p23 | Mandarin help getting medication in Markham | `{}` | `null` | Rejected/abstained | Gemini abstained on “getting”; deterministic aliases and semantic matching already produced the correct result. |
| p24 | Cantonese care in Toronto for someone without OHIP | `{}` | `null` | Rejected/abstained | Gemini abstained on “without”; the deterministic “without OHIP” alias already produced the correct coverage value. |

## What Acceptance Means

The validator checks proposed fields against the allowed ontology values. It accepts values such as `walk-in`, `same-day`, `addiction_services`, `senior`, and `no-ohip`, but it rejects unsupported values or malformed fields. It does not determine whether a proposal is semantically correct for the query; correctness is assessed separately against the gold filter.

The results therefore show three distinct outcomes:

1. Gemini proposed an ontology-valid value and the final result was correct.
2. Gemini abstained because it could not confidently resolve the residual wording; deterministic extraction was retained.
3. Gemini proposed a valid value, but the remaining query ambiguity still caused a gold mismatch, as in `p19`.
