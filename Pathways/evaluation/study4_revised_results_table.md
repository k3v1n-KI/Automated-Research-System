# Study 4 Revised Results Table

This draft replaces the iteration comparison in Table 4.7 with a single **Pathways Data Restoration** breakdown by missing-field category.

Classification follows the manual adjudication rules:

- **Exact:** The restored value matches the sealed reference after normalization.
- **Supported (non-exact):** The value is not identical to the sealed reference, but the manual interpretation supports it as a valid value for the intended organization. This includes phone-number differences plausibly caused by departments, reception desks, or main-office routing; and alternate website representations or sections of the same organization website.
- **Weakly supported:** The value is plausible and related to the intended organization or area, but it requires manual confirmation. For postal codes, this means the code is generally associated with the same area but is not confirmed as the organization’s exact code. For addresses, this includes a plausible alternate branch or location that needs confirmation.
- **Unresolved:** No usable value was restored.

## Proposed Table 4.7

| Missing-field category | Requested fields | Exact | Supported (non-exact) | Weakly supported | Unresolved |
|---|---:|---:|---:|---:|---:|
| Address | 269 | 11 | 201 | 57 | 0 |
| Phone number | 112 | 77 | 34 | 0 | 1 |
| Postal code | 58 | 52 | 0 | 6 | 0 |
| Website | 61 | 2 | 59 | 0 | 0 |
| **Total** | **500** | **142** | **294** | **63** | **1** |

## Classification Notes

### Address

The original audit classified 144 non-exact addresses as high-confidence representations and 57 as likely valid alternatives. Under the revised adjudication, the 57 likely alternatives are promoted to supported where they identify a plausible branch or alternate organizational location in the same city. The remaining 57 materially different addresses are weakly supported and require manual confirmation.

### Phone number

All 34 non-exact phone values are classified as supported for this table. The differences are interpreted as plausible department, reception, main-office, extension, or branch-routing differences. The one returned opening-hours string is unresolved because it is not a phone number. This classification does not claim that every number is the best contact number; it records that the discrepancy is acceptable as an organizational contact variation under the manual review rule.

### Postal code

The six non-exact postal-code values are weakly supported. A postal code can be geographically plausible while still referring to a neighboring building, unit, branch, or nearby service location. Exact organizational postal-code confirmation is therefore required before treating these as exact.

### Website

The 59 non-exact website values are classified as supported. Manual review treats different pages, organizational domains, directory representations, and website sections as valid where they identify the intended organization. The 10 outputs that were business-hours text or Facebook URLs are excluded from this supported classification and should remain invalid extraction/unresolved cases in the detailed audit. The aggregate table therefore requires the website denominator to be reconciled with the detailed review before thesis insertion.

## Important Reconciliation

The raw website mismatch audit contains 59 non-exact website cases, including 10 outputs manually identified as invalid extraction. If those 10 are counted as unresolved rather than supported, the website row becomes:

| Missing-field category | Requested fields | Exact | Supported (non-exact) | Weakly supported | Unresolved |
|---|---:|---:|---:|---:|---:|
| Website, manual-review-adjusted | 61 | 2 | 49 | 0 | 10 |
| **Manual-review-adjusted total** | **500** | **142** | **284** | **63** | **11** |

The first proposed table reflects the requested rule that all non-exact websites count as supported. The second table preserves the explicit manual rejection of the 10 malformed website outputs. The second version is methodologically safer unless those 10 cases are corrected and rerun.
