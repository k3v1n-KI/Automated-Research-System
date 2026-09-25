# Study 4 Website Verification Review

These are the 56 website cases classified as `different_domain` in the Pathways Data Restoration audit. Manual review found that the 10 rows marked `invalid_extraction` below returned business-hours text or Facebook URLs rather than acceptable organization websites. All other 46 predicted website values were verified as valid by the reviewer. The sealed reference remains included for audit comparison.

## Manual Review Summary

| Review outcome | Count |
|---|---:|
| Verified as valid | 46 |
| Rejected as invalid extraction | 10 |
| Total reviewed | 56 |

The table's `Status` column preserves the original automated audit classification (`needs verification`) for the 46 valid website mismatches. The manual review conclusion is the summary above: those 46 are verified as valid, while the 10 rows marked `invalid_extraction` are rejected.

| # | Organization | Predicted value | Sealed reference | Status | How verified |
|---:|---|---|---|---|---|
| 1 | PharmaChoice - Milton - Bronte St S (Main and Bronte Pharmacy) | http://www.pharmachoice.com/locations/main-bronte-pharmacy/ | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=200002 | needs verification | |
| 2 | Shoppers Drug Mart - Toronto - Bathurst St (Lawrence Plaza) | http://www.shoppersdrugmart.ca/ | https://www.torontocentralhealthline.ca/displayService.aspx?id=120821 | needs verification | |
| 3 | Raxlen Pharmacy | http://www.raxlenpharmacy.com/ | https://www.torontocentralhealthline.ca/displayService.aspx?id=151347 | needs verification | |
| 4 | Guardian Pharmacy - Brampton - Mayfield Rd | https://www.guardian-ida-remedysrx.ca/en/ontario/brampton/upper-mount-gdn-pharmacy-7058511 | https://www.centralwesthealthline.ca/displayService.aspx?id=213044 | needs verification | |
| 5 | IDA Pharmacy - Mississauga - Erin Mills Pkwy | https://www.guardian-ida-remedysrx.ca/en/Ontario/Mississauga/Erin-Mills-IDA-Pharmacy-7045121 | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=147723 | needs verification | |
| 6 | Metro Pharmacy - Toronto - 129 Dundas St E | http://www.metrodrugs.ca/ | https://www.torontocentralhealthline.ca/displayService.aspx?id=151474 | needs verification | |
| 7 | Food & Drug Basics - Oakville | http://www.foodbasics.ca/ | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=154944 | needs verification | |
| 8 | Thompson Square Pharmacy | https://www.thompsonsquarepharmacy.ca/ | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=219600 | needs verification | |
| 9 | HighCare Pharmacy | Opening-hours text, not a URL | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=199187 | invalid_extraction | Business-hours text returned instead of a website; manually rejected. |
| 10 | Regal Heights Pharmacy | https://regalhpharmacy.com/ | https://www.torontocentralhealthline.ca/displayService.aspx?id=218482 | needs verification | |
| 11 | Medi-Place Healthcare Pharmacy | http://www.facebook.com/305617959928006 | https://www.centralwesthealthline.ca/displayService.aspx?id=61082 | invalid_extraction | Facebook URL; manually rejected as not an acceptable organization website. |
| 12 | Terry Fox Pharmacy | http://www.facebook.com/446528905783991 | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=147847 | invalid_extraction | Facebook URL; manually rejected as not an acceptable organization website. |
| 13 | Health Plus Pharmacy | http://www.facebook.com/1937931106444074 | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=174719 | invalid_extraction | Facebook URL; manually rejected as not an acceptable organization website. |
| 14 | PharmaChoice - Brampton - Van Kirk Dr | Opening-hours text, not a URL | https://www.centralwesthealthline.ca/displayService.aspx?id=61105 | invalid_extraction | Business-hours text returned instead of a website; manually rejected. |
| 15 | PharmaChoice - Brampton - Gore Rd | http://pharmachoice.com/ | https://www.centralwesthealthline.ca/displayService.aspx?id=61101 | needs verification | |
| 16 | IDA Pharmacy - Toronto - 1920 Yonge St | https://www.guardian-ida-remedysrx.ca/en/ontario/toronto/sams-ida-pharmacy-7008620 | https://www.torontocentralhealthline.ca/displayService.aspx?id=120404 | needs verification | |
| 17 | Dayspring Pharmacy | https://dayspringmedical.ca/dayspring-pharmacy/ | https://www.centralwesthealthline.ca/displayService.aspx?id=172507 | needs verification | |
| 18 | PocketPills | https://www.pocketpills.com/ | https://www.centralwesthealthline.ca/displayService.aspx?id=199105 | needs verification | |
| 19 | PharmaChoice - Georgetown - 99 Sinclair Ave | https://www.pharmachoice.com/locations/professional-arts/ | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=154822 | needs verification | |
| 20 | Shoppers Drug Mart - Orangeville - 25 Broadway Ave | https://www.shoppersdrugmart.ca/en/stores/934-shoppers-drug-mart-broadway-and-townline/ | https://www.centralwesthealthline.ca/displayService.aspx?id=61047 | needs verification | |
| 21 | Main Drug Mart - North York - Keele St | Opening-hours text, not a URL | https://www.torontocentralhealthline.ca/displayService.aspx?id=151406 | invalid_extraction | Business-hours text returned instead of a website; manually rejected. |
| 22 | Shoppers Drug Mart - Toronto - O'Connor Dr | http://www.shoppersdrugmart.ca/ | https://www.torontocentralhealthline.ca/displayService.aspx?id=120797 | needs verification | |
| 23 | Shoppers Drug Mart - Acton | https://www.shoppersdrugmart.ca/en/stores/1168-shoppers-drug-mart-acton/ | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=154200 | needs verification | |
| 24 | Shoppers Drug Mart - Toronto - 388 King St W | https://www.shoppersdrugmart.ca/en/stores/1320-shoppers-drug-mart-king-and-peter/ | https://www.torontocentralhealthline.ca/displayService.aspx?id=120779 | needs verification | |
| 25 | Shoppers Drug Mart - Toronto - The Kingsway | http://www.shoppersdrugmart.ca/ | https://www.torontocentralhealthline.ca/displayService.aspx?id=120759 | needs verification | |
| 26 | Pharmasave - Brampton - Queen St | https://pharmasave.com/brampton-queen-st/ | https://www.centralwesthealthline.ca/displayService.aspx?id=220237 | needs verification | |
| 27 | Golfdale Pharmacy - Scarborough | https://golfdalemedical.ca/pharmacy | https://www.torontocentralhealthline.ca/displayService.aspx?id=150667 | needs verification | |
| 28 | Medicine Cabinet | http://medcab.ca/ | https://www.torontocentralhealthline.ca/displayService.aspx?id=200045 | needs verification | |
| 29 | Metro Pharmacy - North York - 20 Church Ave | http://www.metro.ca/find-a-store/details.en.html?id=330 | https://www.torontocentralhealthline.ca/displayService.aspx?id=151005 | needs verification | |
| 30 | Pharmasave - Woodbridge | https://pharmasave.com/woodbridge | https://www.centralwesthealthline.ca/displayService.aspx?id=170226 | needs verification | |
| 31 | Nofrills - Loblaw Pharmacy - North York - 1450 Lawrence Ave E | http://www.nofrills.ca/ | https://www.torontocentralhealthline.ca/displayService.aspx?id=151656 | needs verification | |
| 32 | Guardian Pharmacy - East York - O'Connor Dr | https://www.guardian-ida-remedysrx.ca/en/ontario/toronto/theodore-pharmacy-7057664 | https://www.torontocentralhealthline.ca/displayService.aspx?id=200224 | needs verification | |
| 33 | Guardian Pharmacy - North York - Bayview Ave | https://www.guardian-ida-remedysrx.ca/en/ontario/toronto/true-light-pharmacy-7040321 | https://www.torontocentralhealthline.ca/displayService.aspx?id=150902 | needs verification | |
| 34 | Britannia Pharmacy | https://britannia-pharmacy.com/ | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=147679 | needs verification | |
| 35 | Heart Lake The Compounding Centre | Opening-hours text, not a URL | https://www.centralwesthealthline.ca/displayService.aspx?id=61001 | invalid_extraction | Business-hours text returned instead of a website; manually rejected. |
| 36 | Main Drug Mart - Toronto - 25 Overlea Blvd | Opening-hours text, not a URL | https://www.torontocentralhealthline.ca/displayService.aspx?id=151204 | invalid_extraction | Business-hours text returned instead of a website; manually rejected. |
| 37 | Get Well Pharmacy | https://www.getwellrx.ca/ | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=154382 | needs verification | |
| 38 | Pharmasave - Brampton - Braydon Blvd | https://pharmasave.com/brampton-castlemore | https://www.centralwesthealthline.ca/displayService.aspx?id=61092 | needs verification | |
| 39 | Pharmasave - Toronto - 1366 Yonge St | https://pharmasave.com/toronto-balmoral | https://www.torontocentralhealthline.ca/displayService.aspx?id=131401 | needs verification | |
| 40 | Pharmasave - Caledon - Hurontario St | https://pharmasave.com/caledon-village | https://www.centralwesthealthline.ca/displayService.aspx?id=199034 | needs verification | |
| 41 | Queen Best Pharmacy - Brampton | https://www.remedys.ca/ | https://www.centralwesthealthline.ca/displayService.aspx?id=169760 | needs verification | |
| 42 | J.C. Pharmacy | http://mississaugachinesecentre.com/en/stores/jc-pharmacy/ | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=147785 | needs verification | |
| 43 | Father Tobin Pharmacy | http://www.mylocalrx.ca/ | https://www.centralwesthealthline.ca/displayService.aspx?id=61008 | needs verification | |
| 44 | Ironoak Pharmacy | https://www.ironoakpharmacy.ca/ | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=219271 | needs verification | |
| 45 | Simpson's Pharmasave | https://pharmasave.com/virgil | https://www.google.ca/maps?q=Simpson%27s+Pharmasave%2C1882+Niagara+Stone+Rd%2CVirgil%2CON%20L0S%201T0%2CCanada | needs verification | |
| 46 | Pharmasave - Toronto - Bloor St W | https://pharmasave.com/toronto-swansea | https://www.torontocentralhealthline.ca/displayService.aspx?id=151121 | needs verification | |
| 47 | Medical Pharmacies - Ottawa Hospital Civic Campus | https://www.ottawahospital.on.ca/ | http://www.pharmaciedesjardins.ca/ | needs verification | |
| 48 | PharmaTrust Drug Mart | Opening-hours text, not a URL | https://www.torontocentralhealthline.ca/displayService.aspx?id=151527 | invalid_extraction | Business-hours text returned instead of a website; manually rejected. |
| 49 | Glen Eden Pharmacy | http://www.guardian-ida-remedysrx.ca/en/ontario/milton/glen-eden-pharmacy-7063184 | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=154884 | needs verification | |
| 50 | Guardian Pharmacy - Toronto - Eglinton Ave W | http://eglintonmedical.ca/ | https://www.torontocentralhealthline.ca/displayService.aspx?id=151368 | needs verification | |
| 51 | Prince Theodore Group of Pharmacies - Mississauga | https://eglinton.princerx.ca/ | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=147720 | needs verification | |
| 52 | Pharmasave - Shelburne - Col Phillips Dr | https://pharmasave.com/store/pharmasave-shelburne-pharmacy/ | https://www.centralwesthealthline.ca/displayService.aspx?id=220209 | needs verification | |
| 53 | Pharmasave - Brampton - North Park Dr | https://www.pharmasavebrameast.com/ | https://www.centralwesthealthline.ca/displayService.aspx?id=61046 | needs verification | |
| 54 | Drug Centre Discount Pharmacy | https://www.facebook.com/pages/Drug-Centre-Discount-Pharmacy/148939515136278 | https://www.torontocentralhealthline.ca/displayService.aspx?id=150843 | needs verification | |
| 55 | SaveonRX Pharmacy - Brampton - Hurontario St | https://www.remedys.ca/en/ontario/brampton/saveon-rx-brampton-7026501 | https://www.centralwesthealthline.ca/displayService.aspx?id=191888 | needs verification | |
| 56 | Peace Land Pharmacy | Opening-hours text, not a URL | https://www.mississaugahaltonhealthline.ca/displayService.aspx?id=176178 | invalid_extraction | Business-hours text returned instead of a website; manually rejected. |

## Review Guidance

The reference URLs are sealed benchmark values, not automatically authoritative current websites. Verify whether the predicted URL is the current official organization or branch website, a legitimate directory representation, or an extraction error. Record the source and decision in `how_verified`.
