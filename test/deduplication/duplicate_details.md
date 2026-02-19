# Duplicate Record Analysis Report 
## Dataset Overview
| Metric | Count | Description |
|---|---|---|
| Total Rows | 200 | The total number of records in the dataset. |
| Total Unique Groups (Entities) | 110 | The number of distinct real-world entities. This is the target output size. |
| Unique Records (Singletons) | 91 | Records that appear only once. |
| Duplicate Groups | 19 | Entities that have more than one record. |
| Total Duplicate Rows | 109 | The total count of rows that belong to the 19 duplicate groups. |
| Redundant Rows to Remove | 90 | The number of rows the algorithm needs to delete to reach the optimal state. |

---
### Group G015
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G015 | 93 | Brantford Clinic, A Fresh New Start | 205 Colborne St Brantford, ON, N3T 2G8 (1km) Clipboard | 519-758-5800 | Original (Control) |
| G015 | 101 | Brantford Clinic, A Fresh New Start | 205 Colborne St Brantford, ON, N3T 2G8 (14km) Clipboard | 519-758-5800 | Original (Control) |


### Group G020
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G020 | 30 | Canadian Mental Health Association, Peel Dufferin Branch | 7700 Hurontario St, Suite 314 Brampton, ON, L6Y 4M3 (9km) Clipboard | *Missing* | Original (Control) |
| G020 | 106 | Canadian Mental Health Association, Peel Dufferin Branch | 7700 Hurontario St, Suite 314 Brampton, ON, L6Y 4M3 (10km) Clipboard | *Missing* | Original (Control) |


### Group G021
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G021 | 80 | Canadian Mental Health Assoc.. Simcoe County Branch | 128 Anne St S Barrie, ON, L4N 6A2 (2km) Clipboard | 1-800-461-4319, 705-726-5033, 705-728-5044 | Original (Control) |
| G021 | 136 | Canadian Mental Health Association. Simcoe County Branch | 50 Nottawasaga St Orillia, ON, L3V 3J4 Clipboard | 1-800-461-4319, 705-726-5033, 705-728-5044, 705-325-4499 | **Twisted Name** |


### Group G027
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G027 | 66 | Cornerstone to Recovery | 570 Steven Court, Unit B Newmarket, ON, L3Y 6Z2 (21km) Clipboard | *Missing* | **Phone Removed** |
| G027 | 73 | Cornerstone to Recovery | 570 Steven Court, Unit B Newmarket, ON, L3Y 6Z2 (18km) Clipboard | 905-762-1551, 289-716-9956 | Original (Control) |
| G027 | 120 | Cornerstone to Recovery | 570 Steven Court, Unit B Newmarket, ON, L3Y 6Z2 (23km) Clipboard | *Missing* | **Phone Removed** |


### Group G031
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G031 | 90 | Durham (Region of). Social Services Department | 605 Rossland Rd E Whitby, ON, L1N 6A3 (6km) Clipboard | 3-1-1, 1-888-721-0622 | Original (Control) |
| G031 | 125 | Durham (Region of). Social Svcs Department | Unit 1, 605 Rossland Rd E Whitby, ON, L1N 6A3 (14km) Clipboard | 3-1-1, 1-888-721-0622 | **Twisted Name** |


### Group G038
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G038 | 2 | Grand River Community Health Centre | 363 Colborne St Brantford, ON, N3S 3N2 (1km) Clipboard | 519-754-0777 | Original (Control) |
| G038 | 46 | Grand River Community Health Centre | 363 Colborne St Brantford, ON, N3S 3N2 (14km) Clipboard | *Missing* | **Phone Removed** |
| G038 | 62 | Grand River Community Health Cenrte | 363 Colborne Street Brantford, ON, N3S 3N2 (1km) Clipboard | 519-754-0777 | **Twisted Name** |
| G038 | 119 | Grand River Community Health Centre | 363 Colborne St Brantford, ON, N3S 3N2 (14km) Clipboard | 519-754-0777 | Original (Control) |


### Group G042
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G042 | 9 | Heart Niagara | 4635B Queen St Niagara Falls, ON, L2E 2L7 (14km) Clipboard | *Missing* | **Phone Removed** |
| G042 | 82 | Heart Niagara | 4635B Queen St Niagara Falls, ON, L2E 2L7 (19km) Clipboard | 905-358-5552 | Original (Control) |
| G042 | 117 | Heart Niagara | 4635B Queen St Niagara Falls, ON, L2E 2L7 (16km) Clipboard | *Missing* | **Phone Removed** |
| G042 | 134 | Heart Niagara | 4635B Queen St Niagara Falls, ON, L2E 2L7 (2km) Clipboard | *Missing* | **Phone Removed** |


### Group G044
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G044 | 16 | Humber River Health | 1235 Wilson Ave Toronto, ON, M3M 0B2 (15km) Clipboard | *Missing* | **Phone Removed** |
| G044 | 32 | Humber River Healht | 1235 Wilson Ave Toronto, ON, M3M 0B2 (12km) Clipboard | 416-242-1000 | **Twisted Name** |
| G044 | 70 | HumberR iver Health | Unit 1, 1235 Wilson Ave Toronto, ON, M3M 0B2 (22km) Clipboard | 416-242-1000 | **Twisted Name** |
| G044 | 138 | Humber River Health | 1235 Wilson Ave Toronto, ON, M3M 0B2 (20km) Clipboard | *Missing* | **Phone Removed** |
| G044 | 165 | Humber River Health | 1235 Wilson Ave Toronto, ON, M3M 0B2 (18km) Clipboard | 416-242-1000 | Original (Control) |
| G044 | 181 | Humbre River Health | 1235 Wilson Ave Toronto, ON, M3M 0B2 (19km) Clipboard | 416-242-1000 | **Twisted Name** |


### Group G052
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G052 | 131 | London InterCommunity Health Centre | 659 Dundas St London, ON, N5W 2Z1 (1km) Clipboard | 519-660-0874 | Original (Control) |
| G052 | 170 | London InterCommunity Health Centre | 659 Dundas St London, ON, N5W 2Z1 (24km) Clipboard | 519-660-0874 | Original (Control) |


### Group G069
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G069 | 43 | Ontario Addiction Treatment Centers - Guelph | Unit 1, 50 Cork St E Guelph, ON, N1H 2W8 (21km) Clipboard | 1-877-937-2282, 226-979-2443 | **Twisted Name** |
| G069 | 76 | Ontario Addiction Treatment Centres - Guelph | 50 Cork St E Guelph, ON, N1H 2W8 (22km) Clipboard | 1-877-937-2282, 226-979-2443 | Original (Control) |
| G069 | 98 | Ontario Addiction Treatment Centres - Guelph | 50 Cork St E Guelph, ON, N1H 2W8 (24km) Clipboard | 1-877-937-2282, 226-979-2443 | Original (Control) |
| G069 | 124 | Ontario Addiction Treatment Centres - Guelph | 50 Cork St E Guelph, ON, N1H 2W8 Clipboard | 1-877-937-2282, 226-979-2443 | Original (Control) |


### Group G070
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G070 | 0 | Ontaroi. Ministry of Health | 777 Bay Street, 5th Fl Toronto, ON, M5G 2C8 (502km) Clipboard | 1-866-532-3161 | **Twisted Name** |
| G070 | 6 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (340km) Clipboard | *Missing* | **Phone Removed** |
| G070 | 10 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (400km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 13 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (22km) Clipboard | *Missing* | **Phone Removed** |
| G070 | 18 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (295km) Clipboard | *Missing* | **Phone Removed** |
| G070 | 19 | Otnario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (397km) Clipboard | 1-866-532-3161 | **Twisted Name** |
| G070 | 23 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (25km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 24 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (386km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 25 | Ontaroi. Ministry of Health | 777 Bay Street, 5th Fl Toronto, ON, M5G 2C8 Clipboard | 1-866-532-3161 | **Twisted Name** |
| G070 | 26 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (49km) Clipboard | *Missing* | **Phone Removed** |
| G070 | 27 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | *Missing* | **Phone Removed** |
| G070 | 33 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (241km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 35 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (24km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 39 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (92km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 40 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (22km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 49 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (160km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 55 | Ontario. Ministry ofH ealth | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (93km) Clipboard | 1-866-532-3161 | **Twisted Name** |
| G070 | 57 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | *Missing* | **Phone Removed** |
| G070 | 59 | Onatrio. Ministry of Health | 777 Bay Street, 5th Fl Toronto, ON, M5G 2C8 (57km) Clipboard | 1-866-532-3161 | **Twisted Name** |
| G070 | 61 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 68 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | *Missing* | **Phone Removed** |
| G070 | 78 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 81 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | *Missing* | **Phone Removed** |
| G070 | 85 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (352km) Clipboard | *Missing* | **Phone Removed** |
| G070 | 89 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 92 | Ontario. Ministry of eHalth | 777 Bay Street, 5th Fl Toronto, ON, M5G 2C8 (924km) Clipboard | 1-866-532-3161 | **Twisted Name** |
| G070 | 107 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (313km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 114 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (71km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 116 | Ontario. Mniistry of Health | Unit 1, 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | 1-866-532-3161 | **Twisted Name** |
| G070 | 121 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 128 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | *Missing* | **Phone Removed** |
| G070 | 132 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | *Missing* | **Phone Removed** |
| G070 | 137 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (94km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 144 | Otnario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (68km) Clipboard | 1-866-532-3161 | **Twisted Name** |
| G070 | 148 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (175km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 149 | Onatrio. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (105km) Clipboard | 1-866-532-3161 | **Twisted Name** |
| G070 | 152 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (31km) Clipboard | *Missing* | **Phone Removed** |
| G070 | 155 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (31km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 159 | Ontario. Ministry o fHealth | 777 Bay Street, 5th Fl Toronto, ON, M5G 2C8 (60km) Clipboard | 1-866-532-3161 | **Twisted Name** |
| G070 | 164 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (50km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 166 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (333km) Clipboard | *Missing* | **Phone Removed** |
| G070 | 169 | Ontari.o Ministry of Health | 777 Bay Street, 5th Fl Toronto, ON, M5G 2C8 (170km) Clipboard | 1-866-532-3161 | **Twisted Name** |
| G070 | 171 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (112km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 173 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (85km) Clipboard | *Missing* | **Phone Removed** |
| G070 | 185 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (1km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 186 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (429km) Clipboard | *Missing* | **Phone Removed** |
| G070 | 188 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (300km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 189 | Ontario. Minsitry of Health | 777 Bay Street, 5th Fl Toronto, ON, M5G 2C8 (1333km) Clipboard | 1-866-532-3161 | **Twisted Name** |
| G070 | 194 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 195 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 197 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (153km) Clipboard | 1-866-532-3161 | Original (Control) |
| G070 | 199 | Ontario. Ministry of Health | 777 Bay St, 5th Fl Toronto, ON, M5G 2C8 (556km) Clipboard | 1-866-532-3161 | Original (Control) |


### Group G072
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G072 | 143 | Orléans-Cumberland Community Resource Centre | 240 Centrum Blvd, Unit 105 Ottawa, ON, K1E 3J4 (19km) Clipboard | *Missing* | **Phone Removed**, **Twisted Name** |
| G072 | 167 | Orléans-Cumberland Community Resource Center | Unit 1, 240 Centrum Blvd, Unit 105 Ottawa, ON, K1E 3J4 (16km) Clipboard | 613-830-4357 | Original (Control) |


### Group G083
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G083 | 52 | Reconnect Health Services | 1281 St Clair Ave W Toronto, ON, M6E 1B8 (19km) Clipboard | *Missing* | **Phone Removed** |
| G083 | 97 | Reconnect Health Services | 1281 St Clair Ave W Toronto, ON, M6E 1B8 (23km) Clipboard | 416-248-2050, 416-248-2050 | Original (Control) |
| G083 | 100 | Reconnect Health Svcs | Unit 1, 1281 St Clair Ave W Toronto, ON, M6E 1B8 (22km) Clipboard | 416-248-2050, 416-248-2050 | **Twisted Name** |
| G083 | 146 | Reconnect Health Services | 1281 St Clair Ave W Toronto, ON, M6E 1B8 (6km) Clipboard | 416-248-2050, 416-248-2050 | Original (Control) |
| G083 | 153 | Reconnect Health Services | 1281 St Clair Ave W Toronto, ON, M6E 1B8 (21km) Clipboard | 416-248-2050, 416-248-2050 | Original (Control) |


### Group G089
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G089 | 42 | Six Nations of the Grand River | 1745 Chiefswood Rd Six Nations Of The Grand River, ON, N0A 1M0 (15km) Clipboard | 519-445-2143, 1-866-445-2204 | Original (Control) |
| G089 | 129 | Six Nations of the Grand River | 1745 Chiefswood Rd Six Nations Of The Grand River, ON, N0A 1M0 (24km) Clipboard | 519-445-2143, 1-866-445-2204 | Original (Control) |


### Group G090
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G090 | 111 | South Asian Welcome Centre | 2 County Court Blvd, Unit 400, Room 462 Brampton, ON, L6W 3W8 (8km) Clipboard | *Missing* | **Phone Removed**, **Twisted Name** |
| G090 | 180 | South Asian Weclome Centre | Unit 1, 2 County Court Blvd, Unit 400, Room 462 Brampton, ON, L6W 3W8 (11km) Clipboard | 647-338-1709, 647-449-0502, 905-595-6777 | Original (Control) |


### Group G093
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G093 | 63 | St Joseph's Care Group | 301 Lillie St N Thunder Bay, ON, P7C 0A6 (2km) Clipboard | *Missing* | **Phone Removed** |
| G093 | 94 | St Joseph'sC are Group | 301 Lillie Street N Thunder Bay, ON, P7C 0A6 (2km) Clipboard | 1-866-346-0463, 807-684-5100 | **Twisted Name** |
| G093 | 154 | St Joseph's Care Group | 301 Lillie St N Thunder Bay, ON, P7C 0A6 (2km) Clipboard | *Missing* | **Phone Removed** |


### Group G106
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G106 | 64 | Wellington Guelph Drug Strategy | Guelph Community Health Centre, 176 Wyndham St N Guelph, ON, N1H 8N9 Clipboard | 519-821-6638 | Original (Control) |
| G106 | 87 | Wellington Guelph Drug Straetgy | Guelph Community Health Centre, 176 Wyndham Street N Guelph, ON, N1H 8N9 (22km) Clipboard | 519-821-6638 | **Twisted Name** |
| G106 | 145 | Wellington Guelph Drug Strategy | Guelph Community Health Centre, 176 Wyndham St N Guelph, ON, N1H 8N9 (21km) Clipboard | *Missing* | **Phone Removed** |
| G106 | 177 | Wellington Guelph Drug Strategy | Guelph Community Health Centre, 176 Wyndham St N Guelph, ON, N1H 8N9 (24km) Clipboard | *Missing* | **Phone Removed** |


### Group G107
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G107 | 1 | Women's College Hospital | 76 Grenville St, 3rd Fl Toronto, ON, M5S 1B2 (22km) Clipboard | 416-323-7559 | Original (Control) |
| G107 | 21 | Women's College Hospital | 76 Grenville St, 3rd Fl Toronto, ON, M5S 1B2 (25km) Clipboard | *Missing* | **Phone Removed** |
| G107 | 31 | Women's College Hospital | 76 Grenville St, 3rd Fl Toronto, ON, M5S 1B2 (24km) Clipboard | 416-323-7559 | Original (Control) |
| G107 | 158 | Women's College Hospital | 76 Grenville St, 3rd Fl Toronto, ON, M5S 1B2 (1km) Clipboard | *Missing* | **Phone Removed** |
| G107 | 196 | Women's College Hosp. | 76 Grenville St, 3rd Fl Toronto, ON, M5S 1B2 (22km) Clipboard | 416-323-7559 | **Twisted Name** |


### Group G109
| Group ID | Row ID | Name | Address | Phone | Status |
|---|---|---|---|---|---|
| G109 | 47 | York Region. Community & Health Services (Public Health) | 17150 Yonge St Newmarket, ON, L3Y 8V3 (24km) Clipboard | 1-800-361-5653 | Original (Control) |
| G109 | 172 | York Region. Community & Health Services (Public Health) | 17150 Yonge St Newmarket, ON, L3Y 8V3 (21km) Clipboard | *Missing* | **Phone Removed** |
| G109 | 192 | York Region. Community & Health Services (Public Health) | 17150 Yonge St Newmarket, ON, L3Y 8V3 (19km) Clipboard | 1-800-361-5653 | Original (Control) |

