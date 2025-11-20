"""Examples for training the hospital extraction model"""

EXAMPLES = [
{
    "input": {
        "page_text": '''
Toronto General Hospital
200 Elizabeth Street
Toronto, Ontario M5G 2C4
Phone: (416) 340-3111
Website: www.uhn.ca
The Toronto General Hospital is a major teaching hospital in downtown Toronto.
        ''',
        "goal": "Find hospitals in Toronto",
        "schema": ["name", "address", "phone", "website", "source_url"]
    },
    "output": {
        "items": [{
            "name": "Toronto General Hospital",
            "address": "200 Elizabeth Street, Toronto, Ontario M5G 2C4",
            "phone": "(416) 340-3111", 
            "website": "www.uhn.ca",
            "source_url": ""
        }]
    }
},
{
    "input": {
        "page_text": '''
Mount Sinai Hospital
600 University Avenue
Toronto, ON M5G 1X5
Tel: 416-596-4200
www.mountsinai.on.ca

Part of Sinai Health, Mount Sinai Hospital is an internationally recognized 442-bed acute care academic health sciences centre.
        ''',
        "goal": "Find hospitals in Toronto",
        "schema": ["name", "address", "phone", "website", "source_url"]
    },
    "output": {
        "items": [{
            "name": "Mount Sinai Hospital",
            "address": "600 University Avenue, Toronto, ON M5G 1X5",
            "phone": "416-596-4200",
            "website": "www.mountsinai.on.ca",
            "source_url": ""
        }]
    }
},
{
    "input": {
        "page_text": '''
Top Hospitals in Toronto:

St. Michael's Hospital
30 Bond Street
Toronto, ON M5B 1W8
Main Phone: (416) 360-4000
www.stmichaelshospital.com

North York General Hospital
4001 Leslie Street
Toronto, ON M2K 1E1
Tel: (416) 756-6000
Visit us at www.nygh.on.ca
        ''',
        "goal": "Find hospitals in Toronto",
        "schema": ["name", "address", "phone", "website", "source_url"]
    },
    "output": {
        "items": [
            {
                "name": "St. Michael's Hospital",
                "address": "30 Bond Street, Toronto, ON M5B 1W8",
                "phone": "(416) 360-4000",
                "website": "www.stmichaelshospital.com",
                "source_url": ""
            },
            {
                "name": "North York General Hospital", 
                "address": "4001 Leslie Street, Toronto, ON M2K 1E1",
                "phone": "(416) 756-6000",
                "website": "www.nygh.on.ca",
                "source_url": ""
            }
        ]
    }
}
]