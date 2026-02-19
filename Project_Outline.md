### Project Overview: AI-Based Healthcare Resource Recommendation System

#### Objective
The primary goal of this project is to develop an AI-driven system that recommends appropriate healthcare resources based on a patient's condition. The system will leverage a local-LLM (Large Language Model) to analyze patient situations, identify relevant sub-types of services, and provide tailored recommendations. Additionally, the system will incorporate user feedback to improve its suggestions over time.

#### Major Tasks
1. **Data Collection and Analysis**
   - Identify and gather a collection of openly available patient situations in natural language, such as doctor notes and patient records.
   
2. **Local-LLM Based Multi-Agent Model Development**
   - Develop a multi-agent model using a local-LLM to analyze patient situations and identify suitable sub-types of services.
   
3. **Integration with Vector Database**
   - Incorporate a vector database to store and retrieve knowledge about suitable sub-categories based on individual patient needs and user feedback.
   
4. **Healthcare Information Retrieval**
   - Retrieve related healthcare information based on identified sub-types and the user's geographic location from sources like 211.
   
5. **User Interface Development**
   - Create a user-friendly interface for the model to facilitate easy interaction and access to recommended resources.

#### Key Components
- **AI and Machine Learning:** Leveraging LLMs to understand and interpret natural language descriptions of patient situations.
- **Multi-Agent Systems:** Utilizing multiple agents to handle various aspects of the recommendation process, ensuring comprehensive and personalized service identification.
- **Vector Databases:** Storing and retrieving knowledge efficiently to enable the system to learn from user feedback and improve over time.
- **Healthcare Information Systems:** Integrating with existing healthcare databases to provide accurate and relevant resource information.
- **User Interface:** Designing an intuitive interface for users to interact with the system, input their conditions, and receive recommendations.

#### Expected Outcomes
- **Personalized Recommendations:** Tailored healthcare resource recommendations based on individual patient situations and preferences.
- **Improved Access to Resources:** Facilitating access to a wide range of healthcare services, including mental health, housing, food, and financial assistance.
- **Continuous Improvement:** An evolving system that becomes more effective with user feedback and ongoing learning.

This project aims to bridge the gap between patient needs and available healthcare resources, leveraging advanced AI technologies to deliver comprehensive, personalized, and location-specific recommendations.

---

### Task 1: Data Collection and Analysis (Focused on Mental Health)

#### Objective
Identify and gather a collection of openly available patient situations in natural language, specifically focusing on mental health-related diagnoses, clinical notes, and treatment descriptions. This data will be used to train and validate the AI model.

#### Steps to Accomplish Task 1

1. **Define Data Requirements**
   - **Type of Data:** Patient situations, doctor notes, patient records, case studies.
   - **Focus Area:** Mental health-related diagnoses and clinical notes.
   - **Format:** Natural language text.
   - **Scope:** Various mental health conditions, including but not limited to anxiety, depression, bipolar disorder, schizophrenia, and PTSD.

2. **Identify Potential Data Sources**
   - **Specific Databases:**
     - **MIMIC-III (Medical Information Mart for Intensive Care)**
       - **Description:** Contains detailed patient records, including mental health diagnoses and psychiatric evaluations for patients admitted to intensive care units.
       - **Link:** [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
       
     - **MIMIC-IV**
       - **Description:** An updated version of MIMIC-III, also containing detailed patient care information, including mental health diagnoses in critical care settings.
       - **Link:** [MIMIC-IV](https://physionet.org/content/mimiciv/1.0/)
       
     - **eICU Collaborative Research Database**
       - **Description:** Contains data from multiple critical care units, including mental health assessments and psychiatric diagnoses for patients admitted to intensive care units.
       - **Link:** [eICU Database](https://physionet.org/content/eicu-crd/2.0/)
       
     - **i2b2 (Informatics for Integrating Biology and the Bedside)**
       - **Description:** A data warehouse containing de-identified clinical data from hospitals, including mental health-related diagnoses and clinical notes.
       - **Link:** [i2b2](https://www.i2b2.org/)
       
     - **n2c2 (National NLP Clinical Challenges)**
       - **Description:** Provides datasets focused on clinical narratives, including mental health-related diagnoses and treatment descriptions.
       - **Link:** [n2c2](https://n2c2.dbmi.hms.harvard.edu/)

3. **Data Collection Process**
   - **Data Search:** Use specific keywords related to mental health (e.g., "anxiety patient records," "depression clinical notes," "bipolar disorder datasets").
   - **Data Evaluation:** Assess the quality and relevance of the datasets. Criteria include completeness, accuracy, diversity, and adherence to privacy regulations.
   - **Data Acquisition:** Download and store the datasets, ensuring compliance with any usage restrictions and privacy policies.

4. **Data Preprocessing**
   - **Cleaning:** Remove any irrelevant information, duplicate entries, and correct any inconsistencies.
   - **Anonymization:** Ensure all patient data is anonymized to protect patient privacy.
   - **Formatting:** Convert data into a uniform format suitable for analysis (e.g., structured text files, CSV).

5. **Documentation**
   - Document the sources of all datasets, including licensing and usage restrictions.
   - Maintain a record of the preprocessing steps applied to each dataset.

#### Suggestions for Finding Available Databases

- **Contact Academic Institutions:** Reach out to universities or research labs conducting studies in mental health. They might have access to datasets or can guide you to relevant sources.
- **Healthcare Conferences and Workshops:** Attend events related to mental health informatics. Networking with professionals can lead to potential data sources.
- **Online Communities and Forums:** Participate in forums like Reddit, ResearchGate, or specialized mental health data science communities where researchers share resources.
- **Collaborations with Healthcare Providers:** Establish partnerships with hospitals or clinics willing to share anonymized mental health data for research purposes.
- **Grants and Funding Proposals:** Apply for grants that facilitate access to proprietary mental health datasets for research purposes.

---

### Task 2: Local-LLM Based Multi-Agent Model Development

#### Objective
Develop a multi-agent model using a local Large Language Model (LLM) to analyze patient situations and identify suitable sub-types of services based on individual mental health needs and preferences. Utilize AutoGen for managing multiple agents and Ollama for deploying LLMs locally.

#### Steps to Accomplish Task 2

1. **Define Model Requirements**
   - **Functionality:** Ability to interpret patient situations, identify relevant mental health sub-types, and provide recommendations.
   - **Agents:** Define the roles and responsibilities of different agents within the multi-agent system.
   - **Integration:** Ensure the model can be easily deployed and managed locally.

2. **Select an LLM Framework**
   - **LLM Platform:** Use Ollama for running LLMs locally, as it simplifies configuration and deployment.
   - **Model Selection:** Choose appropriate models for mental health applications.
     - **Examples:** GPT-4, other open-source models suitable for natural language understanding.

3. **Set Up Ollama for Local LLM Deployment**
   - **Installation:**
     - Download Ollama by visiting the official website or GitHub repo.
     - For Linux, use the command:
       ```sh
       $ curl -fsSL https://ollama.com/install.sh | sh
       ```
   - **Model Download:**
     - Visit the model library on Ollama’s website to select and download the appropriate models.
     - Use the command to pull the model:
       ```sh
       $ ollama run <model-name>
       ```

4. **Design Multi-Agent Architecture Using AutoGen**
   - **Introduction to AutoGen:**
     - AutoGen is an open-source framework that leverages multiple agents to enable complex workflows.
   - **Agents and Roles:**
     - **Input Agent:** Handles the input of patient data and ensures it is in a format suitable for analysis.
     - **Processing Agent:** Uses the LLM to analyze the input data and identify relevant mental health sub-types.
     - **Recommendation Agent:** Matches the identified sub-types with appropriate services and resources.
     - **Feedback Agent:** Collects user feedback to improve the model's recommendations over time.
   - **Communication Protocol:** Define how agents will communicate and share information.

5. **Develop and Configure Agents in AutoGen**
   - **ConversableAgent Example:**
     ```python
     from autogen import ConversableAgent

     agent = ConversableAgent(
         "chatbot",
         llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY")}]},
         code_execution_config=False,  # Turn off code execution, by default it is off.
         function_map=None,  # No registered functions, by default it is None.
         human_input_mode="NEVER",  # Never ask for human input.
     )
     ```

   - **Define Roles and Initiate Conversations:**
     ```python
     cathy = ConversableAgent(
         "cathy",
         system_message="Your name is Cathy and you are a part of a duo of comedians.",
         llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.9, "api_key": os.environ.get("OPENAI_API_KEY")}]},
         human_input_mode="NEVER",  # Never ask for human input.
     )

     joe = ConversableAgent(
         "joe",
         system_message="Your name is Joe and you are a part of a duo of comedians.",
         llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.7, "api_key": os.environ.get("OPENAI_API_KEY")}]},
         human_input_mode="NEVER",  # Never ask for human input.
     )

     result = joe.initiate_chat(cathy, message="Cathy, tell me a joke.", max_turns=2)
     ```

6. **Develop the Model**
   - **Preprocessing Pipeline:** Create a pipeline to preprocess the input data, ensuring it is compatible with the LLM.
   - **Model Training:** Train the LLM on the collected mental health datasets to fine-tune its capabilities in understanding and analyzing mental health situations.
   - **Agent Integration:** Develop the agents and integrate them into the system. Ensure each agent can perform its designated role effectively.
   - **Testing and Validation:** Test the multi-agent model with sample data to validate its accuracy and performance. Fine-tune as necessary.

7. **Optimize Model Performance**
   - **Performance Metrics:** Define metrics to evaluate the model's performance (e.g., accuracy, response time, relevance of recommendations).
   - **Optimization Techniques:** Apply optimization techniques to improve the model’s performance, such as reducing latency, enhancing accuracy, and ensuring scalability.

8. **Implement Feedback Loop**
   - **User Feedback:** Develop a mechanism for collecting and incorporating user feedback to continually improve the model’s recommendations.
   - **Adaptive Learning:** Ensure the model can adapt to new data and feedback, updating its knowledge base and improving over time.

9. **Documentation and Deployment**
   - **Documentation:** Document the entire development process, including the design architecture, training process, and integration steps.
   - **Deployment:** Deploy the model on a local server or within a secure environment ensuring compliance with data privacy and security regulations.

#### Suggested Tools and Technologies
- **LLM Frameworks:** GPT-4, other suitable open-source LLMs.
- **LLM Platform:** Ollama for local deployment.
- **Multi-Agent Framework:** AutoGen.
- **Development Environment:** Python.
- **Collaboration Platforms:** GitHub, JIRA, Confluence.
- **Testing Tools:** Unit testing frameworks, performance benchmarking tools.

---

### Task 3: Incorporate Vector Database with Teachability

#### Objective
Integrate a vector database into the LLM model to persist knowledge about suitable sub-categories and user feedback, enabling the teachable agent to learn and recall information across chat sessions, thus improving healthcare resource recommendations based on patient needs and preferences.

#### Steps to Accomplish Task 3

1. **Define Requirements for Teachability**
   - **Functionality:** The agent should be able to remember user-provided facts, preferences, and skills across sessions.
   - **Persistence:** Use a vector database to store and retrieve memos as needed without copying all of memory into the context window.
   - **Efficiency:** Ensure that the system retrieves only the necessary memos to save valuable space in the context window.

2. **Set Up Environment**
   - **Install Dependencies:** Install the required libraries and dependencies for AutoGen and Teachability.
     ```sh
     pip install "pyautogen[teachable]"
     ```

3. **Create LLM Configuration**
   - **LLM Configuration:** Configure the LLM inference endpoints.
     ```python
     from autogen import config_list_from_json

     filter_dict = {"model": ["local llm"]}  # choose a local LLM
     config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict=filter_dict)
     llm_config = {"config_list": config_list, "timeout": 120}
     ```

4. **Instantiate Agents and Teachability**
   - **Create the Main Agent:** Start by instantiating a `ConversableAgent` tailored to handle healthcare-related tasks.
     ```python
     from autogen import ConversableAgent

     health_agent = ConversableAgent(
         name="health_agent",
         llm_config=llm_config
     )
     ```
   - **Add Teachability:** Create a `Teachability` object and add it to the agent.
     ```python
     from autogen.agentchat.contrib.capabilities.teachability import Teachability

     teachability = Teachability(
         reset_db=False,  # Use True to force-reset the memo DB
         path_to_db_dir="./tmp/interactive/teachability_db"  # Path to the database directory
     )

     teachability.add_to_agent(health_agent)
     ```

5. **Create User Proxy Agent**
   - **Instantiate User Proxy:** Create a user proxy agent for interaction.
     ```python
     from autogen import UserProxyAgent

     user = UserProxyAgent("user", human_input_mode="ALWAYS")
     ```

6. **Chat with the Teachable Agent**
   - **Initiate Chat:** Start a chat session with the teachable agent.
     ```python
     health_agent.initiate_chat(user, message="Hi, I'm your healthcare assistant! How can I assist you today?")
     ```

7. **Example Interactions for Healthcare**
   - **Teach Patient Information:**
     ```python
     user_message = "I have a history of anxiety and need counseling recommendations."
     health_agent.generate_reply(messages=[{"content": user_message, "role": "user"}])
     ```
   - **Recall Patient Information in Later Chat:**
     ```python
     user_message = "What mental health services did you recommend for my anxiety?"
     health_agent.generate_reply(messages=[{"content": user_message, "role": "user"}])
     ```

8. **Learning and Recalling Facts and Preferences**
   - **Teach New Facts:**
     ```python
     user_message = "Cognitive Behavioral Therapy (CBT) is a common treatment for anxiety."
     health_agent.generate_reply(messages=[{"content": user_message, "role": "user"}])
     ```
   - **Teach User Preferences:**
     ```python
     user_message = "I prefer in-person counseling sessions over online ones."
     health_agent.generate_reply(messages=[{"content": user_message, "role": "user"}])
     ```

9. **Optimizing Teachability**
   - **Persistent Storage:** Ensure the vector database is optimized for fast retrieval and storage.
   - **Memo Management:** Implement strategies to manage and retrieve relevant memos efficiently, minimizing latency.

10. **Testing and Validation**
    - **Unit Testing:** Use provided scripts like `test_teachable_agent.py` for quick unit tests.
    - **Interactive Testing:** Use Jupyter notebooks to step through examples and validate the system.

11. **Documentation and Deployment**
    - **Documentation:** Document the teachability integration process, including configuration and usage instructions.
    - **Deployment:** Deploy the model with teachability in a secure environment, ensuring compliance with data privacy and security regulations.

#### Suggested Tools and Technologies
- **LLM Platform:** Ollama for local deployment.
- **Multi-Agent Framework:** AutoGen.
- **Vector Database:** Implement a vector database for persistent memo storage (e.g., FAISS, Milvus, Pinecone).
- **Development Environment:** Python.
- **Collaboration Platforms:** GitHub, JIRA, Confluence.
- **Testing Tools:** Unit testing frameworks, performance benchmarking tools.

---

### Task 4: Retrieve Healthcare Information Based on Sub-Types and User's Location

#### Objective
Understand the URL structure of the 211 website and compile a URL to retrieve healthcare resources based on the sub-type number and the user's location. Extract resource information from the HTML response. A sub-type number lookup table is provided in the appendix. 

#### Steps to Accomplish Task 4

1. **Understand the 211 URL Structure**
   - **Base URL:** The base URL for 211 Ontario is `https://211ontario.ca/results/`.
   - **Parameters:**
     - `latitude`: The latitude of the user's location.
     - `longitude`: The longitude of the user's location.
     - `searchLocation`: The name of the location (city or town).
     - `searchTerms`: Search terms to refine the search (optional).
     - `exct`: Exact match parameter (set to 0 for no exact match).
     - `sd`: Search distance parameter (set to 25 by default).
     - `ss`: Sorting method (set to Distance by default).
     - `topicPath`: The topic path number representing the sub-type of service.

2. **Compile the URL**
   - **Example URL Structure:** `https://211ontario.ca/results/?latitude=<latitude>&longitude=<longitude>&searchLocation=<location>&searchTerms=&exct=0&sd=25&ss=Distance&topicPath=<topicPath>`
   - Replace `<latitude>`, `<longitude>`, `<location>`, and `<topicPath>` with actual values.

3. **Develop a Function to Generate the URL**
   - **Python Function:**
     ```python
     def generate_211_url(latitude, longitude, location, topicPath):
         base_url = "https://211ontario.ca/results/"
         url = (f"{base_url}?latitude={latitude}&longitude={longitude}&searchLocation={location}&"
                f"searchTerms=&exct=0&sd=25&ss=Distance&topicPath={topicPath}")
         return url

     # Example usage:
     latitude = 43.5421184
     longitude = -80.2455552
     location = "Guelph"
     topicPath = 123  # Example topic path number for mental health resources

     url = generate_211_url(latitude, longitude, location, topicPath)
     print(url)  # Output: https://211ontario.ca/results/?latitude=43.5421184&longitude=-80.2455552&searchLocation=Guelph&searchTerms=&exct=0&sd=25&ss=Distance&topicPath=123
     ```

4. **Fetch and Extract Resource Information**
   - **Fetch HTML Data:** Use the generated URL to fetch data from the 211 website.
   - **Extract Information:** Parse the HTML response to extract resource information.
   - **Example Code:**
     ```python
     import requests
     from bs4 import BeautifulSoup

     def fetch_healthcare_information(latitude, longitude, location, topicPath):
         url = generate_211_url(latitude, longitude, location, topicPath)
         response = requests.get(url)
         
         if response.status_code == 200:
             soup = BeautifulSoup(response.content, 'html.parser')
             resources = []

             for record in soup.find_all('div', class_='record-list-row'):
                 title = record.find('div', class_='title').get_text(strip=True)
                 phone_numbers = [a.get_text(strip=True) for a in record.find_all('a', class_='linkphone')]
                 website = record.find('a', class_='linkexternal')['href'] if record.find('a', class_='linkexternal') else None
                 description = record.find('div', class_='record-list-desc').get_text(strip=True) if record.find('div', class_='record-list-desc') else None
                 
                 resource = {
                     'title': title,
                     'phone_numbers': phone_numbers,
                     'website': website,
                     'description': description
                 }
                 resources.append(resource)

             return resources
         else:
             return f"Failed to retrieve data: {response.status_code}"

     # Example usage:
     latitude = 43.5421184
     longitude = -80.2455552
     location = "Guelph"
     topicPath = 123

     resources = fetch_healthcare_information(latitude, longitude, location, topicPath)
     for resource in resources:
         print(resource)
     ```

5. **Document the URL Structure and Extraction Process**
   - **Documentation:**
     - Detail the URL structure and parameters.
     - Provide examples of generating URLs for different locations and sub-types.
     - Include instructions on how to use the function to fetch and extract healthcare resource information.


---

### Task 5: Create a Streamlit App for Testing

#### 5.1. Set Up the Environment
1. **Install Streamlit**:
   ```bash
   pip install streamlit
   ```

2. **Create a New Directory for the Project**:
   ```bash
   mkdir healthcare_resource_finder
   cd healthcare_resource_finder
   ```

3. **Create a Virtual Environment (optional but recommended)**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

#### 5.2. Develop the Streamlit App
1. **Create a New Python File**:
   ```bash
   touch app.py
   ```

2. **Set Up the Basic Structure in `app.py`**:
   ```python
   import streamlit as st
   import requests
   from bs4 import BeautifulSoup

   # Title of the app
   st.title("Healthcare Resource Finder")

   # Inputs for patient condition, location, and service preferences
   condition = st.text_input("Enter the patient's condition:")
   postal_code = st.text_input("Enter the location (postal code):")
   preferences = st.text_input("Enter preferences for type of healthcare services:")

   # Button to trigger analysis and recommendation
   if st.button("Find Resources"):
       if condition and postal_code and preferences:
           st.write("Analyzing condition and preferences...")
           # Placeholder for analysis and recommendation logic
           st.write("Recommending sub types...")
           # Placeholder for URL compilation and data retrieval
           st.write("Fetching and displaying resources...")
       else:
           st.write("Please provide all inputs.")
   ```

#### 5.3. Analyze Condition and Preferences
1. **Analyze the Condition and Preferences**:
   ```python
   def analyze_condition_and_preferences(condition, preferences):
       # Placeholder for NLP and analysis logic
       # Return three recommended sub types from different main types
       return [123, 58, 91]  # Example sub types

   if condition and postal_code and preferences:
       recommended_sub_types = analyze_condition_and_preferences(condition, preferences)
       st.write(f"Recommended Sub Types: {recommended_sub_types}")
   ```

#### 5.4. Compile URLs and Fetch Data
1. **Compile the URLs Based on Recommendations**:
   ```python
   def compile_url(postal_code, sub_type):
       # Example hardcoded latitude and longitude for demonstration
       latitude = 43.5421184
       longitude = -80.2455552
       return (f"https://211ontario.ca/results/?latitude={latitude}"
               f"&longitude={longitude}&searchLocation={postal_code}"
               f"&topicPath={sub_type}&sd=25&ss=Distance")

   if condition and postal_code and preferences:
       urls = [compile_url(postal_code, sub_type) for sub_type in recommended_sub_types]
       st.write(f"Compiled URLs: {urls}")
   ```

2. **Fetch and Display Data**:
   ```python
   def fetch_data_from_url(url):
       response = requests.get(url)
       if response.status_code == 200:
           soup = BeautifulSoup(response.content, "html.parser")
           results = soup.find_all("div", class_="record-list-row")
           return results
       return []

   if condition and postal_code and preferences:
       all_results = []
       for url in urls:
           all_results.extend(fetch_data_from_url(url))

       if all_results:
           for result in all_results:
               title = result.find("div", class_="title").get_text(strip=True)
               phone = result.find("span", class_="linkphone").get_text(strip=True)
               website = result.find("span", class_="small").find("a")["href"]

               st.write(f"**Title**: {title}")
               st.write(f"**Phone**: {phone}")
               st.write(f"**Website**: {website}")
               st.write("---")
       else:
           st.write("No resources found.")
   ```

#### 5.5. Finalize and Test
1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Test the App**:
   - Open the app in a web browser.
   - Enter the patient's condition, postal code, and service preferences.
   - Click on "Find Resources".
   - Verify that the recommendations are made, URLs are compiled, and resources are fetched and displayed properly.

#### 5.6. Additional Enhancements (Optional)
1. **Error Handling**:
   - Add error handling for invalid inputs or network issues.
   - Display user-friendly error messages.

2. **Improved UI/UX**:
   - Enhance the UI for better user experience.
   - Add more input fields or options as needed.

3. **Data Processing**:
   - Implement additional data processing or filtering based on user preferences.
   - Display more detailed information about each resource.


---



## Appendix

### Topic Path Lookup Table with Main Topics

| Topic Path | Description                                    | Main Topic                |
|------------|------------------------------------------------|---------------------------|
| 2          | Child abuse services                           | Abuse / Assault           |
| 3          | Counselling for abused people                  | Abuse / Assault           |
| 4          | Crisis lines for abused people                 | Abuse / Assault           |
| 5          | Sexual assault support                         | Abuse / Assault           |
| 6          | Shelter for abused women                       | Abuse / Assault           |
| 7          | Victims of abuse support programs              | Abuse / Assault           |
| 9          | Community / Recreation centres                 | Community Programs        |
| 10         | Community information                          | Community Programs        |
| 11         | Computer access                                | Community Programs        |
| 13         | Public libraries                               | Community Programs        |
| 14         | Volunteer centres                              | Community Programs        |
| 16         | Assistive devices                              | Disabilities              |
| 17         | Developmental disabilities                     | Disabilities              |
| 18         | Disabilities employment programs               | Disabilities              |
| 19         | Disability associations                        | Disabilities              |
| 20         | Home repairs                                   | Disabilities              |
| 21         | Learning disabilities                          | Disabilities              |
| 22         | Ontario Disability Support Program             | Disabilities              |
| 23         | Parking permits for people with disabilities   | Disabilities              |
| 24         | Physical disabilities                          | Disabilities              |
| 25         | Sport for people with disabilities             | Disabilities              |
| 26         | Transportation for people with disabilities    | Disabilities              |
| 28         | Crisis lines                                   | Emergency / Crisis        |
| 29         | Elder abuse                                    | Emergency / Crisis        |
| 30         | Hospital emergency / Urgent care               | Emergency / Crisis        |
| 31         | In-person crisis services                      | Emergency / Crisis        |
| 32         | Shelters from abuse                            | Emergency / Crisis        |
| 33         | Victims of crime support programs              | Emergency / Crisis        |
| 35         | Academic upgrading                             | Employment / Training     |
| 36         | Adult literacy                                 | Employment / Training     |
| 37         | Apprenticeship                                 | Employment / Training     |
| 38         | Career counselling                             | Employment / Training     |
| 39         | Disabilities employment programs               | Employment / Training     |
| 40         | Employer staffing assistance                   | Employment / Training     |
| 41         | International credentials                      | Employment / Training     |
| 42         | Internationally trained professionals          | Employment / Training     |
| 43         | Job search support / training                  | Employment / Training     |
| 44         | Newcomer employment programs                   | Employment / Training     |
| 45         | Occupational health and safety                 | Employment / Training     |
| 46         | Self-employment / Entrepreneurship             | Employment / Training     |
| 47         | Summer employment                              | Employment / Training     |
| 48         | Work experience                                | Employment / Training     |
| 49         | Youth employment                               | Employment / Training     |
| 51         | Baby items                                     | Family services           |
| 52         | Camps                                          | Family services           |
| 53         | Child abuse services                           | Family services           |
| 54         | Child care                                     | Family services           |
| 55         | Parent / Child programs                        | Family services           |
| 56         | Pregnancy / Postnatal                          | Family services           |
| 58         | Credit Counselling                             | Financial Assistance      |
| 59         | Employment Insurance                           | Financial Assistance      |
| 60         | Federal Government Social Assistance Programs  | Financial Assistance      |
| 61         | Financial assistance programs for seniors      | Financial Assistance      |
| 62         | Ontario Government Social Assistance Programs  | Financial Assistance      |
| 63         | Other Financial Assistance Programs            | Financial Assistance      |
| 64         | Rent and Housing Assistance Programs           | Financial Assistance      |
| 65         | Utility Assistance Programs                    | Financial Assistance      |
| 66         | Workers Compensation                           | Financial Assistance      |
| 68         | Cooking classes and facilities                 | Food                      |
| 69         | Food access for seniors/people with disabilities | Food                    |
| 70         | Food banks and referrals                       | Food                      |
| 71         | Food delivery                                  | Food                      |
| 72         | Food for special dietary needs                 | Food                      |
| 73         | Free / Low-cost meals                          | Food                      |
| 74         | Grocery stores, fresh food and producers       | Food                      |
| 75         | Grow / Pick your own food                      | Food                      |
| 76         | Infant formula / Baby food                     | Food                      |
| 77         | Non-profit catering services and eating establishments | Food               |
| 78         | Other food security initiatives                | Food                      |
| 79         | School meal programs                           | Food                      |
| 81         | Francophones                                   | Francophones              |
| 83         | Community legal clinics                        | Government / Legal        |
| 84         | Community legal services                       | Government / Legal        |
| 85         | Consumer protection / Complaints               | Government / Legal        |
| 86         | Elected officials                              | Government / Legal        |
| 87         | Human rights                                   | Government / Legal        |
| 88         | ID (identification)                            | Government / Legal        |
| 89         | Legal information                              | Government / Legal        |
| 91         | Community health centres                       | Health Care               |
| 92         | Finding a medical professional                 | Health Care               |
| 93         | Home support programs                          | Health Care               |
| 94         | Hospice / palliative care                      | Health Care               |
| 95         | Hospital emergency / Urgent care               | Health Care               |
| 96         | Hospitals                                      | Health Care               |
| 97         | Local Health Integration Networks              | Health Care               |
| 98         | Long term care homes                           | Health Care               |
| 99         | Pregnancy / Postnatal                          | Health Care               |
| 100        | Transportation to medical appointments         | Health Care               |
| 101        | Walk-in medical clinics                        | Health Care               |
| 103        | Free / Low-cost meals                          | Homelessness              |
| 104        | Homeless respite                               | Homelessness              |
| 105        | Homeless shelters                              | Homelessness              |
| 106        | Street outreach                                | Homelessness              |
| 108        | Health related temporary housing               | Housing                   |
| 109        | Help to find housing                           | Housing                   |
| 110        | Home improvement / renovation / accessibility  | Housing                   |
| 111        | Home ownership                                 | Housing                   |
| 112        | Housing expense assistance                     | Housing                   |
| 113        | Other housing related services / organizations | Housing                  |
| 114        | Rental housing                                 | Housing                   |
| 115        | Retirement housing                             | Housing                   |
| 116        | Supportive (semi-independent) housing          | Housing                   |
| 117        | Transitional housing                           | Housing                   |
| 119        | Indigenous Peoples                             | Indigenous Peoples        |
| 121        | LGBTQ+                                         | LGBTQ+                    |
| 123        | Addiction counselling                          | Mental Health / Addictions|
| 124        | Addiction treatment                            | Mental Health / Addictions|
| 125        | Community mental health centres                | Mental Health / Addictions|
| 126        | Crisis lines                                   | Mental Health / Addictions|
| 127        | Geriatric psychiatry                           | Mental Health / Addictions|
| 128        | In-person crisis services                      | Mental Health / Addictions|
| 129        | Justice and mental health programs             | Mental Health / Addictions|
| 130        | Psychiatric hospitals                          | Mental Health / Addictions|
| 131        | Support groups                                 | Mental Health / Addictions|
| 133        | Consulates / Embassies                         | Newcomers                 |
| 134        | English as a second language                   | Newcomers                 |
| 135        | French as a second language                    | Newcomers                 |
| 136        | Immigration / Sponsorship                      | Newcomers                 |
| 137        | International credentials                      | Newcomers                 |
| 138        | Internationally-trained professionals          | Newcomers                 |
| 139        | Interpretation / Translation                   | Newcomers                 |
| 140        | Newcomer employment programs                   | Newcomers                 |
| 141        | Refugees                                       | Newcomers                 |
| 142        | Settlement services                            | Newcomers                 |
| 144        | Elder abuse                                    | Older Adults              |
| 145        | Geriatric psychiatry                           | Older Adults              |
| 146        | Housing for seniors                            | Older Adults              |
| 147        | Income programs for older adults               | Older Adults              |
| 148        | Long term care homes                           | Older Adults              |
| 149        | Retirement homes                               | Older Adults              |
| 150        | Transportation for older adults                | Older Adults              |
| 152        | Shelters for youth                             | Youth                     |
| 153        | Summer employment                              | Youth                     |
| 154        | Young parents                                  | Youth                     |
| 155        | Youth Drop-In Centres                          | Youth                     |
| 156        | Youth employment                               | Youth                     |
