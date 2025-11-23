import sqlite3
import json
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain_ollama import OllamaLLM
import json

def init_db(json_data):
    print("--- Creating SQL Database ---")
    conn = sqlite3.connect('video_metadata.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS detections')
    
    cursor.execute('''
        CREATE TABLE detections (
            track_id INTEGER,
            frame_id INTEGER,
            timestamp REAL,
            label TEXT,
            confidence REAL,
            color TEXT,
            license_plate TEXT
        )
    ''')
    
    rows = []
    for entry in json_data:
        rows.append((
            entry.get('track_id'),
            entry.get('frame_id'),
            entry.get('timestamp'),
            entry.get('label'),
            entry.get('confidence'),
            entry.get('color'),
            entry.get('license_plate')
        ))
    
    cursor.executemany('INSERT INTO detections VALUES (?,?,?,?,?,?,?)', rows)
    conn.commit()
    conn.close()
    print(f"--- Database Ready ({len(rows)} records) ---")

def run_langchain_agent():
    db = SQLDatabase.from_uri("sqlite:///video_metadata.db")
    llm = OllamaLLM(model="llama3", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    CITYEYE_SYSTEM_PROMPT = """
            You are "CityEye", an expert video surveillance AI assistant. You are interacting with a SQL database containing video metadata.

            ### DATABASE SCHEMA
            The table name is `video_metadata`.
            The columns are defined as follows based on the provided JSON structure:
            - `track_id` (INTEGER): Unique ID for a specific object tracked across frames.
            - `frame_id` (INTEGER): The frame number.
            - `timestamp` (FLOAT): Time in seconds where the object appears.
            - `label` (TEXT): The object class. Common values: 'car', 'truck', 'bus', 'motorcycle', 'person'.
            - `confidence` (FLOAT): The detection confidence score (0.0 to 1.0).
            - `bbox` (TEXT): The bounding box coordinates stored as a string (e.g., "[0, 691, 359, 982]").
            - `color` (TEXT): The dominant color (e.g., 'blue', 'white'). Can be NULL.
            - `license_plate` (TEXT): The license plate text. IMPORTANT: This field often contains empty strings ("") if no plate was detected.

            ### QUERY GUIDELINES (CRITICAL)
            1. **License Plates**: When asked to find specific license plates, ALWAYS filter out empty strings.
            - *Correct*: `WHERE license_plate IS NOT NULL AND license_plate != ''`
            2. **Fuzzy Matching**:
            - Use `LIKE` with wildcards for colors and labels (e.g., `color LIKE '%red%'`).
            - If the user asks for "pedestrians", search for `label = 'person'`.
            3. **Timestamps**: If asked "when", return the `timestamp`.
            4. **Empty Results**: If the query returns no rows, your final answer should clearly state: "No matching events were found in the video footage."
            5. **Tone**: Be concise, professional, and factual (like a security officer).

            ### EXAMPLE LOGIC
            - User: "Find the white truck."
            - SQL Intent: `SELECT timestamp, track_id, license_plate FROM video_metadata WHERE label = 'truck' AND color LIKE '%white%' LIMIT 5;`

            - User: "What is the license plate of the blue car?"
            - SQL Intent: `SELECT license_plate FROM video_metadata WHERE label = 'car' AND color LIKE '%blue%' AND license_plate != '' LIMIT 1;`
"""
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,  
        prefix=CITYEYE_SYSTEM_PROMPT 
    )

    print("\n--- CityEye Agent Online (Type 'exit' to quit) ---")
    
    while True:
        user_query = input("\nUser: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        
        try:
            
            response = agent_executor.invoke({"input": user_query})
            print(f"\nAgent: {response['output']}")
        except Exception as e:
            
            print(f"System Error: {e}")

if __name__ == "__main__":
    
    with open('output/metadata.json') as f: data = json.load(f)

    init_db(data)
    run_langchain_agent()