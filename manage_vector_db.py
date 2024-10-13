import json
import sqlite3
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv

load_dotenv()

llm = Ollama(model="mistral", request_timeout=3600.0, force_download=True)
parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")

def create_database(db_path="vectors.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS vectors (
            doc_id TEXT PRIMARY KEY,
            embedding TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_vectors(vectors, db_path="vectors.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for doc_id, embedding in vectors.items():
        c.execute('''
            INSERT OR REPLACE INTO vectors (doc_id, embedding)
            VALUES (?, ?)
        ''', (doc_id, json.dumps(embedding)))
    conn.commit()
    conn.close()

def load_vectors(db_path="vectors.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * FROM vectors')
    rows = c.fetchall()
    vectors = {row[0]: json.loads(row[1]) for row in rows}
    conn.close()
    return vectors

create_database()

#  embeddings for each document
vectors = {}
for doc in documents:
    try:
        # public method if it exists
        embedding = embed_model.embed_text(doc.get_text())
    except AttributeError:
        # private method if necessary
        embedding = embed_model._embed(doc.get_text())
    vectors[doc.doc_id] = embedding

save_vectors(vectors)

loaded_vectors = load_vectors()
vector_index = VectorStoreIndex(loaded_vectors, embed_model=embed_model)

query_engine = vector_index.as_query_engine(llm=llm)

while True:
    user_query = input("Enter Question | or type 'quit' to exit: ")
    if user_query.lower() == 'quit':
        print("See You Soon")
        break

    result = query_engine.query(user_query)
    print(result)
