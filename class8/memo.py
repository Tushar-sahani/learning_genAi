from mem0 import Memory
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

QUADRANT_HOST = "localhost"

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "neo4j-password"

api_key= os.getenv("OPENAI_API_KEY")

config={
    "version":"v1.1",
    "embedder":{
        "provider":"openai",
        "config":{"api_key":api_key,
                  "model":"text-embedding-3-small"}
    },
    "llm":{
        "provider":"openai",
        "config":{"api_key":api_key,
                  "model":"gpt-4.1"}
    },
    "vector_store":{
        "provider":"qdrant",
        "config":{"host":QUADRANT_HOST,
                    "port":6333,
        }
                  
    },
    "graph_store":{
        "provider":"neo4j",
        "config":{"url":NEO4J_URL,
                    "username":NEO4J_USERNAME,
                    "password":NEO4J_PASSWORD}
    }
}

mem_client = Memory.from_config(config)

openai_client = OpenAI(api_key=api_key)


def chat(user_input):
    search_results = mem_client.search(query=user_input, user_id="1223")

    print("Search Results: ", search_results)

    memories = ""
    for memory in search_results.get("results"):
        memories += f"{str(memory.get("memory"))} ({str(memory.get("score"))})"
        SYSTEM_PROMPT = f"""
            You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to systematically analyze input content, extract structured knowledge, and maintain an optimized memory store. Your primary function is information distillation and knowledge preservation with contextual awareness.
            Tone: Professional analytical, precision-focused, with clear uncertainty signaling
            {memories} 
        """
    messages=[{"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": user_input}]

    result = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=messages
    )

    reply = result.choices[0].message.content

    messages.append({"role": "assistant", "content": reply})

    mem_client.add(messages,user_id="1223")
    return reply


while True:
    message=input(">> ")
    print("BOT: ", chat(message))

