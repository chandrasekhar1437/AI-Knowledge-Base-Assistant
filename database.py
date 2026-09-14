import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in your .env file")

supabase: Client = create_client(url, key)