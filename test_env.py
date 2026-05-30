from dotenv import load_dotenv
import os

load_dotenv(override=True)
print("CWD:", os.getcwd())
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
print(".env existe:", os.path.exists(".env"))