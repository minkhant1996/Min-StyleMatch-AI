import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv
load_dotenv()

session_keys = {}

DEFAULT_KEYS = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "NEBIUS_API_KEY": os.getenv("NEBIUS_API_KEY"),
}


ENV_PATH = ".env"
CryptographyKey = os.getenv("CryptographyKey")
if not CryptographyKey:
    key = Fernet.generate_key().decode()
    CryptographyKey = key

    # Append the key to the existing .env content (don't overwrite)
    with open(ENV_PATH, "a") as f:
        f.write(f"\nCryptographyKey={CryptographyKey}\n")

    print("✅ CryptographyKey added to .env")
else:
    print("🔐 CryptographyKey already exists")


load_dotenv(dotenv_path=ENV_PATH)
CryptographyKey = os.getenv("CryptographyKey")