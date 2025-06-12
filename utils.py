import uuid
from PIL import Image
from io import BytesIO
import base64
import os
from typing import Union
import numpy as np
import json
import re
import pandas as pd
import time
from cryptography.fernet import Fernet
import shutil
import asyncio
import logging
from config import session_keys, CryptographyKey


cipher = Fernet(CryptographyKey)
session_expiry = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_session_keys(session_id, keys):
    with open(f"session_keys/{session_id}.json", "w") as f:
        json.dump(keys, f)

def load_session_keys(session_id):
    try:
        with open(f"session_keys/{session_id}.json") as f:
            return json.load(f)
    except:
        return {}
    
def get_or_create_session():
    return str(uuid.uuid4())

def reset_session():
    return str(uuid.uuid4())

def encode_image_to_base64(image: Union[Image.Image, np.ndarray]) -> str:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def decode_base64_image(base64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_bytes))


def save_uploaded_image(image_name: str, image: Image.Image, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"{image_name}.png")
    image.save(image_path)
    return image_path

class JsonFileHandle:
    @staticmethod
    def save_json_data(json_name: str, data: dict, save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, f"{json_name}.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return json_path
    
    @staticmethod
    def load_json_data(json_name: str, save_dir: str) -> dict:
        json_path = os.path.join(save_dir, f"{json_name}.json")

        if not os.path.exists(json_path):
            return {}

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {}
        
def parse_face_color_string(color_string: str) -> dict:
    pattern = r"<(.*?)>#([0-9a-fA-F]{6})<\1>"
    matches = re.findall(pattern, color_string)

    color_mapping = {part: f"#{color}" for part, color in matches}
    return color_mapping


def encrypt_session_id(session_id: str) -> str:
    return cipher.encrypt(session_id.encode()).decode()

def decrypt_session_id(encrypted_id: str) -> str:
    return cipher.decrypt(encrypted_id.encode()).decode()


def load_combined_dataset(styles_path: str, images_path: str) -> pd.DataFrame:
    try:
        # Load both CSVs and skip malformed lines
        styles_df = pd.read_csv(styles_path, on_bad_lines='skip')
        images_df = pd.read_csv(images_path, on_bad_lines='skip')
    except Exception as e:
        raise RuntimeError(f"❌ Failed to read CSV files: {e}")

    try:
        # Remove '.jpg' and convert to int for joining
        images_df["id"] = images_df["filename"].str.replace(".jpg", "", regex=False).astype(int)
    except Exception as e:
        raise ValueError(f"❌ Failed to process 'filename' column in images.csv: {e}")

    try:
        # Drop rows with any null values
        styles_df.dropna(inplace=True)
        images_df.dropna(inplace=True)

        # Only keep rows where IDs exist in both tables
        combined_df = pd.merge(styles_df, images_df, on="id", how="inner")
        combined_df.dropna(inplace=True)  # Ensure final result has no nulls

    except Exception as e:
        raise RuntimeError(f"❌ Failed during merge or cleaning: {e}")

    product_to_sell = ['Apparel', 'Footwear']
    combined_df = pd.concat([
        combined_df[combined_df["masterCategory"].str.lower() == product.lower()]
        for product in product_to_sell
    ], ignore_index=True)

    return combined_df


def mark_session_active(session_id):
    session_expiry[session_id] = time.time()

async def cleanup_old_sessions(threshold_seconds=600):
    while True:
        now = time.time()
        to_delete = []
        for session_id, last_active in session_expiry.items():
            if now - last_active > threshold_seconds:
                to_delete.append(session_id)
            else:
                logger.info(f"{session_id} still active!")

        for session_id in to_delete:
            # Delete CSV file
            csv_path = f"csv_logs/{session_id}.csv"
            if os.path.exists(csv_path):
                os.remove(csv_path)
                logger.info(f"Deleted CSV: {csv_path}")
            
            # Delete session folder
            folder_path = f"tmp/{session_id}"
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                logger.info(f"Deleted folder: {folder_path}")

            del session_expiry[session_id]
            
            if session_id in session_keys:
                del session_keys[session_id]
                logger.info(f"Deleted API keys for session: {session_id}")

        await asyncio.sleep(5)
        
def save_encrypted_session_keys(session_id: str, data: dict):
    folder = f"tmp/{session_id}"
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, "session_settings.enc")

    try:
        encrypted = cipher.encrypt(json.dumps(data).encode())
        with open(filepath, "wb") as f:
            f.write(encrypted)
        logger.info(f"🔐 Encrypted session settings saved to {filepath}")
    except Exception as e:
        logger.error(f"❌ Failed to save encrypted session keys for {session_id}: {e}")
        
def load_encrypted_session_keys(session_id: str) -> dict:
    filepath = os.path.join("tmp", session_id, "session_settings.enc")

    if not os.path.exists(filepath):
        logger.warning(f"⚠️ Encrypted session file not found: {filepath}")
        return {}

    try:
        with open(filepath, "rb") as f:
            encrypted_data = f.read()
        decrypted = cipher.decrypt(encrypted_data)
        return json.loads(decrypted.decode())
    except Exception as e:
        logger.error(f"❌ Failed to load or decrypt session keys for {session_id}: {e}")
        return {}
