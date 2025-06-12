from mcp.server.fastmcp import FastMCP
import json
import sys
import io, os
import logging
from llm import LLM_Client
from utils import parse_face_color_string, encode_image_to_base64, decrypt_session_id, JsonFileHandle, load_combined_dataset, load_encrypted_session_keys
from PIL import Image
from config import session_keys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"👻 {__name__}")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

mcp = FastMCP("huggingface_spaces_image_display")

df = load_combined_dataset(
    "sample_clothing_dataset/styles.csv",
    "sample_clothing_dataset/images.csv"
)

product_unique_values = {
    "masterCategory": df["masterCategory"].dropna().unique().tolist(),
    "subCategory": df["subCategory"].dropna().unique().tolist(),
    "articleType": df["articleType"].dropna().unique().tolist(),
    "baseColour": df["baseColour"].dropna().unique().tolist(),
    "season": df["season"].dropna().unique().tolist(),
    "usage": df["usage"].dropna().unique().tolist()
}

logger.info("✅ Final combined dataset shape: %s", df.shape)
logger.info(product_unique_values)

@mcp.tool()
async def developer_bio() -> str:
    """Call this function if user inquiry about the author or developer. This will return info of the devloper who built you."""
    text_prompt_path = "promptsDB"
    with open(os.path.join(text_prompt_path, "developer_bio.txt"), "r", encoding="utf-8") as f:
        developer_bio = f.read().strip()
    
    
    return json.dumps({
        "type": "text",
        "message": developer_bio
    }) 


@mcp.tool()
async def filter_product(masterCategory: str=None, subCategory: str=None, articleType: str=None, color_list: list=[], usage: str=None, unit: int=5, gender: str="male") -> str:
    """
    Call this function if user want to know about the products. You can filter the products by inputing either one or more of the following args;
    Args:
        masterCategory: 'Apparel', 'Footwear' or None
        subCategory: 'Topwear', 'Bottomwear', 'Innerwear', 'Saree', 'Dress', 'Loungewear and Nightwear', 'Apparel Set', 'Socks', 'Shoes', 'Flip Flops', 'Sandal', or None
        articleType: 'Shirts', 'Jeans', 'Track Pants', 'Tshirts', 'Tops', 'Bra', 'Sweatshirts', 'Kurtas', 'Waistcoat', 'Shorts', 'Briefs', 'Sarees', 'Innerwear Vests', 'Dresses', 'Night suits', 'Skirts', 'Blazers', 'Kurta Sets', 'Shrug', 'Trousers', 'Camisoles', 'Boxers', 'Dupatta', 'Capris', 'Bath Robe', 'Tunics', 'Jackets', 'Trunk', 'Lounge Pants', 'Sweaters', 'Tracksuits', 'Swimwear', 'Nightdress', 'Baby Dolls', 'Leggings', 'Kurtis', 'Jumpsuit', 'Suspenders', 'Robe', 'Salwar and Dupatta', 'Patiala', 'Stockings', 'Tights', 'Churidar', 'Lounge Tshirts', 'Lounge Shorts', 'Shapewear', 'Nehru Jackets', 'Salwar', 'Jeggings', 'Rompers', 'Booties', 'Lehenga Choli', 'Clothing Set', 'Rain Jacket', 'Belts', 'Suits', 'Casual Shoes', 'Flip Flops', 'Sandals', 'Formal Shoes', 'Flats', 'Sports Shoes', 'Heels', 'Sports Sandals', or None
        color_list: a list of your choice from ['Navy Blue', 'Blue', 'Black', 'Grey', 'Green', 'Purple', 'White', 'Beige', 'Brown', 'Pink', 'Maroon', 'Red', 'Off White', 'Yellow', 'Charcoal', 'Multi', 'Magenta', 'Orange', 'Sea Green', 'Cream', 'Peach', 'Olive', 'Burgundy', 'Grey Melange', 'Rust', 'Rose', 'Lime Green', 'Teal', 'Khaki', 'Lavender', 'Mustard', 'Coffee Brown', 'Skin', 'Turquoise Blue', 'Nude', 'Mauve', 'Mushroom Brown', 'Tan', 'Gold', 'Taupe', 'Silver', 'Fluorescent Green', 'Copper', 'Bronze', 'Metallic']
        usage: 'Casual', 'Ethnic', 'Formal', 'Sports', 'Smart Casual', 'Party', 'Travel'
        unit: 1,2,3,4, or 5 - no more than 5,
        gender: Men or Women (based on user image)
    """
    df_copy = df.copy()
    logger.info(f"Args: {masterCategory}, {subCategory}, {articleType}, {color_list}, {usage}")
    if masterCategory:
        df_copy = df_copy[df_copy["masterCategory"].str.lower() == masterCategory.lower()]
    if subCategory:
        df_copy = df_copy[df_copy["subCategory"].str.lower() == subCategory.lower()]
    if articleType:
        df_copy = df_copy[df_copy["articleType"].str.lower() == articleType.lower()]
    if color_list:
        df_copy = df_copy[df_copy["baseColour"].str.lower().isin([c.lower() for c in color_list])]
    if usage:
        df_copy = df_copy[df_copy["usage"].str.lower() == usage.lower()]
    if gender:
        df_copy = df_copy[df_copy["gender"].str.lower() == gender.lower()]

    if df_copy.empty:
        logger.info("No product")
        return json.dumps({
            "type": "text",
            "message": "❌ No products found for the selected filters."
        })

    # Format top 5 results
    number_row = min(unit, 5)
    top_products = df_copy.head(number_row)
    logger.info(top_products)
    product_list = [
        {
            "name": row["productDisplayName"],
            "color": row["baseColour"],
            "season": row["season"],
            "usage": row["usage"],
            "image_url": row["link"]
        }
        for _, row in top_products.iterrows()
    ]
    
    logger.info(f"{product_list}")
    
    return json.dumps({
        "type": "product_list",
        "products": product_list,
    })


@mcp.tool()
async def extract_face_color(tmp_id: str) -> str:
    """
    Call this tool to get color season of the user. It also extract facial parts' color codes of the user face.
    Args:
        tmp_id: tmp_id user provided
    """
    
    try:
        sessionId = decrypt_session_id(tmp_id)
        image_filepath = f"tmp/{sessionId}/face.png"
        logger.info("File Path: %s", image_filepath)
        settings = load_encrypted_session_keys(sessionId)
        

        
        try:
            image = Image.open(image_filepath)
            image_base64 = encode_image_to_base64(image)
        except Exception as e:
            logger.error(f"{e}")
            return json.dumps({"type": "error", "message": f"Failed to load or encode image: {str(e)}"})
        
        provider =  settings.get("provider", "OpenAI")
        try:
            if provider.lower() == "openai":
                openai_key = settings.get("OPENAI_API_KEY")
                if not openai_key:
                    raise ValueError("OpenAI API key is missing in session settings.")
                client = LLM_Client(sessionId.strip(), sourceAI="openai", api_key=openai_key)

            else:
                nebius_key = settings.get("NEBIUS_API_KEY")
                if not nebius_key:
                    raise ValueError("Nebius API key is missing in session settings.")
                client = LLM_Client(sessionId.strip(), sourceAI="nebius", api_key=nebius_key)

        except Exception as e:
            logger.error(f" Failed to initialize LLM_Client: {e}")
            return json.dumps({
                "type": "error",
                "message": f"❌ LLM client initialization failed: {str(e)}"
            })
        vllm_model = settings.get("VLLM_model")
        if not vllm_model:
            if provider.lower() == "openai":
                vllm_model = "gpt-4.1-mini"
            else:
                vllm_model = "Qwen/Qwen2.5-VL-72B-Instruct"


        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You task is to extract the color of facial parts for color analysis."
            },
            {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Extract the color of the face, hair, eye, lips, eyebrow.\n"
                    "Your result must be in the following exact format:\n"
                    "<facecolor>#Colorcode<facecolor>\n"
                    "<haircolor>#Colorcode<haircolor>\n"
                    "<eyecolor>#Colorcode<eyecolor>\n"
                    "<lipscolor>#Colorcode<lipscolor>\n"
                    "<eyebrowcolor>#Colorcode<eyebrowcolor>"
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]}]
        for i in range(3):
            try:
                error = None
                face_data_response = client.get_completion(
                        model=vllm_model,
                        max_tokens=300,
                        messages=messages,
                    )
                
                raw_message = face_data_response["choices"][0]["message"]["content"]
                logger.info(f" Raw face data LLM response: {raw_message}")
                
                face_data = parse_face_color_string(raw_message)
                logger.info(f"face_data: {face_data}")
                break
                
            except (KeyError, IndexError) as e:
                error = e
                logger.error(f"❌ Invalid LLM response structure on attempt {i + 1}: {e}")

            except Exception as e:
                error = e
                logger.error(f"❌ Failed to parse face data on attempt {i + 1}: {e}")

        if error:
            logger.error("Failed to extract valid face data after 3 attempts.")
            return json.dumps({
                "type": "error",
                "message": "Failed to extract valid face data after 3 attempts."
            })

        logger.info(" Facial Parts' Color Extracted")
        
        if face_data:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": """
        You are a color analysis expert. Analyze the color season of the user based on the user's color of facial part.
        overall color season: Spring, Summer, Autumn, Winter
        Undertone: Warm, Cool, Neutral, Clear Cool, Deep, Soft, Light
        """
                    },
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            f"""
        My facial parts' color: {face_data}.
        Your task is to analyze and provide me - overall color season and undertone.
        Your resonse must only be color season with undertone.
        example: Clear Winter
                            """
                        )}
                ]}]
                face_data_response = client.get_completion(
                                        model=vllm_model,
                                        max_tokens=300,
                                        messages=messages,
                                    )
                color_season = face_data_response["choices"][0]["message"]["content"]
                
            except:
                logger.error(" Failed to get color season")

        logger.info(" Color Season Extracted")
        return json.dumps({
            "type": "FaceData",
            "FaceData": face_data,
            "ColorSeason": color_season
        }) 

    except Exception as e:
        logger.exception(f"❌ Error in extract_face_color, {str(e)}")
        return json.dumps({
            "type": "error",
            "message": f"Error executing extract_face_color"
        })
if __name__ == "__main__":
    mcp.run(transport='stdio')