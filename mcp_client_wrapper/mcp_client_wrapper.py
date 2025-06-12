import asyncio
from typing import List, Dict, Any, Union
from contextlib import AsyncExitStack
import json
import gradio as gr
from gradio.components.chatbot import ChatMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from custom_html_render import render_face_data_html
import logging
from utils import save_uploaded_image, encode_image_to_base64, encrypt_session_id, JsonFileHandle, mark_session_active, save_encrypted_session_keys
import os
from config import session_keys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"🤩 {__name__}")


from llm_node import LLMNode  # ✅ USE THIS NOW
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

import base64
from io import BytesIO
from PIL import Image
import numpy as np


class MCPClientWrapper:
    def __init__(self, session_id=None):
        self.session_id = session_id
        self.session = None
        self.exit_stack = None
        self.tools = []

    def connect(self) -> str:
        server_path = "gradio_mcp_server.py"
        return loop.run_until_complete(self._connect(server_path))

    async def _connect(self, server_path: str) -> str:
        if self.exit_stack:
            await self.exit_stack.aclose()

        self.exit_stack = AsyncExitStack()

        is_python = server_path.endswith('.py')
        command = "python" if is_python else "node"

        server_params = StdioServerParameters(
            command=command,
            args=[server_path],
            env={"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport

        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        init_response = await self.session.initialize()


        response = await self.session.list_tools()
        self.tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]


        tool_names = [tool["function"]["name"] for tool in self.tools]
        return f"Connected to MCP server. Available tools: {', '.join(tool_names)}"

    def process_message(self, session_id, message: str, history: List[Union[Dict[str, Any], ChatMessage]], image_input=None) -> tuple:
        if not self.session:
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Please connect to an MCP server first."}
            ], gr.Textbox(value=""), [None, None, None, None, None], render_face_data_html({})

        
        mark_session_active(session_id) # Update Session
        
        
        self.llm_node = LLMNode(session_id=session_id)

        image_base64 = None
        image = None
        user_data_path = f"tmp/{session_id}"
        save_encrypted_session_keys(session_id, session_keys[session_id])

        if image_input is not None:
            try:
                if isinstance(image_input, str): 
                    image = Image.open(image_input)
                elif isinstance(image_input, np.ndarray): 
                    image = Image.fromarray(image_input.astype("uint8"))
                elif isinstance(image_input, Image.Image):
                    image = image_input
                
                # Save image with session ID
                if image:
                    image_name = save_uploaded_image(f"face", image, user_data_path)
                    image_base64 = encode_image_to_base64(image)
                    logger.info("✅ Input image saved and converted to base64")
            except Exception as e:
                logger.error(f"❌ Failed to handle input image: {e}")

        user_data = JsonFileHandle.load_json_data("user_data", user_data_path)
        # Send to LLM
        new_messages, image_url, new_face_data, new_color_season, product_images = loop.run_until_complete(
            self._process_query(
                message=message,
                history=history,
                image_base64=image_base64,
                face_data=user_data["FaceData"] if user_data.get("FaceData") else None,
                color_season=user_data["ColorSeason"] if user_data.get("ColorSeason") else None,
                encryptId=encrypt_session_id(session_id)
            )
        )


        # Fallback face_data structure if none found
        data_update = False
        if user_data.get("FaceData") is None and new_face_data:
            user_data["FaceData"] = new_face_data
            data_update = True
            
        if user_data.get("ColorSeason") is None and new_color_season: 
            user_data["ColorSeason"] = new_color_season
            data_update = True
            
        if data_update:
            JsonFileHandle.save_json_data("user_data", user_data, user_data_path)

        html_display = render_face_data_html(new_face_data)
        
        
        while len(product_images) < 5:
            product_images.append(None)
        product_images = product_images[:5]

        return history + [{"role": "user", "content": message}] + new_messages, gr.Textbox(value=""), *product_images, html_display


    async def _process_query(
                    self, 
                    message: str, 
                    history: List[Union[Dict[str, Any], ChatMessage]], 
                    image_base64: str = None,
                    face_data: dict = None,
                    color_season: dict = None,
                    encryptId: str = None
                ):
        
        logger.info(f"Image Exist: {True if image_base64 else False}")
        # Run first step: tool suggestion
        if image_base64:
            messages = self.llm_node.build_prompt(history, message, image_base64 =image_base64, vision_enabled=True, 
                                                  type="toolcall", encryptId=encryptId, 
                                                  history_len = 10, face_data = face_data, color_season = color_season)
        else:
            messages = self.llm_node.build_prompt(history, message, vision_enabled=True,
                                                  type="toolcall", history_len = 10, 
                                                  face_data = face_data, color_season = color_season)
        
        step1 = self.llm_node.call_tool_step(messages, self.tools)
        choice = step1["choices"][0]
        tool_calls = choice["message"].get("tool_calls", [])
        logger.info(f"Tool Called: {tool_calls}")
        image_url = None
        result_messages = []
        product_images = []
        if not tool_calls:
            result_messages.append({
                "role": "assistant",
                "content": choice["message"]["content"]
            })
            return result_messages, image_url, face_data, color_season, product_images

        tool = tool_calls[0] #just use 1 tool a time for now
        tool_name = tool["function"]["name"]
        tool_args_json = tool["function"]["arguments"]
        

        try:
            tool_args = json.loads(tool_args_json)
        except Exception:
            tool_args = {}


        # 🛠️ Call the actual tool via MCP
        result = await self.session.call_tool(tool_name, tool_args)

        result_content = result.content
        logger.info(f"Tool Called result: {result_content}")

        if isinstance(result_content, list):
            result_content = "\n".join(str(item.text) for item in result_content)
    
        result_json = None
        try:
            if isinstance(result_content, dict):
                result_json = result_content
            else:
                result_json = json.loads(result_content)

    
        except Exception as e:

          
            step2_messages = self.llm_node.call_generation_step(
                    message=message,
                    history=history,
                    face_data=face_data,
                    color_season=color_season
                )

            result_messages.extend(step2_messages)


        if result_json:
            if isinstance(result_json, dict) and "type" in result_json:
                if result_json["type"] == "product_list":
                    products = result_json["products"]
                    logger.info(f"Products: {products}")
                    result_content =  "\n\n".join(
                            f"{row['name']} ({row['color']}, {row['season']} Wear, {row['usage']}), {row['image_url']}"
                            for row in products
                                    )
                    for row in products:
                        if isinstance(row, dict) and "image_url" in row:
                            product_images.append(row["image_url"])

                    step2_messages = self.llm_node.call_generation_step(
                        message=message,
                        history=history,
                        tool_result=result_content,
                        face_data=face_data,
                        color_season=color_season
                    )
                    result_messages.extend(step2_messages)
                if result_json["type"] == "FaceData":
                    face_data = result_json["FaceData"]
                    color_season = result_json["ColorSeason"]
                    step2_messages = self.llm_node.call_generation_step(
                        message=message,
                        history=history,
                        tool_result=result_content,
                        face_data=face_data,
                        color_season=color_season
                    )
                    result_messages.extend(step2_messages)
                    
                if result_json["type"] == "image":
                    
                    if "url" in result_json:
                        image_url = result_json["url"]
                        result_messages.append({
                            "role": "assistant",
                            "content": f"You can view and download it on the right.",
                        })
                        

                elif result_json["type"] == "text":
                    
                    # 🧠 Run second step using Nebius to finalize the response
                    step2_messages = self.llm_node.call_generation_step(
                        message=message,
                        history=history,
                        tool_result=result_content,
                        face_data=face_data,
                        color_season=color_season
                    )

                    result_messages.extend(step2_messages)

        return result_messages, image_url, face_data, color_season, product_images

