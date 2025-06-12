import json
from typing import List, Dict, Any, Union
from gradio.components.chatbot import ChatMessage
from llm import LLM_Client
import os
from config import session_keys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"🤖 {__name__}")

class LLMNode:
    def __init__(self, session_id):
        self.session_id = session_id
        settings = session_keys.get(session_id, {})
        
        self.provider = settings.get("provider", "OpenAI")
        self.tool_model = settings.get("tool_call_model", "gpt-4o-mini")
        self.response_model = settings.get("response_model", "gpt-4o-mini")
        logger.info(f"Provider: {self.provider}")
        logger.info(f"Tool_model: {self.tool_model}")
        logger.info(f"Response_model: {self.response_model}")
        self.tool_selector = LLM_Client(session_id, sourceAI="openai")
        self.generator = None

        if self.provider.lower() == "openai":
            if settings.get("OPENAI_API_KEY"):
                self.generator = LLM_Client(session_id, sourceAI="openai")
        elif self.provider.lower() == "nebius":
            if settings.get("NEBIUS_API_KEY"):
                self.generator = LLM_Client(session_id, sourceAI="nebius")


    def build_prompt(
        self,
        history: List[Union[Dict[str, Any], ChatMessage]],
        message: str,
        image_base64: str = None,
        vision_enabled: bool = False,
        type: str = "generate", 
        encryptId: str = None,
        history_len = 20,
        face_data: dict = None,
        color_season: str = None
    ) -> List[Dict[str, Any]]:
        prompts = []
        chat_prompt = []
        for msg in history:
            if isinstance(msg, ChatMessage):
                role, content = msg.role, msg.content
            else:
                role, content = msg.get("role"), msg.get("content")
            if role in ["user", "assistant", "system"]:
                chat_prompt.append({"role": role, "content": content})
        
        
        user_data_str = ""
        if face_data:
            user_data_str += f"User facial parts' color: {face_data}.\n" 
        if color_season:
            user_data_str += f"User color season: {color_season}\n"
        if encryptId:
            user_data_str += f"tmp_id: {encryptId}"

        user_message = ""
        if user_data_str != "":
            user_message += f"UserData info: {user_data_str}\n\n"
        user_message += message
        if vision_enabled:
            if image_base64:
          
                chat_prompt.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                })
            else:
                chat_prompt.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message}
                    ]
                })
        else:
            chat_prompt.append({"role": "user", "content": message})
        
        # Construct Main System Prompt
        text_prompt_path = "promptsDB"
        with open(os.path.join(text_prompt_path, "system_prompt.txt"), "r", encoding="utf-8") as f:
                system_prompt_str = f.read().strip()
        system_prompt = {
                    "role": "system",
                    "content": system_prompt_str
                }
        prompts.append(system_prompt)
        
        # Construct User Data
        logger.info(f"User Message: \n{user_message}")
        logger.info(f"Image Exist: {True if image_base64 else False}")

        chat_prompt
        prompts = prompts + chat_prompt[-history_len:]
        
        return prompts



    def call_tool_step(
        self,
        messages: List[Union[Dict[str, str], Dict[str, Any]]],
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return self.tool_selector.get_completion(
            model=self.tool_model,
            max_tokens=300,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )


    def call_final_step(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:        
        return self.generator.get_completion(
            model=self.response_model,
            max_tokens=300,
            messages=messages,
            tool_choice="none"
        )

    def sanitize_messages_for_nebius(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        sanitized = []
        for msg in messages:
            if msg["role"] in ["user", "assistant", "system"]:
                sanitized.append({"role": msg["role"], "content": msg.get("content", "")})
            elif msg["role"] == "tool":
                content = msg.get("content", "")
                sanitized.append({"role": "assistant", "content": f"Tool '{msg['name']}' returned:\n{content}"})
        return sanitized


    def call_generation_step(
        self,
        message: str,
        history: List[Union[Dict[str, Any], ChatMessage]],
        tool_result: str = None,
        face_data: dict = None,
        color_season: str = None
    ) -> List[Dict[str, Any]]:
        messages = self.build_prompt(
            history=history,
            message=message,
            image_base64=None,
            vision_enabled=False,
            face_data=face_data,
            color_season=color_season
        )

        if tool_result:
            messages.append({
                "role": "assistant",
                "content": f"Information (in your system) - use this if you required to answer that: \n\n{tool_result}"
            })

        step2 = self.call_final_step(messages)

        return [{"role": "assistant", "content": step2["choices"][0]["message"]["content"]}]


