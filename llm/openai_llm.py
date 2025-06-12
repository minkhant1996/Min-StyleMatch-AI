import os
import logging
from typing import Union, Dict, Any
from openai import OpenAI
from config import session_keys, DEFAULT_KEYS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"🎃 {__name__}")

class LLM_Client:
    def __init__(self, session_id, sourceAI: str = "openai", base_url: str = None, api_key: str = None):
        self.sourceAI = sourceAI.lower() 
        settings = session_keys.get(session_id, {})
        if sourceAI.lower() == "openai":
            self.api_key = api_key or settings.get("OPENAI_API_KEY") or DEFAULT_KEYS.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("API key must be provided or set as OPENAI_API_KEY environment variable.")
            self.client  = OpenAI(api_key=self.api_key)
        elif sourceAI.lower() == "nebius":
            base_url = base_url or "https://api.studio.nebius.com/v1/"
        
            self.api_key = api_key or settings.get("NEBIUS_API_KEY") or DEFAULT_KEYS.get("NEBIUS_API_KEY")
            if not self.api_key:
                raise ValueError("API key must be provided or set as NEBIUS_API_KEY environment variable.")
            self.client  = OpenAI(api_key=self.api_key, base_url = base_url)
        else:
            raise ValueError("sourceAI must be either 'openai' or 'nebius'.")

        try:
            self.client.models.list()  # To Check if API key is valid
        except Exception as e:
            raise ValueError(f"Invalid OpenAI key: {e}")

            
    def get_completion(
        self,
        messages: list,
        model: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        tools: list = [],
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        extra_body: dict = None,
    ):
        if not messages:
            raise ValueError("Messages list must not be empty.")
        
        request_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        # logger.info(messages)
        # Only send tools and tool_choice if using OpenAI
        if tools and self.sourceAI.lower() == "openai":
            request_params["tools"] = tools
            request_params["tool_choice"] = tool_choice

        if extra_body:
            request_params.update(extra_body)
                    
        response = self.client.chat.completions.create(**request_params)
        response_json = response.to_dict()

        usage = response_json.get("usage", {})
        
        logger.info(f"Token Usage - Prompt: {usage.get('prompt_tokens')}, "
                    f"Completion: {usage.get('completion_tokens')}, "
                    f"Total: {usage.get('total_tokens')}")

        return response_json



