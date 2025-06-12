import asyncio
import gradio as gr
import os
from config import session_keys
from mcp_client_wrapper import MCPClientWrapper
import logging
from utils import get_or_create_session, reset_session, decode_base64_image, cleanup_old_sessions
from custom_html_render import render_face_data_html
import asyncio
from llm import LLM_Client
import numpy as np
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"😎 {__name__}") # Do not wonder why I uses emoj in logger - It is visually easier to track

def clear_data(sessionId):
    folder_path = f"tmp/{sessionId}"
    message = ""
    try:
        session_keys[sessionId] = {}
        message += "API keys and model selection cleared! If you chat without adding new, there will be error."
    except:
        message += "Could not del API keys and model selection"
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        message += "Image Data and Color Analysis Data are cleared."
    except:
        message += "Failed to remove Image Data and Color Analysis Data"
    return message

def show_images(*image_urls):
    updates = []
    for i in range(5):
        try:
            if i < len(image_urls):
                updates.append(show_image(image_urls[i]))
            else:
                updates.append(gr.update(visible=False, value=None))
        except Exception as e:
            logger.warning(f"Failed to show image at index {i}: {e}")
            updates.append(gr.update(visible=False, value=None))
    return updates


def show_image(image_url):
    if isinstance(image_url, np.ndarray):
        return gr.update(value=image_url, visible=True)
    if isinstance(image_url, str):
        if not image_url.strip():
            return gr.update(visible=True)
        try:
            img = f"https://ysharma-sanasprint.hf.space/gradio_api/file={image_url}"
            return gr.update(value=img, visible=True)
        except Exception as e:
            logger.warning(f"Failed to decode image: {e}")
            return gr.update(visible=True)
    return gr.update(visible=True)

def check_keys_and_toggle_inputs(session_id):
    settings = session_keys.get(session_id, {})
    openai_key = settings.get("OPENAI_API_KEY")
    nebius_key = settings.get("NEBIUS_API_KEY")
    provider = settings.get("provider", "OpenAI")
    if provider == "OpenAI":
        if not openai_key:
            return (
                gr.update(interactive=False),  
                gr.update(interactive=False),  
                gr.update(interactive=False), 
                "⚠️ Please go to **Settings** and add your OpenAI API key to start."
            )
        try:
            _ = LLM_Client(session_id, sourceAI="openai", api_key=openai_key)
        except Exception as e:
            return (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "⚠️ Please go to **Settings** and add your OpenAI API key to start."
            )
    if provider == "Nebius":
        if not nebius_key:
            return (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "⚠️ Please go to **Settings** and add your Nebius API key to start."
            )
        try:
            _ = LLM_Client(session_id, sourceAI="nebius", api_key=nebius_key)
        except Exception as e:
            return (
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "⚠️ Please go to **Settings** and add your Nebius API key to start."
            )
    return (
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        f"Using {provider} Provider"
    )
        
def gradio_interface():
    with gr.Blocks(title="StyleMatch Assistant – Find your colors. Find your style.",
                   css=".custom-image { height: 200px !important; width: 300px !important; object-fit: contain; } .custom-image2 { height: 250px  !important; object-fit: contain; }") as demo:
        session_id_state = gr.State(str(get_or_create_session()))
        client_state = gr.State()

        gr.Markdown("# StyleMatch Assistant – Find your colors. Find your style.")
        
        def message_handler(session_id, message, history, image_input, client):
            if client is None:
                logger.warning(f"⚠️ No client found in state for session {session_id[:5]}")
                return history + [{"role": "assistant", "content": "⚠️ Please save your API key first in Settings tab."}], gr.Textbox(value=""), *[None]*5, render_face_data_html({})

            return client.process_message(session_id, message, history, image_input)


        with gr.Tab("Chat"):
            key_status = gr.Markdown(
                    "⚠️ <span style='color:orange; font-size: 18px; font-weight: bold;'>Please add OpenAI API key to continue.</span>"
                )

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Face Image", 
                        visible=True, 
                        interactive=False,
                        elem_classes="custom-image2",
                        type="filepath",
                        scale=2
                    )
                    with gr.Row(scale=3):
                        face_data_display = gr.HTML(label="Face Analysis Result",
                                                    value=render_face_data_html({}))
                    
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        value=[],
                        height=500,
                        type="messages",
                        show_copy_button=True,
                        avatar_images=("asset/avatar.png", "asset/bot.png"),
                        scale=4
                    )
                    with gr.Row(equal_height=True):
                        msg = gr.Textbox(
                            label="What would you like to know?",
                            placeholder="What color do I match with?",
                            scale=2,  
                            interactive=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", size="sm", interactive=False) 

            image_outputs = []
            with gr.Row(equal_height=True):
                for i in range(5):
                    img = gr.Image(
                        label=f"Product Image {i+1}", 
                        visible=True, 
                        interactive=False,
                        elem_classes="custom-image"
                    )
                    image_outputs.append(img)

            def bind_message_submission(trigger):
                return trigger(
                fn=message_handler,
                inputs=[session_id_state, msg, chatbot, image_input, client_state],
                outputs=[chatbot, msg] + image_outputs + [face_data_display]
            ).then(
                fn=show_images,
                inputs=image_outputs,
                outputs=image_outputs
            )

            bind_message_submission(msg.submit)
            bind_message_submission(send_btn.click)

        with gr.Tab("Settings"):
            clear_data_btn = gr.Button("Clear Data", variant="primary", size="sm", interactive=True) 
            data_clear_status = gr.Markdown("")
            gr.Markdown("## 🔧 Model Settings")

            provider_selector = gr.Dropdown(
                label="Provider",
                choices=["OpenAI", "Nebius"],
                value="OpenAI"
            )

            tool_call_selector = gr.Dropdown(
                label="Tool Call Model (OpenAI only)",
                choices=["gpt-4o-mini", "gpt-4.1-mini"],
                visible=True,
                value="gpt-4o-mini"
            )

            response_model_selector = gr.Dropdown(
                label="Response Model",
                choices=["gpt-4o-mini", "gpt-4.1-mini"],  
                value="gpt-4o-mini"
            )
            
            vllm_model_selector = gr.Dropdown(
                label="VLLM Model",
                choices=["gpt-4o-mini", "gpt-4.1-mini"],  
                value="gpt-4o-mini"
            )

            model_feedback = gr.Markdown("")
            set_model_btn = gr.Button("Set Models")
            
            gr.Markdown("## 🔐 API Key Settings")

            with gr.Column():
                openai_key_input = gr.Textbox(
                    label="OpenAI API Key (Required)", 
                    placeholder="Enter your OpenAI key", 
                    type="password"
                )

                nebius_key_input = gr.Textbox(
                    label="Nebius API Key (Optional)", 
                    placeholder="Enter your Nebius API Key", 
                    type="password"
                )

                set_keys_btn = gr.Button("🔐 Save Keys")
                key_feedback = gr.Markdown("")

            clear_data_btn.click(fn=clear_data, inputs=session_id_state, outputs=data_clear_status)


            def set_user_keys(session_id, provider, openai_key, nebius_key, tool_model, response_model, vllm_model):
                openai_key = openai_key.strip()
                nebius_key = nebius_key.strip()

                provider = provider.strip()

                if not openai_key: # OpenAI is a must
                    return "⚠️ <span style='color:orange'>Please provide a valid OpenAI API key.</span>"
                try:
                    _ = LLM_Client(session_id, sourceAI="openai", api_key=openai_key)
                except Exception as e:
                    logger.warning(f"❌ Invalid OpenAI key: {e}")
                    return "⚠️ <span style='color:orange'>OpenAI key is invalid.</span>"

                if provider == "Nebius": #Only if Nebius is selected
                    if not nebius_key:
                        return "⚠️ <span style='color:orange'>Please provide a valid Nebius API key.</span>"
                    try:
                        _ = LLM_Client(session_id, sourceAI="nebius", api_key=nebius_key)
                    except Exception as e:
                        logger.warning(f"❌ Invalid Nebius key: {e}")
                        return "⚠️ <span style='color:orange'>Nebius key is invalid.</span>"


                # Save only if validation passed
                session_keys[session_id] = {
                    "OPENAI_API_KEY": openai_key,
                    "NEBIUS_API_KEY": nebius_key,
                    "provider": provider 
                }
                session_keys[session_id]["provider"] = provider.capitalize()
                session_keys[session_id]["tool_call_model"] = tool_model 
                session_keys[session_id]["response_model"] = response_model
                session_keys[session_id]["VLLM_model"] = vllm_model
                logger.info(f"✅ API keys set for session {session_id[:5]}")
                return f"✅ Keys saved for session `{session_id[:5]}`."


            def init_client_after_key_save(session_id):
                client = MCPClientWrapper(session_id=session_id)
                connect_msg = client.connect()
                logger.info(connect_msg)
                return client, f"✅ Client ready for session {session_id[:5]}"

            
            set_keys_btn.click(
                fn=set_user_keys,
                inputs=[session_id_state, provider_selector, openai_key_input, nebius_key_input, tool_call_selector, response_model_selector, vllm_model_selector],
                outputs=[key_feedback]
            ).then(
                fn=check_keys_and_toggle_inputs,
                inputs=[session_id_state],
                outputs=[msg, send_btn, image_input, key_status]
            ).then(
                fn=init_client_after_key_save,
                inputs=[session_id_state],
                outputs=[client_state, key_feedback]
            )
            
            
            def update_model_options(provider):
                if provider == "OpenAI":
                    return (
                        gr.update(
                            choices=["gpt-4o-mini", "gpt-4.1-mini"],
                            value="gpt-4o-mini"
                        ),
                        gr.update(
                            choices=["gpt-4o-mini", "gpt-4.1-mini"],
                            value="gpt-4o-mini"
                        ),
                        gr.update(
                            choices=["gpt-4o-mini", "gpt-4.1-mini"],
                            value="gpt-4o-mini"
                        )
                    )
                else:
                    return (
                        gr.update(
                            choices=["gpt-4o-mini", "gpt-4.1-mini"],
                            value="gpt-4o-mini"
                        ),
                        gr.update(
                            choices=["mistralai/Mistral-Nemo-Instruct-2407"],
                            value="mistralai/Mistral-Nemo-Instruct-2407"
                        ),
                        gr.update(
                            choices=["Qwen/Qwen2.5-VL-72B-Instruct"],
                            value="Qwen/Qwen2.5-VL-72B-Instruct"
                        )
                    )
            provider_selector.change(
                fn=lambda provider, session_id: (*update_model_options(provider), *check_keys_and_toggle_inputs(session_id)),
                inputs=[provider_selector, session_id_state],
                outputs=[
                    tool_call_selector,
                    response_model_selector,
                    vllm_model_selector,
                    msg,
                    send_btn,
                    image_input,
                    key_status
                ]
            )

            
            def set_user_models(session_id, provider, tool_model, response_model, vllm_model):
                openai_key = session_keys.get(session_id, {}).get("OPENAI_API_KEY")
                nebius_key = session_keys.get(session_id, {}).get("NEBIUS_API_KEY")
                provider = provider.strip().lower()

                if not openai_key:
                    return "⚠️ <span style='color:orange'>Please enter a valid OpenAI API key before setting the model.</span>"
                try:
                    _ = LLM_Client(session_id, sourceAI="openai", api_key=openai_key)
                except Exception as e:
                    logger.warning(f"❌ Invalid OpenAI key when setting model: {e}")
                    return "⚠️ <span style='color:orange'>OpenAI API key is invalid.</span>"

                if provider == "nebius":
                    if not nebius_key:
                        return "⚠️ <span style='color:orange'>Please enter a valid Nebius API key before setting the model.</span>"
                    try:
                        _ = LLM_Client(session_id, sourceAI="nebius", api_key=nebius_key)
                    except Exception as e:
                        logger.warning(f"❌ Invalid Nebius key when setting model: {e}")
                        return "⚠️ <span style='color:orange'>Nebius API key is invalid.</span>"

                if session_id not in session_keys:
                    logger.warning(f"❌ Invalid Nebius key when setting model: {e}")
                    session_keys[session_id] = {}

                session_keys[session_id]["provider"] = provider.capitalize()
                session_keys[session_id]["tool_call_model"] = tool_model 
                session_keys[session_id]["response_model"] = response_model
                session_keys[session_id]["VLLM_model"] = vllm_model

                logger.info(
                    f"✅ Models set for {session_id[:5]} | Provider: {provider}, Tool: {tool_model}, "
                    f"Response: {response_model}, VLLM_model: {vllm_model}"
                )
                return f"✅ Models saved for {session_id[:5]}"


            set_model_btn.click(
                fn=set_user_models,
                inputs=[session_id_state, provider_selector, tool_call_selector, response_model_selector, vllm_model_selector],
                outputs=[model_feedback]
            ).then(
                fn=check_keys_and_toggle_inputs,
                inputs=[session_id_state],
                outputs=[msg, send_btn, image_input, key_status]
            )
    return demo

if __name__ == "__main__":
    # folder_path = f"tmp"
    # if os.path.exists(folder_path):
    #     shutil.rmtree(folder_path)
    #     logger.info(f"Cleaned folder: {folder_path}")
    
    asyncio.get_event_loop().create_task(cleanup_old_sessions(threshold_seconds=600)) # 10 min
    interface = gradio_interface()
    interface.launch(debug=False)
