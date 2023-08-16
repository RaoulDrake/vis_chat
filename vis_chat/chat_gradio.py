from typing import Sequence
import re

import gradio as gr
import guidance
import numpy as np
import os
from PIL import Image
from transformers import LlamaForCausalLM, LlamaTokenizer
import uuid

from vis_chat import prompt_templates, system_message_templates
from vis_chat import tools as tools_lib
from vis_chat.load_llm import load_llm, Vicuna


def format_response_images(text: str):
    formatted = re.sub(
        '(image/[-\w]*.png)',
        lambda m: f'![](file={m.group(0)})*{m.group(0)}*',
        text
    )
    return formatted


def build_guidance_program(
        llm,
        temperature: float = 0,
        max_new_tokens: int = 512,
        tools: Sequence[tools_lib.Tool] = (),
        num_cot_iterations: int = 5
):
    tools = list(tools)
    name_to_tool = {}
    tool_names_list = []
    tool_strings_list = []
    for t in tools:
        if hasattr(t, 'name') and hasattr(t, 'description') and hasattr(t, 'example'):
            name_to_tool[t.name] = t
            tool_names_list.append(t.name)
            tool_strings_list.append(
                f"- {t.name}\n"
                f"  - description: {t.description}\n"
                f"  - example: {t.example}")
    tool_strings = "\n".join(tool_strings_list)

    system_msgs = []
    if hasattr(llm, 'default_system_prompt'):
        system_msgs.append(llm.default_system_prompt)
    system_msgs += [
        system_message_templates.TOOL_BLOCK.format(tool_strings=tool_strings),
        system_message_templates.SYSTEM_MSG_SCRATCHPAD_V2
    ]
    system_msg = ('\n'.join(system_msgs)).strip()
    prompt_str = prompt_templates.VIS_CHAT_TEMPLATE_V3  # VIS_CHAT_TEMPLATE

    get_tool = tools_lib.get_tool_factory(name_to_tool)
    get_schema_regex = tools_lib.get_schema_regex_factory(name_to_tool)
    call_tool = tools_lib.call_tool_factory(name_to_tool)

    def has_input(tool_name: str) -> bool:
        tool = get_tool(tool_name)
        return hasattr(tool, 'schema') and tool.schema

    allowed_end_punctuation = [".", "?"]

    def endswith_punctuation(s: str) -> bool:
        return s.strip().endswith(tuple(allowed_end_punctuation))

    pseudo_eos = '#SCRATCHPAD-END'

    def is_cot_eos(s: str) -> bool:
        return pseudo_eos in s.upper()

    tool_call_attempts = [s + '(' for s in tool_names_list]

    def strip_action(s: str) -> str:
        return s.strip()[:-1]

    prompt_kwargs = dict(
        system_message=system_msg,
        temperature=dict(
            response=temperature, cot=temperature, action_input=temperature),
        max_new_tokens=dict(
            response=max_new_tokens, cot=max_new_tokens, action_input=max_new_tokens),
        stop_words=dict(cot=tool_call_attempts + [pseudo_eos]),
        choices=dict(
            cot=tool_call_attempts + [pseudo_eos],
            allowed_end_punctuation=allowed_end_punctuation
        ),
        pseudo_eos=pseudo_eos,
        tool_names_list=tool_names_list,
        num_cot_iterations=num_cot_iterations,
        get_schema_regex=get_schema_regex,
        call_tool=call_tool,
        has_input=has_input,
        lstrip=lambda s: s.lstrip(),
        endswith_punctuation=endswith_punctuation,
        is_cot_eos=is_cot_eos,
        strip_action=strip_action
    )

    prompt = guidance(prompt_str, llm=llm)

    return prompt, prompt_kwargs


def chat(
        base_model: str = "lmsys/vicuna-7b-v1.3",
        load_8bit: bool = True,
        model: LlamaForCausalLM = None,
        tokenizer: LlamaTokenizer = None,
        temperature: float = 0,
        max_new_tokens: int = 512,
        tools: Sequence[tools_lib.Tool] = (),
        num_cot_iterations: int = 5
):
    if model is None:
        llm = load_llm(base_model=base_model, load_8bit=load_8bit)
    else:
        if tokenizer is None:
            tokenizer = LlamaTokenizer.from_pretrained(base_model, legacy=False)
        llm = Vicuna(model=model, tokenizer=tokenizer, device=None)

    prompt, prompt_kwargs = build_guidance_program(
        llm=llm,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        tools=tools,
        num_cot_iterations=num_cot_iterations
    )

    def run_text(txt_: str, state_, chatbot_):
        txt_ = txt_.strip()
        history = []
        if state_:
            history = [{'user': u, 'assistant': a} for u, a in state_]
        result = prompt(
            query=txt_,
            history=history,
            **prompt_kwargs
        )

        response_cot = []
        if 'cot' in result and result['cot']:
            for i, step in enumerate(result['cot']):
                content = step['content'].strip()
                # if i == 0 and result['use_tool']:
                #     content = "I will use the #" + content
                if i == 0:
                    content = "Let me think step-by-step" + step['content'].rstrip()
                response_cot += [content]
                if 'action' in step:
                    action = step['action'].strip()
                    observation = step['observation'].strip()
                    response_cot += [(
                        f"Action: {action}"
                        f"({'' if 'action_input' not in step else step['action_input'].strip()})"
                        f"\nObservation: {observation}")]
                    if action == "#generate-image":
                        response_cot += [
                            "I will use the #vqa tool to get a description "
                            "of the style and content of the generated image."]
        response_cot = "\n".join(response_cot)
        response_cot = response_cot.strip()
        response_cot_formatted = format_response_images(response_cot)
        response_cot_formatted = "#SCRATCHPAD-START\n" + response_cot_formatted + "\n#SCRATCHPAD-END"

        response = result['response'].strip()
        if 'end_punctuation' in result:
            end_punctuation = result['end_punctuation']
            if len(end_punctuation) > 1:
                end_punctuation = end_punctuation[-1]
            response += end_punctuation
        response_formatted = format_response_images(response)
        response_formatted = "\n\n".join([response_cot_formatted, response_formatted])
        chatbot_ = chatbot_ + [(txt_, response_formatted)]
        state_ = state_ + [(txt_, response)]
        print(f"\nProcessed run_text, Input text: {txt_}\nCurrent state: {state_}")
        return chatbot_, state_

    def run_image(image_, state_, txt_, chatbot_):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        print("======>Auto Resize Image...")
        img = Image.open(image_.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image from {width}x{height} to {width_new}x{height_new}")
        user_prompt = f'I have uploaded image "{image_filename}". ' \
                      f'You should use tools to finish following tasks, rather than imagination. ' \
                      f'If you understand, say "Received".'
        assistant_prompt = "Received."
        chatbot_ = chatbot_ + [(f"![](file={image_filename})*{image_filename}*", assistant_prompt)]
        state_ = state_ + [(user_prompt, assistant_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state_}")
        return chatbot_, state_, txt_

    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="Visual Vicuna")
        state = gr.State([])
        with gr.Row(visible=True) as input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton(label="🖼️", file_types=["image"])

        txt.submit(run_text, [txt, state, chatbot], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(run_image, [btn, state, txt, chatbot], [chatbot, state, txt])
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    demo.launch(server_name="0.0.0.0", server_port=7861)


if __name__ == '__main__':
    vqa = tools_lib.VisualQuestionAnswering('cuda', llm_model="vicuna-7b")  # "vicuna-13b"
    sd = tools_lib.Text2Image('cuda')
    TOOLS: Sequence[tools_lib.Tool] = [vqa.inference, sd.inference]
    chat(
        model=vqa.model.language_model,
        tokenizer=vqa.processor.tokenizer,
        tools=TOOLS,
        num_cot_iterations=5
    )
