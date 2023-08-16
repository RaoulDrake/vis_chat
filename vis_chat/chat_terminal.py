from typing import Optional, Sequence

import guidance

from vis_chat import prompt_templates, system_message_templates
from vis_chat import tools as tools_lib
from vis_chat.load_llm import load_llm


def chat(
        base_model: str = "lmsys/vicuna-7b-v1.3",
        load_8bit: bool = True,
        temperature: float = 0,
        max_new_tokens: int = 512,  # 300,
        stop: str = "END",
        greeting: Optional[str] = None,  # 'Hi there! How can I help you?',
        tools: Sequence[tools_lib.Tool] = (),
        num_cot_iterations: int = 5
):
    if greeting is None:
        greeting = False

    llm = load_llm(base_model=base_model, load_8bit=load_8bit)

    tools = list(tools)
    name_to_tool = {}
    tool_names_list = []
    tool_strings_list = []
    for t in tools:
        if hasattr(t, 'name') and hasattr(t, 'description') and hasattr(t, 'example'):
            name_to_tool[t.name] = t
            tool_names_list.append(t.name)
            # Do not include dummy tool in tool def of system message
            if t.name not in ["<action>", "<tool>", "<tool-name>"]:
                tool_strings_list.append(
                    f"> name: {t.name}\n"
                    f"  description: {t.description}\n"
                    f"  example: {t.example}")
    tool_strings = "\n".join(tool_strings_list)

    system_msgs = []
    if hasattr(llm, 'default_system_prompt'):
        system_msgs.append(llm.default_system_prompt)
    system_msgs += [
        # system_message_templates.CAPABILITIES_LIMITATIONS_BLOCK,
        # system_message_templates.RULES_GUIDELINES_PRINCIPLES_BLOCK,
        system_message_templates.TOOL_BLOCK_STRUCT.format(tool_strings=tool_strings),
        # system_message_templates.SYSTEM_MSG_AIO_STRUCT,
        # system_message_templates.SYSTEM_MSG_EXAMPLE_CHAT,
        # system_message_templates.SYSTEM_MSG_SCRATCHPAD,
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

    pseudo_eos = '#SCRATCHPAD-END'  # '\n\n'

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
    if greeting:
        print(f"ASSISTANT: {greeting}")
    conversation = []
    query = input('USER: ')
    while query != stop:
        result = prompt(
            query=query,
            history=conversation,
            **prompt_kwargs
        )
        response_cot = []
        if 'cot' in result and result['cot']:
            for i, step in enumerate(result['cot']):
                content = step['content'].strip()
                if i == 0 and result['use_tool']:
                    content = "I will use the #" + content
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
                            "I will use the #vqa tool to describe the style and content of the generated image."]
        response_cot = "\n".join(response_cot)
        response_cot = response_cot.strip()
        print(f"\nASSISTANT [COT]: {response_cot}\n")
        response = result['response'].strip()
        if 'end_punctuation' in result:
            end_punctuation = result['end_punctuation']
            if len(end_punctuation) > 1:
                end_punctuation = end_punctuation[-1]
            response += end_punctuation
        print(f"ASSISTANT: {response}\n")
        conversation.append({'user': query, 'assistant': response})
        query = input('USER: ')


if __name__ == '__main__':
    BASE_MODEL = "lmsys/vicuna-13b-v1.3"
    vqa = tools_lib.VisualQuestionAnswering('cuda')
    sd = tools_lib.Text2Image('cuda')
    TOOLS: Sequence[tools_lib.Tool] = [vqa.inference, sd.inference]
    chat(tools=TOOLS, num_cot_iterations=5)
