from typing import Callable, Dict, List, Optional, Union
import json
import os
import re
import string
import uuid

from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

Tool = Callable


def tool(
        name: str,
        description: str,
        example: str,
        schema: Union[List[Dict[str, str]], bool],
        schema_regex: str
):
    def decorator(func):
        func.name = name
        func.description = description
        func.example = example
        func.schema = schema
        func.schema_regex = schema_regex
        return func

    return decorator


@tool(
    name="<tool>",
    description="",
    example="",
    schema=[{"name": "input", "type": "str"}],  # dummy value, not used
    schema_regex='\<input\>'
)
def dummy_tool(input_=None) -> str:
    return "<action-result>"


def get_tool_factory(
        name_to_tool: Dict[str, Tool]
) -> Callable[[str], Optional[Tool]]:
    def get_tool(tool_name: str) -> Optional[Tool]:
        tool_name = tool_name.strip().strip('"')
        _tool = name_to_tool.get(tool_name, None)
        return _tool

    return get_tool


def get_schema_regex_factory(
        name_to_tool: Dict[str, Tool],
        default_schema_regex: str = '[^\)]+'
) -> Callable[[str], Optional[str]]:
    get_tool = get_tool_factory(name_to_tool)

    def get_schema_regex(tool_name: str) -> str:
        _tool = get_tool(tool_name)
        if _tool is not None and hasattr(_tool, 'schema_regex'):
            return _tool.schema_regex
        return default_schema_regex

    return get_schema_regex


def call_tool_factory(
        name_to_tool: Dict[str, Tool],
        verbose: bool = False
) -> Callable[[str, str], str]:
    get_tool = get_tool_factory(name_to_tool)

    def call_tool(tool_name: str, tool_input: str) -> str:
        _tool = get_tool(tool_name)
        if _tool is not None:
            if _tool.schema and (_tool.name != "<tool-name>"):
                try:
                    tool_input.strip(')')
                    tool_input_dict: Dict = json.loads(tool_input)
                    tool_input_dict_keys = list(tool_input_dict.keys())
                    schema_keys = [s['name'] for s in _tool.schema]
                    assert all([(s in tool_input_dict_keys) for s in schema_keys])
                    tool_input_dict = {
                        k.replace(' ', '_').replace('-', '_').lower(): v
                        for k, v in tool_input_dict.items()
                    }
                except Exception as e:
                    if verbose:
                        print(e)
                    exception_msg = f"{tool_input} is not a valid input for the {tool_name} tool."
                    example_input = ""
                    if hasattr(_tool, 'example'):
                        example_input = " Consider this example: "
                        example_input += re.sub('Action: ', '', _tool.example)
                    exception_msg += example_input
                    return exception_msg
                return _tool(**tool_input_dict)
            return _tool()
        return f"{tool_name} is not a valid tool."

    return call_tool


class VisualQuestionAnswering:

    def __init__(
            self,
            device: str,
            load_in_8bit: bool = True,
            verbose: bool = False,
            llm_model: str = "vicuna-7b"
    ):
        print(f"Initializing VisualQuestionAnswering to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.device = device
        self.processor = InstructBlipProcessor.from_pretrained(f"Salesforce/instructblip-{llm_model}")
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            f"Salesforce/instructblip-{llm_model}",
            torch_dtype=self.torch_dtype,
            load_in_8bit=load_in_8bit
        )
        if not load_in_8bit:
            self.model.to(self.device)
        self.history = {}
        self.verbose = verbose

    # @tool(
    #     name="Visual Question Answering",
    #     description='Useful when assistant or user needs an answer for a question based on an image. '
    #                 'Assistant should phrase question in an unambiguous manner, so that it can be answered '
    #                 'without any further context, other than the image itself. '
    #                 'Observation from this tool may not be entirely reliable, '
    #                 'so assistant considers if observation is plausible or incorrect/incomplete '
    #                 'and tries to overcome the shortcomings of this tool, e.g., '
    #                 'by using it with paraphrased or counterfactual questions to check for consistency.',
    #     example='Action: Visual Question Answering({'
    #             '"image path": "image/xxx.png", '
    #             '"question": "Describe the content of the image with as much detail as possible."'
    #             '})',
    #     schema=[
    #         {"name": "image path", "type": "str"},
    #         {"name": "question", "type": "str"}
    #     ],
    #     schema_regex='{"image path": "image/[a-z0-9]+\.png", "question": "[^"]+"}'
    # )
    @tool(
        name="#vqa",
        description='Visual Question Answering system. Useful for image related questions.',
                    # 'Assistant should phrase question in an unambiguous manner, '
                    # 'so that it can be answered without any further context, other than the image itself. '
                    # 'Assistant will use #vqa to understand an image before talking about it and '
                    # 'will never add any fake details when talking about an image.',
                    # 'Assistant must use #vqa before talking about an image.',
        example='#vqa({'
                '"image-path": "image/xxx.png", '
                '"question": "What is unusual about this image?"'
                '})',  # <output>answer</output>
                # '<output>answer from the Visual Question Answering system here</output>',
        schema=[
            {"name": "image-path", "type": "str"},
            {"name": "question", "type": "str"}
        ],
        schema_regex='{"image-path": "image/[a-z0-9]+\.png", "question": "[^"]+"}'
    )
    def inference(self, image_path, question):
        if (image_path in self.history) and (question in self.history[image_path]):
            return "You already asked this question about this image. " \
                   "Please consider asking a different question about this image."
        elif image_path not in self.history:
            self.history[image_path] = []
        self.history[image_path].append(question)

        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=raw_image, text=question, return_tensors="pt").to(self.device, self.torch_dtype)

        tmp_prepare = None
        if hasattr(self.model.language_model, '_orig_prepare_method'):
            # reverse guidance monkey-patching of LlamaForCausalLM prepare_inputs_for_generation method
            tmp_prepare = self.model.language_model.prepare_inputs_for_generation
            self.model.language_model.prepare_inputs_for_generation = self.model.language_model._orig_prepare_method
        tmp_update = None
        if hasattr(self.model.language_model, '_orig_update_method'):
            # reverse guidance monkey-patching of LlamaForCausalLM _update_model_kwargs_for_generation method
            tmp_update = self.model.language_model._update_model_kwargs_for_generation
            self.model.language_model._update_model_kwargs_for_generation = self.model.language_model._orig_update_method

        out = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
            use_cache=True
        )

        if hasattr(self.model.language_model, '_orig_prepare_method'):
            # reinstate guidance monkey-patching of LlamaForCausalLM prepare_inputs_for_generation method
            self.model.language_model.prepare_inputs_for_generation = tmp_prepare
        if hasattr(self.model.language_model, '_orig_update_method'):
            # reinstate guidance monkey-patching of LlamaForCausalLM _update_model_kwargs_for_generation method
            self.model.language_model._update_model_kwargs_for_generation = tmp_update

        answer = self.processor.batch_decode(out, skip_special_tokens=True)[0]

        if self.verbose:
            print(f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
                  f"Output Answer: {answer}")
        raw_image.close()
        answer = answer.strip()
        if answer[-1] not in string.punctuation:
            answer += '.'
        return answer


class Text2Image:
    def __init__(self, device, verbose: bool = False):
        print(f"Initializing Text2Image to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=self.torch_dtype,
            safety_checker=None, requires_safety_checker=False
        )
        self.pipe.to(device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'
        self.verbose = verbose

    # @tool(
    #     name="Generate Image From Input Text",
    #     description=(
    #             'Useful when user wants assistant to generate an image from an input text and save it to a file. '
    #             'The input to this tool should be a string, '
    #             'representing the text used to generate the image (should be less than 50 words). '
    #             'A good strategy to formulate an input text is '
    #             'to imagine a caption that describes how the generated image should look like. '
    #             'The observation returned from this tool is the file name of the generated image '
    #             'and the assistant will remember to mention this file name when responding to the user. '
    #             'Since the generated image may not match desired specifications, other tools may be used '
    #             'by the assistant to check for consistency between desired specifications and generated image. '
    #             'The assistant does NOT use this tool autonomously, but rather only if the user says something like '
    #             '"create an image of [...]" or "generate an image in the style of [...]" to the assistant.'
    #     ),
    #     example='Action: Generate Image From Input Text({'
    #             '"input text": "a photo of an astronaut riding a horse on mars"})',
    #     schema=[{"name": "input text", "type": "str"}],
    #     schema_regex='{"input text": "[^"]+"}'
    # )
    @tool(
        name="#generate-image",
        description=(
                'Text-to-image system. '
                'Useful for generating an image and saving it to a file. '
                'Output of #generate-image is the generated image file name, e.g., image/yyy.png.'
                # 'Assistant never uses #generate-image on its own, '
                # 'only if user tells it to create or generate an image.'

                # 'Assistant will use #generate-image if user asks assistant to create/generate an image and '
                # 'assistant will include the file name of the generated image in its response.'
                # 'Useful for creating or generating images like photos, paintings, etc.'
                # 'Useful if user says something like "create an image" or "generate an image".'
                # 'Assistant should phrase input text in an unambiguous manner, '
                # 'so that it can be understood without any further context, other than the input text itself.'
                # 'Assistant will report all file names of generated images back to user, e.g., by responding '
                # 'I have generated the following image: "image/xxx.png".'
                # 'After using #generate-image, assistant should use #vqa '
                # 'to check for consistency between generated image and input text.'
        ),
        example='#generate-image({'
                '"input-text": "a photo of an astronaut riding a horse on mars"})',
                # '<output>image/yyy.png</output>',
        schema=[{"name": "input-text", "type": "str"}],
        schema_regex='{"input-text": "[^"]+"}'
    )
    def inference(self, input_text):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        prompt = input_text + ', ' + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(image_filename)
        if self.verbose:
            print(
                f"\nProcessed Text2Image, Input Text: {input_text}, Output Image: {image_filename}")
        return image_filename
