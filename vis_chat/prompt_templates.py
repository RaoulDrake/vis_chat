
BASE_TEMPLATE = '''
{{~#system~}}
{{system_message}}
{{~/system~}}

{{#each history~}}
{{#user~}}
{{this.user}}
{{~/user~}}
{{#assistant~}}
{{this.assistant}}
{{~/assistant}}
{{~/each~}}

{{#user~}}
{{query}}
{{~/user}}'''

CHAT_TEMPLATE = BASE_TEMPLATE + '''
{{~#assistant~}}
{{gen 'response' temperature=temperature.response max_tokens=max_new_tokens.response}}
{{~/assistant}}'''

VIS_CHAT_TEMPLATE = BASE_TEMPLATE + '''
{{~#assistant~}}
Let me think and reason step-by-step, and, if necessary, use tools:
{{#geneach 'cot' num_iterations=num_cot_iterations~}}
{{lstrip (gen 'this.content' temperature=temperature.cot max_tokens=max_new_tokens.cot stop=stop_words.cot)}}{{#block hidden=True}}{{select 'this.action_or_response' options=stop_words.cot}}{{~/block}}
{{~#if ((strip this.action_or_response)==(strip actual_response))~}}
{{break}}
{{~else~}}
{{~#if ((strip this.action_or_response)!=(strip dummy_response))~}}
Action: {{strip (select 'this.action' options=tool_names_list)}}(
{{~#if (has_input (strip this.action))~}}
    {{strip (gen 'this.action_input' temperature=temperature.action_input max_tokens=max_new_tokens.action_input pattern=(get_schema_regex (this.action)))}}
{{~else~}}
{{set 'this.action_input' ""}}
{{~/if~}}
)
Observation: {{set 'this.observation' (call_tool (strip this.action) (strip this.action_input))~}}
{{this.observation}}
{{~else~}}
{{dummy_response}}:"
{{~/if}}
{{~/if}}
{{~/geneach}}

Here's my response:
{{lstrip (gen 'response' temperature=temperature.response max_tokens=max_new_tokens.response)}}
{{~/assistant}}'''

VIS_CHAT_TEMPLATE_V2 = BASE_TEMPLATE + '''
{{~#assistant~}}
{{#geneach 'cot' num_iterations=num_cot_iterations~}}
{{lstrip (gen 'this.content' temperature=temperature.cot max_tokens=max_new_tokens.cot stop=stop_words.cot)}}
{{~#block hidden=True}}{{select 'this.action_or_eos' options=choices.cot}}{{~/block}}
{{~#if (this.action_or_eos==pseudo_eos)~}}
{{break}}
{{~else~}}
{{~set 'this.action' this.action_or_eos~}}
{{this.action}}(
{{~#if (has_input (strip this.action))~}}
    {{strip (gen 'this.action_input' temperature=temperature.action_input max_tokens=max_new_tokens.action_input pattern=(get_schema_regex (this.action)))}}
{{~else~}}
{{set 'this.action_input' ""}}
{{~/if~}}
)
```
{{set 'this.observation' (call_tool (strip this.action) (strip this.action_input))~}}
{{this.observation}}
```
{{~/if}}
{{~/geneach}}
{{~/assistant}}'''

VIS_CHAT_TEMPLATE_V3 = BASE_TEMPLATE + '''
{{~#assistant~}}
{{~set 'has_used_action' False~}}
{{~set 'has_generated_image' False~}}
{{~set 'has_closed_scratchpad' False~}}
{{~set 'use_tool' False~}}
#SCRATCHPAD-START
{{#geneach 'cot' num_iterations=num_cot_iterations~}}
{{gen 'this.content' temperature=temperature.cot max_tokens=max_new_tokens.cot stop=stop_words.cot}}
{{~#block hidden=True~}}
{{~strip (select 'this.action_or_eos' options=choices.cot)}}
{{~/block~}}
{{~#if (is_cot_eos (strip this.action_or_eos))~}}
{{~set 'has_closed_scratchpad' True~}}
{{~break}}
{{~else~}}
{{~set 'this.action' strip_action(this.action_or_eos)~}}
{{~this.action~}}
{{~set 'has_used_action' True~}}
(
{{~#if (has_input this.action)~}}
    {{strip (gen 'this.action_input' temperature=temperature.action_input max_tokens=max_new_tokens.action_input pattern=(get_schema_regex (this.action)))}}
{{~else~}}
{{set 'this.action_input' ""}}
{{~/if~}}
) {{set 'this.observation' (call_tool this.action (strip this.action_input))~}}
<output>{{strip this.observation}}</output>
{{~#if this.action=="#generate-image"~}}
{{set 'has_generated_image' True}}

I will use the #vqa tool to get a description of the content and style of the generated image.
{{~/if~}}
{{~/if}}
{{~/geneach}}I will now write my response to the user after the end of the scratchpad.
{{~#if has_used_action}} I will stay loyal to the tool outputs rather than faking their content.
{{~/if}}
{{~#if has_generated_image}} I will also include all file names of generated images in my response, e.g., by starting the response with 'I have generated the following images'.
{{~/if}}
#SCRATCHPAD-END

Here's my response: {{lstrip (gen 'response' temperature=temperature.response max_tokens=max_new_tokens.response)}}
{{~#if (endswith_punctuation response)==False~}}
{{strip (select 'end_punctuation' options=choices.allowed_end_punctuation)}}
{{~/if~}}
{{~/assistant}}'''


_CONFIRM_ACTION = '''
{{strip this.action_or_eos~}}
{{#block hidden=True~}}
[Do I want to use this tool right now? {{strip (select 'this.use_tool' options=["Yes", "No"])}}]
{{~/block~}}
{{#if (strip this.use_tool)=="Yes"~}}
{{~set 'has_used_action' True~}}
{{~set 'this.action' strip(this.action_or_eos)~}}
(
{{~#if (has_input this.action)~}}
    {{strip (gen 'this.action_input' temperature=temperature.action_input max_tokens=max_new_tokens.action_input pattern=(get_schema_regex (this.action)))}}
{{~else~}}
{{set 'this.action_input' ""}}
{{~/if~}}
) {{set 'this.observation' (call_tool this.action (strip this.action_input))~}}
<output>{{strip this.observation}}</output>{{#if this.action=="#generate-image"~}}{{set 'has_generated_image' True}}{{/if}}
{{~/if}}
{{~/if}}
{{~/geneach}}I will now close the scratchpad and respond to the user. {{#if has_used_action~}}
I will stay loyal to the tool outputs rather than faking their content. 
{{~/if}}{{#if has_generated_image}} I will also cite the file name of the generated image in my response. 
{{~/if}}
#SCRATCHPAD-END
'''


_CONFIRM_COT_END = '''
{{#block hidden=True~}}
{{strip this.action_or_eos~}}
[Do I want to end the scratchpad now? {{strip (select 'this.close_scratchpad' options=["Yes", "No"])}}]
{{~/block~}}
{{#if (strip this.close_scratchpad)=="Yes"~}}
{{set 'has_closed_scratchpad' True}}
{{~break}}
{{~else~}}
{{strip this.action_or_eos}}
{{~/if}}
'''
