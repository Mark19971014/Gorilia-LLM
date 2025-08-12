import json
import os
import time
from typing import Any
from copy import deepcopy

from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.model_style import ModelStyle
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI, RateLimitError

# Base schema; we will clone and inject the allowed tool name enum per request
BASE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "response_format",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A detailed explanation outlining the reasoning behind invoking the tool calls and any missing information. It should be between 50 and 500 words. It should be finished with an argument like 'tools should be called: 1- ... 2- ..., etc'",
                },
                "tool_calls": {
                    "type": "array",
                    "description": "A list of tool calls to be executed without any repetition. There is a penalty for having unnecessary repetitions.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "A unique identifier in UUID format for this tool call."
                            },
                            "type": {
                                "type": "string",
                                "enum": ["function"],
                                "description": "Must always be 'function' as per OpenAI's tool call format."
                            },
                            "function": {
                                "type": "object",
                                "description": "A tool that needs to be executed.",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The name of the tool being invoked."
                                    },
                                    "arguments": {
                                        "type": "string",
                                        "description": "A stringified JSON object containing the required parameters for the function."
                                    }
                                },
                                "required": ["name", "arguments"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["id", "type", "function"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["reasoning", "tool_calls"],
            "additionalProperties": False
        }
    }
}


class AzureOpenAICompletionsHandlerSO(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Completions

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("ENDPOINT_URL", "https://aoai-l-swedencentral.openai.azure.com/"),
            api_version="2025-01-01-preview",
            azure_ad_token_provider=token_provider,
        )

        #self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-2024-08-06-global")
        #self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini-2024-07-18")
        self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
       
        # Reverse mapping and name set are rebuilt for each test via _compile_tools
        self._openapi_to_gorilla: dict[str, str] = {}
        self._gorilla_names: set[str] = set()

    def decode_ast(self, result, language="Python"):
        if "FC" in self.model_name or self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language)

    def decode_execute(self, result):
        if "FC" in self.model_name or self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        # Respect a per-call response_format if already provided; otherwise use base
        if "response_format" not in kwargs:
            kwargs["response_format"] = BASE_RESPONSE_FORMAT
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()
        return api_response, end_time - start_time

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        # Build a per-call response_format that constrains function.name to allowed tool names
        rf = deepcopy(BASE_RESPONSE_FORMAT)
        try:
            allowed_names = [t["function"]["name"] for t in tools] if tools else []
            fn_name_schema = (
                rf["json_schema"]["json_schema"]["schema"]["properties"]
                  ["tool_calls"]["items"]["properties"]["function"]["properties"]["name"]
            )
            if allowed_names:
                fn_name_schema["enum"] = allowed_names
        except Exception:
            # If schema injection fails, we still proceed with the base format
            pass

        kwargs = {
            "messages": message,
            "model": self.deployment,
            "temperature": self.temperature,
            "store": False,
            "response_format": rf,
        }

        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        inference_data["tools"] = tools

        # Build reverse name map: OpenAPI-safe -> Gorilla original (restores dots)
        self._gorilla_names = {f["name"] for f in functions}  # e.g., {"math.hypot", "math.factorial"}
        self._openapi_to_gorilla.clear()
        for t, f in zip(tools, functions):
            # OpenAI tool item shape: {"type": "function", "function": {"name": "...", ...}}
            openapi_name = t["function"]["name"]  # e.g., "math_hypot"
            gorilla_name = f["name"]              # e.g., "math.hypot"
            self._openapi_to_gorilla[openapi_name] = gorilla_name

        return inference_data

    def _parse_query_response_FC(self, api_response: any) -> dict:
        try:
            # Structured output (json_schema) path
            content_raw = api_response.choices[0].message.content
            parsed_json = json.loads(content_raw)

            tool_calls = parsed_json.get("tool_calls", [])
            # Optional: keep reasoning for local logs if needed
            _reasoning = parsed_json.get("reasoning", "")

            model_responses = []
            tool_call_ids = []

            for tool_call in tool_calls:
                fn = tool_call["function"]
                name = fn["name"]

                if isinstance(name, str) and name.startswith("functions."):
                    name = name[len("functions."):]

                if name in self._openapi_to_gorilla:
                    name = self._openapi_to_gorilla[name]
                else:
                    dotted = name.replace("_", ".")
                    if dotted in self._gorilla_names:
                        name = dotted

                # 3) Parse arguments (can be str or dict depending on model behavior)
                args = fn.get("arguments", {})
                args_dict = json.loads(args) if isinstance(args, str) else args

               
                model_responses.append({name: json.dumps(args_dict)})
                tool_call_ids.append(tool_call.get("id", ""))

            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": content_raw,
                "tool_calls": tool_calls,
            }

            return {
                "model_responses": model_responses,
                "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
                "tool_call_ids": tool_call_ids,
                "input_token": api_response.usage.prompt_tokens,
                "output_token": api_response.usage.completion_tokens,
            }

        except Exception as e:
            print(f"[ERROR] Failed to parse structured output: {e}")
            return {
                "model_responses": [],
                "model_responses_message_for_chat_history": {},
                "tool_call_ids": [],
                "input_token": api_response.usage.prompt_tokens,
                "output_token": api_response.usage.completion_tokens,
            }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # Add the execution results to the current round result, one at a time
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    def _add_reasoning_content_if_available_FC(
        self, api_response: Any, response_data: dict
    ) -> None:
        """
        OpenAI models don't show reasoning content in the api response,
        but many other models that use the OpenAI interface do, such as DeepSeek and Grok.
        This method is included here to avoid code duplication.
        """
        message = api_response.choices[0].message

        # Preserve tool_call information but strip unsupported fields before inserting into chat history.
        if getattr(message, "tool_calls", None):
            assistant_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
            response_data["model_responses_message_for_chat_history"] = assistant_message

        elif hasattr(message, "reasoning_content"):
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": message.content,
            }

        if hasattr(message, "reasoning_content"):
            response_data["reasoning_content"] = message.reasoning_content

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}

        return self.generate_with_backoff(
            messages=inference_data["message"],
            model=self.deployment,
            temperature=self.temperature,
            store=False,
        )

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: any) -> dict:
        return {
            "model_responses": api_response.choices[0].message.content,
            "model_responses_message_for_chat_history": api_response.choices[0].message,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )

        return inference_data

    def _add_reasoning_content_if_available_prompting(
        self, api_response: Any, response_data: dict
    ) -> None:
        """
        OpenAI models don't show reasoning content in the api response,
        but many other models that use the OpenAI interface do, such as DeepSeek and Grok.
        This method is included here to avoid code duplication.
        """
        message = api_response.choices[0].message
        if hasattr(message, "reasoning_content"):
            response_data["reasoning_content"] = message.reasoning_content
            # Reasoning content should not be included in the chat history
            response_data["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": str(response_data["model_responses"]),
            }
