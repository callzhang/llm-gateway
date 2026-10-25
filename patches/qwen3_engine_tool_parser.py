# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import xgrammar as xg
from xgrammar.structural_tag import StructuralTag, OrFormat, JSONSchemaFormat, SequenceFormat
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_tool_dicts

from vllm.parser.engine.registered_adapters import Qwen3ParserToolAdapter


class Qwen3EngineToolParser(Qwen3ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "qwen_3_coder"

    def get_structural_tag(self, request, *, reasoning=False):
        if not (isinstance(request, ResponsesRequest) and request.tools and (request.tool_choice == 'auto') and (request.text is not None) and (request.text.format is not None) and (request.text.format.type == 'json_schema')):
            return super().get_structural_tag(request, reasoning=reasoning)
        final_constraint = request.extract_structured_outputs()
        functions = construct_tool_dicts(request.tools, request.tool_choice)
        if not functions:
            return super().get_structural_tag(request, reasoning=reasoning)
        tool_tag = xg.get_model_structural_tag(model=self.structural_tag_model, tools=functions, tool_choice='required', reasoning=reasoning)
        final_json = JSONSchemaFormat(json_schema=final_constraint.json)
        if reasoning:
            # Keep the reasoning prefix outside the choice. Otherwise emitting
            # reasoning commits the grammar to a tool and excludes final JSON.
            prefix, tool_format = tool_tag.format.elements[:-1], tool_tag.format.elements[-1]
            return StructuralTag(format=SequenceFormat(elements=[
                *prefix, OrFormat(elements=[tool_format, final_json]),
            ]))
        return StructuralTag(format=OrFormat(elements=[tool_tag.format, final_json]))
