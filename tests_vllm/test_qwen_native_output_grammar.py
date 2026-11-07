import importlib.util
from pathlib import Path
import unittest
import xgrammar as xg
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

class QwenNativeOutputGrammarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'patches/qwen3_engine_tool_parser.py'
        spec = importlib.util.spec_from_file_location('candidate_qwen_parser', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.parser = object.__new__(module.Qwen3EngineToolParser)
        cls.compiler = xg.GrammarCompiler(xg.TokenizerInfo(
            [bytes([i]) for i in range(256)], stop_token_ids=[0]))
        cls.request = ResponsesRequest(model='qwen',input='x',tool_choice='auto',
            tools=[{'type':'function','name':'read_source','parameters':{
                'type':'object','properties':{},'additionalProperties':False}}],
            text={'format':{'type':'json_schema','name':'result','schema':{
                'type':'object','properties':{'ok':{'type':'boolean'}},
                'required':['ok'],'additionalProperties':False}}})

    def accepted(self, text, *, reasoning):
        tag = self.parser.get_structural_tag(self.request, reasoning=reasoning)
        matcher = xg.GrammarMatcher(self.compiler.compile_structural_tag(tag))
        return matcher.accept_string(text)

    def test_final_json_and_tool_after_reasoning(self):
        prefix = 'source is verified</think>\n\n'
        self.assertTrue(self.accepted(prefix + '{"ok":true}', reasoning=True))
        self.assertTrue(self.accepted(prefix + '<tool_call>\n<function=read_source>\n\n</function>\n</tool_call>', reasoning=True))

    def test_final_json_and_tool_without_reasoning(self):
        self.assertTrue(self.accepted('{"ok":true}', reasoning=False))
        self.assertTrue(self.accepted('<tool_call>\n<function=read_source>\n\n</function>\n</tool_call>', reasoning=False))

    def test_schema_and_declared_tool_remain_strict(self):
        for reasoning in (False,True):
            prefix = 'verified</think>\n\n' if reasoning else ''
            for invalid in ('{"ok":"wrong"}', '{"wrong":true}',
                            '<tool_call>\n<function=unknown>\n\n</function>\n</tool_call>'):
                with self.subTest(reasoning=reasoning,invalid=invalid):
                    self.assertFalse(self.accepted(prefix + invalid,reasoning=reasoning))

if __name__=='__main__':
    unittest.main()
