"""Offline regression tests. Run: python -m unittest discover -s backend/tests -p test_scope_context.py -v"""
import ast
import hashlib
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SERVICES = ROOT / "backend" / "app" / "services"
KB = Path(os.environ.get("SAP_SCOPE_TEST_KB", ROOT / "knowledge_base" / "documents"))
spec = importlib.util.spec_from_file_location("scope_context_under_test", SERVICES / "evidence_context.py")
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


def load_isolated(filename, values):
    """Load real service logic without booting the DB or downloading models."""
    tree = ast.parse((SERVICES / filename).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("backend.")
    )]
    exec(compile(tree, filename, "exec"), values)
    return values


class ScopeContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf = KB / "pdfdownload.pdf"
        if not cls.pdf.is_file():
            raise unittest.SkipTest("Place the original pdfdownload.pdf in knowledge_base/documents")
        cls.digest = hashlib.sha256(cls.pdf.read_bytes()).hexdigest()

    def chunk(self, **changes):
        result = dict(document_id="original-pdf", document_name="pdfdownload.pdf",
                      page_number=161, section="Co_Isx_Cc_Ctr_Prov_Services",
                      chunk_text="In Convergent Invoicing, a billable item class containing the interface component ALLOWANCE must be defined and the",
                      rerank_score=0.9)
        result.update(changes)
        return result

    def expand(self, chunks, digest=None):
        return context.expand_page_context(chunks, [KB], {"original-pdf": digest or self.digest})

    def test_allowance_passage_retains_app_scope_and_note_condition(self):
        item = self.expand([self.chunk()])[0]
        self.assertIn("App Implementation: Display Charging Data for Subscription Contracts", item["section"])
        self.assertNotIn("Co_Isx", item["section"])
        self.assertIn("optional integration", item["chunk_text"])
        self.assertIn("for the app to work as expected", context._normalize(item["chunk_text"]))
        self.assertIn("2022 FPS00", item["chunk_text"])
        self.assertEqual(item["rerank_score"], 0.9)

    def test_two_hits_on_same_expanded_page_are_deduplicated(self):
        hits = [self.chunk(), self.chunk(chunk_text="SAP Convergent Charging must be enabled and connected to your backend system.")]
        self.assertEqual(len(self.expand(hits)), 1)

    def test_changed_source_keeps_indexed_passage_without_new_context(self):
        original = self.chunk()
        item = self.expand([original], "0" * 64)[0]
        self.assertEqual(item["scope_context_status"], "passage_only")
        self.assertEqual(item["chunk_text"], original["chunk_text"])
        self.assertNotIn("Co_Isx", item["section"])

    def test_wrong_page_is_not_used_as_evidence(self):
        item = self.expand([self.chunk(page_number=1)])[0]
        self.assertEqual(item["scope_context_status"], "passage_only")

    def test_path_traversal_does_not_open_source(self):
        with patch.object(context, "_read_page") as read:
            self.expand([self.chunk(document_name="../pdfdownload.pdf")])
            read.assert_not_called()

    def test_budget_does_not_truncate_conditions_or_drop_hits(self):
        with patch.object(context, "MAX_CONTEXT_CHARS", 50):
            item = self.expand([self.chunk()])[0]
        self.assertEqual(item["scope_context_status"], "passage_only")
        self.assertEqual(item["chunk_text"], self.chunk()["chunk_text"])

    def test_missing_source_preserves_original_passage(self):
        item = self.expand([self.chunk(document_name="missing.pdf")])[0]
        self.assertEqual(item["scope_context_status"], "passage_only")

    def test_cloud_sync_and_stream_receive_same_scoped_evidence(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            if kwargs.get("stream"):
                return iter([types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="test output"))])])
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="test output"))])
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        settings = types.SimpleNamespace(LLM_PROVIDER="groq", LLM_API_KEY="", GROQ_API_KEY="gsk_test", LLM_MODEL="test-model")
        loaded = load_isolated("llm_service.py", {"settings": settings})
        service = loaded["llm_service"]
        evidence = self.expand([self.chunk()])
        with patch.dict(sys.modules, {"groq": types.SimpleNamespace(Groq=lambda **kw: client)}):
            self.assertEqual(service.generate_answer("CC and CI integration?", evidence), "test output")
            self.assertEqual("".join(service.stream_answer("CC and CI integration?", evidence)), "test output")
        self.assertEqual(calls[0]["messages"], calls[1]["messages"])
        self.assertIn("App Implementation", calls[0]["messages"][1]["content"])
        self.assertIn("optional integration", calls[0]["messages"][1]["content"])

    def test_local_fallback_keeps_scope_and_labels_extracts(self):
        settings = types.SimpleNamespace(LLM_PROVIDER="local", LLM_API_KEY="", GROQ_API_KEY="", LLM_MODEL="unused")
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            service = load_isolated("llm_service.py", {"settings": settings})["llm_service"]
        answer = service.generate_answer("CC and CI integration?", self.expand([self.chunk()]))
        self.assertIn("not a verified integration procedure", answer)
        self.assertIn("App Implementation", answer)
        self.assertIn("optional integration", answer)



if __name__ == "__main__":
    unittest.main()
