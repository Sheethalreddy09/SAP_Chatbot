"""Offline contract tests with mocked LLM responses, not semantic accuracy tests."""
import copy
import hashlib
import importlib.util
import json
import sys
import types
import unittest
from unittest.mock import patch
from test_scope_context import SERVICES, KB, context, load_isolated

spec = importlib.util.spec_from_file_location('verification_under_test', SERVICES / 'answer_verification_service.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
TEXT = 'SAP Convergent Charging can be integrated with Convergent Invoicing in SAP ERP.'
ITEM = dict(document_name='SAP CC.pdf', page_number=15, section='Integration', chunk_text=TEXT, rerank_score=.9)
CLAIM = dict(id='C1', text=TEXT, source_ids=['S1'])


def audit(claim_id='C1', quote=TEXT, source_id='S1', supported=True):
    return dict(id=claim_id, verdict='supported' if supported else 'unsupported',
                all_parts_supported=supported, scope_preserved=supported, direction_preserved=supported,
                evidence=[dict(source_id=source_id, quote=quote)] if supported else [])


def responses(claims=None, verdicts=None, coverage='complete'):
    return [dict(statements=[CLAIM] if claims is None else claims),
            dict(coverage=coverage, verdicts=[audit()] if verdicts is None else verdicts)]


def service_with(sequence, calls=None):
    sequence = iter(sequence)
    def request(prompt, payload):
        if calls is not None:
            calls.append((prompt, copy.deepcopy(payload)))
        value = next(sequence)
        if isinstance(value, Exception):
            raise value
        return value
    return v.AnswerVerificationService(request)


class VerificationTests(unittest.TestCase):
    def check(self, sequence, items=None, source_type='knowledge_base'):
        return service_with(sequence).generate('CC and CI integration?', [ITEM] if items is None else items, source_type)

    def test_valid_claim_uses_exact_supporting_excerpt(self):
        result = self.check(responses())
        self.assertEqual(result['verification_status'], 'checked')
        self.assertEqual(result['citations'][0]['snippet'], TEXT)
        self.assertIn('SAP CC.pdf, p. 15', result['answer'])

    def test_rejected_claim_removed_and_partial_explained(self):
        bad = dict(id='C2', text='CI automatically creates a class for each charge.', source_ids=['S2'])
        result = self.check(responses([CLAIM, bad], [audit(), audit('C2', supported=False)]),
                            [ITEM, {**ITEM, 'document_name':'Wrong.pdf'}])
        self.assertEqual(result['verification_status'], 'partial')
        self.assertNotIn(bad['text'], result['answer'])
        self.assertEqual(len(result['citations']), 1)
        self.assertIn(v.PARTIAL, result['answer'])

    def test_fabricated_quote_never_passes(self):
        result = self.check(responses(verdicts=[audit(quote='A completely invented supporting passage here.')]))
        self.assertEqual(result['accepted_count'], 0)
        self.assertEqual(result['citations'], [])

    def test_missing_verdict_fails_closed(self):
        self.assertEqual(self.check(responses(verdicts=[]))['verification_status'], 'unavailable')

    def test_duplicate_verdict_fails_closed(self):
        c2 = {**CLAIM, 'id':'C2'}
        self.assertEqual(self.check(responses([CLAIM,c2], [audit(),audit()]))['verification_status'], 'unavailable')

    def test_invalid_draft_source_fails_closed(self):
        self.assertEqual(self.check(responses([{**CLAIM, 'source_ids':['S9']}]))['verification_status'], 'unavailable')

    def test_invalid_audit_source_rejects_claim(self):
        self.assertEqual(self.check(responses(verdicts=[audit(source_id='S9')]))['accepted_count'], 0)

    def test_false_scope_flag_rejects_supported_verdict(self):
        verdict = {**audit(), 'scope_preserved':False}
        self.assertEqual(self.check(responses(verdicts=[verdict]))['accepted_count'], 0)

    def test_non_boolean_flag_fails_closed(self):
        verdict = {**audit(), 'scope_preserved':'true'}
        self.assertEqual(self.check(responses(verdicts=[verdict]))['verification_status'], 'unavailable')

    def test_timeout_never_exposes_draft(self):
        result = self.check([responses()[0], TimeoutError()])
        self.assertEqual(result['answer'], v.CHECK_FAILED)
        self.assertNotIn(TEXT, result['answer'])

    def test_invalid_json_never_exposes_draft(self):
        self.assertEqual(self.check([ValueError('invalid JSON')])['answer'], v.CHECK_FAILED)

    def test_empty_evidence_makes_no_provider_call(self):
        self.assertEqual(self.check([], [])['verification_status'], 'insufficient')

    def test_over_budget_source_is_skipped_not_truncated(self):
        self.assertEqual(self.check([], [{**ITEM,'chunk_text':'x'*12001}])['accepted_count'], 0)

    def test_statuses_contain_no_draft(self):
        calls=[]
        stream=service_with(responses(), calls).stream_check('integration?', [ITEM], 'knowledge_base')
        self.assertEqual(next(stream)['type'], 'status')
        self.assertEqual(len(calls),0)
        self.assertEqual(next(stream)['type'], 'status')
        self.assertEqual(len(calls),1)
        self.assertEqual(next(stream)['type'], 'checked')
        self.assertEqual(len(calls),2)

    def test_web_evidence_is_checked_and_cited(self):
        item=dict(title='SAP docs',url='https://help.sap.com/example',domain='help.sap.com',snippet=TEXT)
        result=self.check(responses(),[item],'web')
        self.assertEqual(result['citations'],[])
        self.assertEqual(result['web_sources'][0]['snippet'],TEXT)

    def test_real_app_page_blocks_generalized_allowance_even_if_audit_approves(self):
        pdf=KB/'pdfdownload.pdf'
        if not pdf.is_file(): self.skipTest('Original PDF required')
        quote='In Convergent Invoicing, a billable item class containing the interface component ALLOWANCE must be defined and the'
        chunk={**ITEM,'document_name':'pdfdownload.pdf','document_id':'doc','page_number':161,'chunk_text':quote}
        items=context.expand_page_context([chunk],[KB],{'doc':hashlib.sha256(pdf.read_bytes()).hexdigest()})
        bad={**CLAIM,'text':'Every CC-CI integration requires an ALLOWANCE billable item class.'}
        result=self.check(responses([bad],[audit(quote=quote)]),items)
        self.assertEqual(result['accepted_count'],0)
        scoped={**CLAIM,'text':'For the Display Charging Data for Subscription Contracts app, the documented optional allowance integration requires an ALLOWANCE billable item class in CI.'}
        self.assertEqual(self.check(responses([scoped],[audit(quote=quote)]),items)['accepted_count'],1)


class RagIntegrationTests(unittest.TestCase):
    def load(self,winner,verifier):
        items=[ITEM] if winner=='knowledge_base' else [dict(title='SAP',url='https://help.sap.com/',snippet=TEXT)]
        comparison=dict(winner=winner,is_in_rag_pipeline=True,kb_score=.9,web_score=.8,winning_score=.9,
                        top_kb_chunks=items if winner=='knowledge_base' else [],top_web_results=items if winner=='web' else [])
        ns=dict(settings=types.SimpleNamespace(TOP_K=10,WEB_FALLBACK_ENABLED=False,SIMILARITY_THRESHOLD=.6),
                sap_classifier=types.SimpleNamespace(classify=lambda q:(True,'SAP')),
                query_rewriter=types.SimpleNamespace(resolve_query=lambda q,**kw:dict(resolved_query=q,is_follow_up=True,conversation_topic='CC')),
                retrieval_service=types.SimpleNamespace(hybrid_search=lambda *a,**kw:items),
                reranking_service=types.SimpleNamespace(compare_and_select=lambda *a,**kw:comparison),
                grounding_service=types.SimpleNamespace(evaluate_grounding=lambda *a,**kw:.5),
                prepare_scoped_evidence=lambda db,chunks:chunks,answer_verification_service=verifier)
        with patch.dict(sys.modules,{'sqlalchemy.orm':types.SimpleNamespace(Session=object)}):
            return load_isolated('rag_service.py',ns)['rag_service']

    def test_sync_stream_parity_and_no_token_before_audit_for_kb_and_web(self):
        for winner in ['knowledge_base','web']:
            with self.subTest(winner=winner):
                sync=self.load(winner,service_with(responses())).process_query(None,'integration?')
                calls=[]
                stream=self.load(winner,service_with(responses(),calls)).stream_query(None,'integration?')
                events=[]
                for event in stream:
                    if event['type']=='token': self.assertEqual(len(calls),2)
                    events.append(event)
                self.assertEqual(''.join(e['content'] for e in events if e['type']=='token'),sync['answer'])
                self.assertEqual(events[-1]['answer'],sync['answer'])
                self.assertEqual(events[-1]['grounding_score'],.5)

    def test_failed_audit_emits_only_safe_notice_in_both_paths(self):
        for winner in ['knowledge_base','web']:
            sequence=[responses()[0],TimeoutError()]
            sync=self.load(winner,service_with(sequence)).process_query(None,'integration?')
            events=list(self.load(winner,service_with(sequence)).stream_query(None,'integration?'))
            self.assertEqual(sync['grounding_score'],0)
            self.assertEqual(events[-1]['grounding_score'],0)
            self.assertEqual(''.join(e['content'] for e in events if e['type']=='token'),v.CHECK_FAILED)
            self.assertEqual(events[-1]['citations'],[])


class ProviderContractTests(unittest.TestCase):
    def request(self,finish='stop'):
        calls=[];closed=[];options=[]
        def create(**kw):
            calls.append(kw)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason=finish,message=types.SimpleNamespace(content='{"statements":[]}'))])
        client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)),close=lambda:closed.append(True))
        settings=types.SimpleNamespace(LLM_PROVIDER='groq',LLM_API_KEY='',GROQ_API_KEY='fake-key',LLM_MODEL='configured-model')
        service=load_isolated('llm_service.py',{'settings':settings})['llm_service']
        def factory(**kw): options.append(kw);return client
        with patch.dict(sys.modules,{'groq':types.SimpleNamespace(Groq=factory)}):
            if finish=='stop':
                self.assertEqual(service.request_evidence_json('Return JSON',{'sources':[]} ),{'statements':[]})
            else:
                with self.assertRaises(ValueError): service.request_evidence_json('Return JSON',{})
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['model'],'configured-model')
        self.assertEqual(calls[0]['response_format'],{'type':'json_object'})
        self.assertFalse(calls[0]['stream'])
        self.assertEqual(options[0]['max_retries'],0)
        self.assertEqual(options[0]['timeout'],45)
        self.assertEqual(closed,[True])

    def test_bounded_json_request_uses_configured_model_only(self): self.request()
    def test_truncated_output_is_rejected(self): self.request('length')

if __name__=='__main__': unittest.main()
