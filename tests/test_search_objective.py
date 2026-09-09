"""Finite-difference checks of the actual search objective and token ranking.

Extract pure numerical functions so these tests need PyTorch but no model weights,
datasets installation, or Hugging Face cache initialization.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
NAMES={'normalized_linear_decay_weights','reduce_rollout_soft_collapse',
       'reduce_rollout_tv','sequence_log_probs_from_embeds',
       'hard_objective_from_suffix_embeds','gradient_ranked_token_ids'}
source=ast.parse((ROOT/'scripts/prompt_search/optimize_universal_suffix.py').read_text())
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)] +
                  [n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in NAMES],type_ignores=[])
namespace={'torch':torch,'F':F}
exec(compile(ast.fix_missing_locations(module),'<actual-search-functions>','exec'),namespace)

class ToyCausalModel:
    def __init__(self, weight):self.weight=weight
    def __call__(self,inputs_embeds,attention_mask):
        hidden=torch.tanh(inputs_embeds.cumsum(dim=1))
        return SimpleNamespace(logits=hidden@self.weight)

class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        g=torch.Generator().manual_seed(14)
        draw=lambda *shape:torch.randn(shape,generator=g,dtype=torch.float64)*.4
        self.dw=draw(11,4);self.tw=draw(11,4)
        self.draft=ToyCausalModel(draw(4,11));self.target=ToyCausalModel(draw(4,11))
        self.samples=[]
        seq=namespace['sequence_log_probs_from_embeds']
        for _ in range(2):
            sample=SimpleNamespace(draft_context_embeds=draw(1,2,4),target_context_embeds=draw(1,2,4),
                draft_context_mask=torch.ones(1,2),target_context_mask=torch.ones(1,2),
                draft_rollout_input_embeds=draw(1,2,4),target_rollout_input_embeds=draw(1,2,4),
                draft_rollout_input_mask=torch.ones(1,2),target_rollout_input_mask=torch.ones(1,2),
                rollout_ids=[0,1,2])
            sample.target_benign_log_probs=seq(model=self.target,context_embeds=sample.target_context_embeds,
                context_mask=sample.target_context_mask,suffix_embeds=torch.empty(0,4,dtype=torch.float64),
                rollout_input_embeds=sample.target_rollout_input_embeds,rollout_input_mask=sample.target_rollout_input_mask,
                target_length=3,vocab_size=11).detach()
            self.samples.append(sample)
        self.kw=dict(samples=self.samples,draft_model=self.draft,target_model=self.target,vocab_size=11,
            collapse_weight=2.,collapse_all_reduction='linear',tv_weight=0.,revkl_weight=0.,target_preserve_weight=1.)
    def objective(self,ds,ts):
        return namespace['hard_objective_from_suffix_embeds'](draft_suffix_embeds=ds,target_suffix_embeds=ts,**self.kw)
    def test_autograd_matches_finite_difference_for_both_models(self):
        ds=self.dw[[1,2]].clone().requires_grad_();ts=self.tw[[1,2]].clone().requires_grad_()
        self.objective(ds,ts).backward();epsilon=1e-5
        for role,gradient in [('draft',ds.grad),('target',ts.grad)]:
            left,right=ds.detach().clone(),ts.detach().clone()
            vector=left if role=='draft' else right
            vector[0,1]+=epsilon;plus=float(self.objective(left,right))
            vector[0,1]-=2*epsilon;minus=float(self.objective(left,right))
            self.assertAlmostEqual((plus-minus)/(2*epsilon),float(gradient[0,1]),places=6)
    def test_token_ranking_uses_ascent_and_both_embedding_spaces(self):
        ids=[1,2];ds=self.dw[ids];ts=self.tw[ids];epsilon=1e-5
        derivatives=[]
        for token in range(11):
            delta_d=torch.zeros_like(ds);delta_t=torch.zeros_like(ts)
            delta_d[0]=self.dw[token]-ds[0];delta_t[0]=self.tw[token]-ts[0]
            derivatives.append(float((self.objective(ds+epsilon*delta_d,ts+epsilon*delta_t)-
                                      self.objective(ds-epsilon*delta_d,ts-epsilon*delta_t))/(2*epsilon)))
        ranked=namespace['gradient_ranked_token_ids'](suffix_ids=ids,position=0,
            draft_embed_weight=self.dw,target_embed_weight=self.tw,topk=3,**self.kw)
        expected=sorted([i for i in range(11) if i!=ids[0]],key=lambda i:derivatives[i],reverse=True)[:3]
        self.assertEqual(ranked,[ids[0]]+expected)
