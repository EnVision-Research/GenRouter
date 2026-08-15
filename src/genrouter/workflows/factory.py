from __future__ import annotations

from typing import Any

from genrouter.knowledge.skill_bank import SkillBank
from genrouter.primitives import Analyze, Decompose, ExperienceSummarizer, Generate, Reason, Refine, Search, Sketch, SkillQuery, Verify
from genrouter.primitives.rewrite import Rewrite
from genrouter.workflows.code_sketch_gen import CodeSketchGenWorkflow
from genrouter.workflows.direct_gen import DirectGenWorkflow
from genrouter.workflows.hybrid_gen import HybridGenWorkflow
from genrouter.workflows.reason_gen import ReasonGenWorkflow
from genrouter.workflows.ref_gen import RefGenWorkflow
from genrouter.workflows.rewrite_gen import RewriteGenWorkflow
from genrouter.workflows.search_gen import SearchGenWorkflow
from genrouter.workflows.skill_gen import SkillGenWorkflow
from genrouter.workflows.verify_refine import VerifyRefineWorkflow


def build_workflow(
    name: str,
    skill_bank: SkillBank,
    *,
    llm: Any,
    mllm: Any,
    search_backend: Any,
):
    generate = Generate()
    search = Search(search_backend)
    if name == "DirectGen":
        return DirectGenWorkflow(generate=generate)
    if name == "RewriteGen":
        return RewriteGenWorkflow(rewrite=Rewrite(llm), generate=generate, analyze=Analyze(llm))
    if name == "SearchGen":
        return SearchGenWorkflow(search=search, rewrite=Rewrite(llm), generate=generate, analyze=Analyze(llm))
    if name == "RefGen":
        return RefGenWorkflow(search=search, rewrite=Rewrite(llm), generate=generate, analyze=Analyze(llm))
    if name == "ReasonGen":
        return ReasonGenWorkflow(reason=Reason(llm), rewrite=Rewrite(llm), generate=generate, analyze=Analyze(llm))
    if name == "SkillGen":
        return SkillGenWorkflow(
            skill_query=SkillQuery(skill_bank, llm=llm),
            rewrite=Rewrite(llm),
            generate=generate,
            analyze=Analyze(llm),
        )
    if name == "CodeSketchGen":
        return CodeSketchGenWorkflow(
            analyze=Analyze(llm),
            sketch=Sketch(llm),
            generate=generate,
        )
    if name == "VerifyRefine":
        return VerifyRefineWorkflow(
            analyze=Analyze(llm),
            decompose=Decompose(llm),
            generate=generate,
            verify=Verify(mllm),
            refine=Refine(llm),
            rewrite=Rewrite(llm),
            experience=ExperienceSummarizer(mllm),
        )
    if name == "HybridGen":
        return HybridGenWorkflow(
            search=search,
            reason=Reason(llm),
            skill_query=SkillQuery(skill_bank, llm=llm),
            rewrite=Rewrite(llm),
            decompose=Decompose(llm),
            generate=generate,
            verify=Verify(mllm),
            refine=Refine(llm),
            experience=ExperienceSummarizer(mllm),
            analyze=Analyze(llm),
            sketch=Sketch(llm),
        )
    raise KeyError(f"Manual execution is not implemented for workflow: {name}")
