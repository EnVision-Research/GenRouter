"""Workflow template namespace."""

from genrouter.workflows.code_sketch_gen import CodeSketchGenWorkflow
from genrouter.workflows.direct_gen import DirectGenWorkflow
from genrouter.workflows.hybrid_gen import HybridGenWorkflow
from genrouter.workflows.reason_gen import ReasonGenWorkflow
from genrouter.workflows.ref_gen import RefGenWorkflow
from genrouter.workflows.rewrite_gen import RewriteGenWorkflow
from genrouter.workflows.search_gen import SearchGenWorkflow
from genrouter.workflows.skill_gen import SkillGenWorkflow
from genrouter.workflows.verify_refine import VerifyRefineWorkflow

__all__ = [
    "CodeSketchGenWorkflow",
    "DirectGenWorkflow",
    "HybridGenWorkflow",
    "ReasonGenWorkflow",
    "RefGenWorkflow",
    "RewriteGenWorkflow",
    "SearchGenWorkflow",
    "SkillGenWorkflow",
    "VerifyRefineWorkflow",
]
