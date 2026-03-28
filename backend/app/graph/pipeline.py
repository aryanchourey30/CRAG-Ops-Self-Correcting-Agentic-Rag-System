"""LangGraph CRAG pipeline."""

from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.evaluator.service import evaluate_retrieval
from app.generation.service import generate_answer
from app.graph.state import GraphState
from app.guardrail.service import validate_answer
from app.observability.tracer import append_step_log, new_trace_id, persist_trace
from app.retrieval.service import retrieve_chunks

pipeline_logger = logging.getLogger("crag_ops.pipeline")


def retrieve_node(state: GraphState) -> GraphState:
    chunks = retrieve_chunks(state["query"], document_id=state.get("document_id"))
    append_step_log(
        state,
        step="retrieve_node",
        input_data={"query": state["query"], "document_id": state.get("document_id")},
        output_data={"count": len(chunks)},
        decision="RETRIEVED",
    )
    pipeline_logger.info("retrieve_node | trace_id=%s | count=%s", state["trace_id"], len(chunks))
    return {"retrieved_chunks": chunks}


def evaluator_node(state: GraphState) -> GraphState:
    score, decision = evaluate_retrieval(state["query"], state.get("retrieved_chunks", []))

    if decision == "REJECT" and not state.get("web_search_attempted", False):
        decision = "EXPAND"

    append_step_log(
        state,
        step="evaluator_node",
        input_data={
            "chunk_count": len(state.get("retrieved_chunks", [])),
            "web_search_attempted": state.get("web_search_attempted", False),
        },
        output_data={"relevance_score": score},
        decision=decision,
    )
    pipeline_logger.info(
        "evaluator_node | trace_id=%s | score=%.4f | decision=%s",
        state["trace_id"],
        score,
        decision,
    )
    return {"relevance_score": score, "decision": decision}


def web_search_node(state: GraphState) -> GraphState:
    expanded_query = f"{state['query']} supporting evidence"
    chunks = retrieve_chunks(expanded_query, document_id=None)
    combined = state.get("retrieved_chunks", []) + chunks
    append_step_log(
        state,
        step="web_search_node",
        input_data={"query": expanded_query},
        output_data={"additional_chunks": len(chunks)},
        decision="EXPANDED",
    )
    pipeline_logger.info("web_search_node | trace_id=%s | extra_count=%s", state["trace_id"], len(chunks))
    return {"retrieved_chunks": combined, "web_search_attempted": True}


def generator_node(state: GraphState) -> GraphState:
    answer = generate_answer(state["query"], state.get("retrieved_chunks", []))
    citations: list[dict] = []
    for chunk in state.get("retrieved_chunks", []):
        metadata = chunk.get("metadata", {})
        citations.append(
            {
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page"),
                "snippet": chunk.get("text", "")[:220],
                "url": metadata.get("url"),
            }
        )

    append_step_log(
        state,
        step="generator_node",
        input_data={"chunk_count": len(state.get("retrieved_chunks", []))},
        output_data={"answer_preview": answer[:200]},
        decision="GENERATED",
    )
    pipeline_logger.info("generator_node | trace_id=%s | citations=%s", state["trace_id"], len(citations))
    return {"generated_answer": answer, "citations": citations}


def guardrail_node(state: GraphState) -> GraphState:
    answer, regenerated = validate_answer(
        state["query"],
        state.get("generated_answer", ""),
        state.get("retrieved_chunks", []),
    )
    decision = "REGENERATED" if regenerated else "VALIDATED"
    append_step_log(
        state,
        step="guardrail_node",
        input_data={"answer_preview": state.get("generated_answer", "")[:200]},
        output_data={"answer_preview": answer[:200]},
        decision=decision,
    )
    pipeline_logger.info("guardrail_node | trace_id=%s | decision=%s", state["trace_id"], decision)
    return {"generated_answer": answer}


def route_after_evaluator(state: GraphState) -> str:
    decision = state.get("decision", "REJECT")
    if decision == "APPROVE":
        return "generator"
    if decision == "EXPAND":
        return "web_search"
    return "clarify"


def post_web_search_node(state: GraphState) -> GraphState:
    score, decision = evaluate_retrieval(state["query"], state.get("retrieved_chunks", []))
    final_decision = "APPROVE" if decision in {"APPROVE", "EXPAND"} else "REJECT"

    append_step_log(
        state,
        step="post_web_search_node",
        input_data={"chunk_count": len(state.get("retrieved_chunks", []))},
        output_data={"relevance_score": score},
        decision=final_decision,
    )
    pipeline_logger.info(
        "post_web_search_node | trace_id=%s | score=%.4f | decision=%s",
        state["trace_id"],
        score,
        final_decision,
    )
    return {"relevance_score": score, "decision": final_decision}


def route_after_post_web_search(state: GraphState) -> str:
    if state.get("decision") == "APPROVE":
        return "generator"
    return "clarify"


def clarification_node(state: GraphState) -> GraphState:
    answer = (
        "I do not have enough reliable evidence to answer confidently yet. "
        "Please refine the question or provide a more relevant PDF."
    )
    append_step_log(
        state,
        step="clarification_node",
        input_data={"relevance_score": state.get("relevance_score", 0.0)},
        output_data={"answer_preview": answer},
        decision="CLARIFY",
    )
    pipeline_logger.info("clarification_node | trace_id=%s", state["trace_id"])
    return {"generated_answer": answer, "citations": []}


@lru_cache(maxsize=1)
def build_graph():
    """Compile the CRAG state graph once."""

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluator_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("post_web_search", post_web_search_node)
    graph.add_node("generator", generator_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("clarify", clarification_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluator,
        {
            "generator": "generator",
            "web_search": "web_search",
            "clarify": "clarify",
        },
    )
    graph.add_edge("web_search", "post_web_search")
    graph.add_conditional_edges(
        "post_web_search",
        route_after_post_web_search,
        {
            "generator": "generator",
            "clarify": "clarify",
        },
    )
    graph.add_edge("generator", "guardrail")
    graph.add_edge("guardrail", END)
    graph.add_edge("clarify", END)
    return graph.compile()


def run_pipeline(query: str, *, mode: str, document_id: str | None = None) -> GraphState:
    """Invoke the CRAG graph and persist structured traces."""

    settings = get_settings()
    trace_id = new_trace_id()
    state: GraphState = {
        "query": query,
        "mode": mode,
        "document_id": document_id,
        "retrieved_chunks": [],
        "relevance_score": 0.0,
        "decision": "",
        "generated_answer": "",
        "citations": [],
        "trace_id": trace_id,
        "logs": [],
        "web_search_attempted": False,
    }
    result = build_graph().invoke(state)
    persist_trace(settings.log_path, trace_id, result)
    return result
