"""LangGraph agent: router -> tool (knowledge/calculator/direct) -> answer.

Agent V1 uses a correct router. Agent V2 (buggy=True) swaps the order of the
routing checks, which causes some calculation inputs that also contain a
knowledge-like word (e.g. "Evaluate 12 * 8") to be misrouted to knowledge.
"""
from __future__ import annotations

import logging
import re
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Knowledge base: 7 small hard-coded documents, no vector DB.
# ---------------------------------------------------------------------------
KNOWLEDGE_DOCS = [
    {
        "id": "doc_llm",
        "keywords": ["llm", "large language model", "gpt"],
        "text": "A large language model (LLM) is a neural network trained on large "
        "amounts of text to predict and generate natural language.",
    },
    {
        "id": "doc_rag",
        "keywords": ["rag", "retrieval-augmented", "retrieval"],
        "text": "Retrieval-Augmented Generation (RAG) combines a language model with "
        "an external document retriever to ground answers in real data.",
    },
    {
        "id": "doc_embedding",
        "keywords": ["embedding", "embeddings", "vector"],
        "text": "An embedding is a numeric vector representation of text that captures "
        "semantic meaning so similar texts have similar vectors.",
    },
    {
        "id": "doc_agent",
        "keywords": ["agent", "agents", "autonomous", "tool use"],
        "text": "An AI agent is a system that plans, chooses tools, and takes actions "
        "to achieve a goal, going beyond a single question-answer exchange.",
    },
    {
        "id": "doc_hallucination",
        "keywords": ["hallucinat"],
        "text": "Hallucination is when a language model generates confident but "
        "factually incorrect or unsupported information.",
    },
    {
        "id": "doc_evaluation",
        "keywords": ["evaluat", "benchmark", "metric"],
        "text": "Evaluation of AI systems measures correctness, reliability, and "
        "behavior using metrics and test cases rather than opinion alone.",
    },
    {
        "id": "doc_finetuning",
        "keywords": ["fine-tun", "finetun", "training"],
        "text": "Fine-tuning adapts a pretrained model to a specific task by further "
        "training it on a smaller, task-specific dataset.",
    },
]

KNOWLEDGE_KEYWORDS = sorted({kw for d in KNOWLEDGE_DOCS for kw in d["keywords"]}, key=len, reverse=True)

CALC_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?\s*[-+*/]\s*[-+]?\d+(?:\.\d+)?")


def knowledge_search(query: str) -> str:
    """Simple keyword-matching search over the hard-coded docs (no vector DB).

    Score = total character length of matched keywords, so a specific match
    (e.g. "hallucinat") outweighs a short, more generic one (e.g. "llm").
    """
    q = query.lower()
    best, best_score = KNOWLEDGE_DOCS[0], 0
    for doc in KNOWLEDGE_DOCS:
        score = sum(len(kw) for kw in doc["keywords"] if kw in q)
        if score > best_score:
            best, best_score = doc, score
    return best["text"]


def safe_calculate(expression: str) -> float:
    """Safe arithmetic for + - * / only. No eval()."""
    match = CALC_PATTERN.search(expression)
    if not match:
        raise ValueError(f"no arithmetic expression found in: {expression!r}")
    op_match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*([-+*/])\s*([-+]?\d+(?:\.\d+)?)", match.group(0))
    a, op, b = float(op_match.group(1)), op_match.group(2), float(op_match.group(3))
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise ZeroDivisionError("division by zero")
        return a / b
    raise ValueError(f"unsupported operator: {op}")  # pragma: no cover


class AgentState(TypedDict):
    input: str
    route: str
    tool_output: str
    answer: str
    path: List[str]


def make_router(buggy: bool = False):
    """Build the router node. buggy=True reproduces the Agent V2 routing bug."""

    def router_node(state: AgentState) -> AgentState:
        text = state["input"].lower()
        has_calc = bool(CALC_PATTERN.search(text))
        has_kw = any(kw in text for kw in KNOWLEDGE_KEYWORDS)

        if buggy:
            # BUG (V2): knowledge keywords are checked before the calculator
            # pattern, so a calc question containing a word like "evaluate"
            # gets misrouted to the knowledge tool.
            route = "knowledge" if has_kw else ("calculator" if has_calc else "direct")
        else:
            route = "calculator" if has_calc else ("knowledge" if has_kw else "direct")

        logger.debug("routed %r -> %s", state["input"], route)
        return {**state, "route": route, "path": state["path"] + ["router"]}

    return router_node


def knowledge_node(state: AgentState) -> AgentState:
    result = knowledge_search(state["input"])
    return {**state, "tool_output": result, "path": state["path"] + ["knowledge"]}


def calculator_node(state: AgentState) -> AgentState:
    try:
        result = safe_calculate(state["input"])
        output = str(int(result)) if result == int(result) else str(result)
    except (ValueError, ZeroDivisionError) as exc:
        output = f"error: {exc}"
    return {**state, "tool_output": output, "path": state["path"] + ["calculator"]}


_DIRECT_REPLIES = [
    (["hello", "hi "], "Hello! I'm a simple assistant here to help."),
    (["name"], "I don't have a personal name, I'm just an AI agent."),
    (["joke"], "I don't tell jokes, but I can search knowledge or do math."),
    (["thank"], "You're welcome!"),
]


def direct_node(state: AgentState) -> AgentState:
    text = state["input"].lower()
    reply = "I can help with knowledge questions or calculations."
    for triggers, response in _DIRECT_REPLIES:
        if any(t in text for t in triggers):
            reply = response
            break
    return {**state, "tool_output": reply, "path": state["path"] + ["direct"]}


def answer_node(state: AgentState) -> AgentState:
    return {**state, "answer": state["tool_output"], "path": state["path"] + ["answer"]}


def build_agent(buggy: bool = False):
    """Compile the LangGraph agent. buggy=True yields Agent V2."""
    graph = StateGraph(AgentState)
    graph.add_node("router", make_router(buggy))
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("direct", direct_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {"knowledge": "knowledge", "calculator": "calculator", "direct": "direct"},
    )
    graph.add_edge("knowledge", "answer")
    graph.add_edge("calculator", "answer")
    graph.add_edge("direct", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def run_agent(app, user_input: str) -> AgentState:
    """Invoke the compiled graph on a single input and return the final state."""
    initial: AgentState = {"input": user_input, "route": "", "tool_output": "", "answer": "", "path": []}
    return app.invoke(initial)
