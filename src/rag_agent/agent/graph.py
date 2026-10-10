
"""
graph.py
========
LangGraph agent graph definition and compilation.

Assembles the nodes from nodes.py into a directed state graph
and compiles it with a memory checkpointer for conversation persistence.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag_agent.agent.nodes import (
    generation_node,
    query_rewrite_node,
    retrieval_node,
    should_retry_retrieval,
)
from rag_agent.agent.state import AgentState


class AgentGraphBuilder:
    """
    Constructs and compiles the LangGraph RAG agent.

    Pipeline:
        START
          |
          v
    query_rewrite
          |
          v
       retrieval
          |
          v
    Conditional decision
       /       \
    generate    end
       |
       v
    generation
       |
       v
      END

    MemorySaver maintains conversation state for each thread.
    """

    def __init__(self) -> None:
        self._checkpointer = MemorySaver()

    def build(self):
        """
        Assemble and compile the LangGraph agent.

        Returns
        -------
        CompiledStateGraph
            Compiled graph with conversation memory.
        """

        # Step 1: Create the state graph
        graph = StateGraph(AgentState)

        # Step 2: Register the agent nodes
        graph.add_node(
            "query_rewrite",
            query_rewrite_node,
        )

        graph.add_node(
            "retrieval",
            retrieval_node,
        )

        graph.add_node(
            "generation",
            generation_node,
        )

        # Step 3: Define the execution flow
        graph.add_edge(
            START,
            "query_rewrite",
        )

        graph.add_edge(
            "query_rewrite",
            "retrieval",
        )

        # Step 4: Route based on retrieval results
        graph.add_conditional_edges(
            "retrieval",
            should_retry_retrieval,
            {
                "generate": "generation",
                "end": END,
            },
        )

        # Step 5: End after answer generation
        graph.add_edge(
            "generation",
            END,
        )

        # Step 6: Compile with conversation memory
        return graph.compile(
            checkpointer=self._checkpointer
        )


@lru_cache(maxsize=1)
def get_compiled_graph():
    """
    Return a cached compiled LangGraph agent.
    """
    return AgentGraphBuilder().build()
