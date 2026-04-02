from langgraph.graph import StateGraph, END

from paper_agent.state import PaperState
from paper_agent.nodes import parser_node, researcher_node

# from paper_agent.nodes import planner_node, writer_node, reviewer_node, review_decision

workflow = StateGraph(PaperState)

workflow.add_node("Parser", parser_node)
workflow.add_node("Researcher", researcher_node)
# workflow.add_node("Planner", planner_node)
# workflow.add_node("Writer", writer_node)
# workflow.add_node("Reviewer", reviewer_node)

workflow.add_edge("Parser", "Researcher")
# workflow.add_edge("Researcher", "Planner")
# workflow.add_edge("Planner", "Writer")
# workflow.add_edge("Writer", "Reviewer")

# workflow.add_conditional_edges(
#     "Reviewer",
#     review_decision,
#     {"end": END, "rewrite": "Writer"}
# )

workflow.set_entry_point("Parser")

app = workflow.compile()
