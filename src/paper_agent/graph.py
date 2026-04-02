from langgraph.graph import StateGraph, END

from paper_agent.state import PaperState
from paper_agent.nodes import parser_node, info_node, researcher_node, writer_node


def _check_parser_result(state: PaperState):
    if state.get("file_uri"):
        return "continue"
    return "abort"


workflow = StateGraph(PaperState)

workflow.add_node("Parser", parser_node)
workflow.add_node("Info", info_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Writer", writer_node)

workflow.add_conditional_edges("Parser", _check_parser_result, {"continue": "Info", "abort": END})
workflow.add_edge("Info", "Researcher")
workflow.add_edge("Researcher", "Writer")
workflow.add_edge("Writer", END)

workflow.set_entry_point("Parser")

app = workflow.compile()
