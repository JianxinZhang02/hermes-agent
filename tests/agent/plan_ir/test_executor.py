from agent.plan_ir.executor import NodeResult, execute_plan
from agent.plan_ir.schema import PlanIR, PlanNode


def _node(node_id, dependencies=(), on_error="abort"):
    return PlanNode(
        node_id=node_id,
        tool_name="read_file",
        arguments={"path": f"{node_id}.md"},
        depends_on=dependencies,
        on_error=on_error,
        timeout_seconds=5,
    )


def test_executor_batches_fanout_then_dependency():
    plan = PlanIR(1, "answer", (_node("a"), _node("b"), _node("c", ("a", "b"))))
    batches = []

    def handler(batch):
        batches.append([node.node_id for node, _ in batch])
        return [
            NodeResult(node.node_id, True, {"path": f"{node.node_id}.md"})
            for node, _ in batch
        ]

    result = execute_plan(plan, handler, max_concurrency=4)
    assert result.ok is True
    assert batches == [["a", "b"], ["c"]]
    assert result.batch_count == 2
    assert result.longest_depth == 2


def test_continue_keeps_running_but_abort_falls_back():
    continued = PlanIR(
        1, "answer", (_node("a", on_error="continue"), _node("b", ("a",)))
    )

    def handler(batch):
        return [
            NodeResult(node.node_id, node.node_id != "a", None, "tool_failed")
            for node, _ in batch
        ]

    result = execute_plan(continued, handler, max_concurrency=1)
    assert result.ok is True
    aborted = PlanIR(1, "answer", (_node("a"), _node("b", ("a",))))
    result = execute_plan(aborted, handler, max_concurrency=1)
    assert result.ok is False
    assert result.error_code == "tool_failed"
