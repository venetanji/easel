"""Minimal builder for ComfyUI API-format prompt graphs."""


class NodeRef:
    """Handle to an emitted node; `ref[i]` is the ComfyUI link [node_id, output_slot]."""

    __slots__ = ("id",)

    def __init__(self, node_id: str):
        self.id = node_id

    def __getitem__(self, slot: int):
        return [self.id, slot]


class WorkflowGraph:
    """Minimal builder for ComfyUI API-format prompt graphs."""

    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._counter = 0

    def node(self, class_type: str, **inputs) -> NodeRef:
        self._counter += 1
        node_id = str(self._counter)
        self._nodes[node_id] = {"class_type": class_type, "inputs": inputs}
        return NodeRef(node_id)

    def to_dict(self) -> dict:
        return {
            nid: {"class_type": n["class_type"], "inputs": dict(n["inputs"])}
            for nid, n in self._nodes.items()
        }
