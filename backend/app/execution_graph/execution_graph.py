"""ASTER execution graph module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExecutionNode:
    """A single node in the execution graph."""

    node_name: str
    purpose: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the node."""

        return {
            "node_name": self.node_name,
            "purpose": self.purpose,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
        }


@dataclass(slots=True)
class ExecutionGraph:
    """Represents the Planner's output as an ordered graph of nodes with explicit dependencies."""

    workflow_name: str
    intent: str
    intent_classification: str
    nodes: list[ExecutionNode] = field(default_factory=list)
    entrypoint: str | None = None
    exitpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the graph."""

        return {
            "workflow_name": self.workflow_name,
            "intent": self.intent,
            "intent_classification": self.intent_classification,
            "nodes": [node.to_dict() for node in self.nodes],
            "entrypoint": self.entrypoint,
            "exitpoint": self.exitpoint,
        }

    def get_execution_order(self) -> list[str]:
        """Return the node names in dependency order (topological sort)."""

        if not self.nodes:
            return []

        # Build dependency map
        node_names = {node.node_name for node in self.nodes}
        dependency_map: dict[str, set[str]] = {name: set() for name in node_names}

        for node in self.nodes:
            for dep in node.dependencies:
                if dep in node_names:
                    dependency_map[node.node_name].add(dep)

        # Simple topological sort (Kahn's algorithm)
        in_degree: dict[str, int] = {name: 0 for name in node_names}
        for node in node_names:
            for dep in dependency_map[node]:
                in_degree[node] += 1

        queue = [name for name in node_names if in_degree[name] == 0]
        result: list[str] = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for node in node_names:
                if current in dependency_map[node]:
                    in_degree[node] -= 1
                    if in_degree[node] == 0:
                        queue.append(node)

        if len(result) != len(node_names):
            # Cycle detected - fall back to original order
            return [node.node_name for node in self.nodes]

        return result


def build_execution_graph(planner_output: dict[str, Any]) -> ExecutionGraph:
    """Build an ExecutionGraph from the Planner's output."""

    steps = planner_output.get("steps", [])
    nodes: list[ExecutionNode] = []

    # Build nodes and infer dependencies
    for i, step in enumerate(steps):
        node_name = step.get("node")
        purpose = step.get("purpose", "")
        inputs = step.get("inputs", [])
        outputs = step.get("outputs", [])

        # Dependencies: previous nodes whose outputs are used as inputs
        # Use substring matching to handle path variations
        dependencies: list[str] = []
        for j, prev_step in enumerate(steps[:i]):
            prev_outputs = prev_step.get("outputs", [])
            prev_node = prev_step.get("node")
            if not prev_node:
                continue
            
            for prev_output in prev_outputs:
                # Extract the base filename from the output
                output_filename = prev_output.split("/")[-1] if "/" in prev_output else prev_output
                
                # Check if any input references this output
                for inp in inputs:
                    input_filename = inp.split("/")[-1] if "/" in inp else inp
                    if output_filename in input_filename or input_filename in output_filename:
                        if prev_node not in dependencies:
                            dependencies.append(prev_node)
                            break

        nodes.append(
            ExecutionNode(
                node_name=node_name,
                purpose=purpose,
                inputs=inputs,
                outputs=outputs,
                dependencies=dependencies,
            )
        )

    return ExecutionGraph(
        workflow_name=planner_output.get("workflow_name", "unknown"),
        intent=planner_output.get("intent", "unknown"),
        intent_classification=planner_output.get("intent_classification", "full_workflow"),
        nodes=nodes,
        entrypoint=planner_output.get("entrypoint"),
        exitpoint=planner_output.get("exitpoint"),
    )
