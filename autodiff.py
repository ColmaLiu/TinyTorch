from typing import (
    Dict,
    List,
    Tuple,
)

from basic import Tensor, TensorOp

def find_topo_sort(node_list: List[Tensor]) -> List[Tensor]:
    n = len(node_list)
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node: Tensor, visited, topo_order):
    if node in visited:
        return
    for input in node.inputs:
        topo_sort_dfs(input, visited, topo_order)
    topo_order.append(node)
    visited.add(node)

def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_tensor] = [out_grad]

    reverse_topo_order = list(
        reversed(
            find_topo_sort(
                [output_tensor]
            )
        )
    )

    def sum_node_list(node_list):
        from operator import add
        from functools import reduce

        return reduce(add, node_list)
    
    for node in reverse_topo_order:
        adjoint = sum_node_list(
            node_to_output_grads_list[node]
        )
        node.grad = adjoint
        if node.op is None:
            continue
        partial_adjoints = node.op.gradient_as_tuple(adjoint, node)
        for input, partial_adjoint in zip(node.inputs, partial_adjoints):
            if input not in node_to_output_grads_list:
                node_to_output_grads_list[input] = []
            node_to_output_grads_list[input].append(partial_adjoint)