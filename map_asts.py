# Logic for creating a mapping between two ManipulableAst objects

import time
from collections import defaultdict

import networkx as nx

import manip_ast
import ast
import copy
from apted import APTED, Config
from difflib import SequenceMatcher


class CompareConfig(Config):
    def __init__(self, rename_weight=1.9, use_assign_depth=False):
        self.rename_weight = rename_weight
        self.use_assign_depth = use_assign_depth

    def delete(self, node):
        """Calculates the cost of deleting a node"""
        return 1.0

    def insert(self, node):
        """Calculates the cost of inserting a node"""
        '''if (node.assign_depth is not None):
            return node.num_children + 1'''

        return 1.0

    def rename(self, node1, node2):
        """Calculates the cost of renaming the label of the source node
        to the label of the destination node"""
        base_cost = int(node1.name != node2.name)

        if self.use_assign_depth and \
                (node1.assign_depth is not None) and (node2.assign_depth is not None) and\
                node1.assign_depth != node2.assign_depth:
            base_cost = 1.0

        if (node1.isList != node2.isList) or (node1.isLiteral != node2.isLiteral):
            # comparing two different type of nodes - very expensive to rename one into the other!
            # (this is effectively infinitely expensive, since delete + add is cheaper)
            return 10.0*base_cost
        if node1.nodeType != node2.nodeType or (node1.isList and node2.isList and node1.name != node2.name):
            # "renaming" in this case is more expensive than just deleting one,
            # but (maybe) less expensive than delete + insert
            # (also consider nodeLists of different kinds of nodes to be different types)
            # TODO: consider Binop a special case where the op has to match?
            #  or some other special weight for changing op?
            return self.rename_weight * base_cost
        # TODO: maybe changing literals should be more expensive, to map constants onto each other better?..
        #  But off-by-one errors will be harder to detect then.

        return float(base_cost)


def _generate_subtrees(tree: manip_ast.ManipulableAst, exclude_set):
    # Generate a ManipulableAst which contains a subset of the nodes in tree:
    # include all nodes except those in exclude_set.
    # if a node is in exclude_set, skip it and move its children to its parent.
    child_subtrees = []
    for c in tree.children:
        child_subtrees.extend(_generate_subtrees(c, exclude_set))

    if tree in exclude_set:
        # we should not keep this node: just return generated children, they will get propagated up
        return child_subtrees
    else:
        # we should keep this node
        # make shallow copy of node, add subtrees generated from children
        # TODO: is this deepcopy necessary? (shallow ManipulableAst prunes the ast's children anyway)
        new_node = manip_ast.ManipulableAst(copy.deepcopy(tree.ast), tree.index, shallow=True, name=tree.name)
        # retain name of old node.
        for new_c in child_subtrees:
            new_node.add_child_anywhere(copy.deepcopy(new_c))
        return [new_node]  # return list for consistency


def _generate_unmapped_subtrees(source_tree: manip_ast.ManipulableAst,
                                dest_tree: manip_ast.ManipulableAst,
                                node_mapping: list):
    # Generate subtrees of both source_tree and dest_tree
    # which consist only of nodes that aren't mapped to each other in node_mapping.
    source_mapped = [s_n for (s_n, c_n) in node_mapping if (c_n is not None and s_n is not None)]
    dest_mapped = [c_n for (s_n, c_n) in node_mapping if (c_n is not None and s_n is not None)]

    # Make new fake root nodes,
    # set a special FakeRoot type to make sure they are mapped to each other (not some arbitrary NodeList)
    source_root = manip_ast.ManipulableAst([], source_tree.index, shallow=True, name='FakeRoot')
    source_root.nodeType = 'FakeRoot'
    dest_root = manip_ast.ManipulableAst([], dest_tree.index, shallow=True, name='FakeRoot')
    dest_root.nodeType = 'FakeRoot'

    source_children = _generate_subtrees(source_tree, set(source_mapped))

    for c in source_children:
        source_root.add_child_anywhere(copy.deepcopy(c))

    dest_children = _generate_subtrees(dest_tree, set(dest_mapped))

    for d_c in dest_children:
        dest_root.add_child_anywhere(copy.deepcopy(d_c))

    return source_root, dest_root


def generate_mapping(source_tree: manip_ast.ManipulableAst, dest_tree: manip_ast.ManipulableAst):
    index_mapping = set()
    index_mapping.add((source_tree.index, dest_tree.index))  # add original roots to mapping right away
    should_continue = True
    rename_weight = 1.0  # the first time around, prefer renaming to delete+add (if < 2.0) TODO: make parameter?
    use_assign_depth = True
    # i = 0

    while should_continue:  # the roots of two trees will always be 'mappable'
        node_mapping = APTED(source_tree, dest_tree,
                             config=CompareConfig(rename_weight=rename_weight, use_assign_depth=use_assign_depth)
                             ).compute_edit_mapping()
        rename_weight = 2.0  # after the first time around, don't rename unnecessarily
        use_assign_depth = False  # only use it for the first (coarse-grained) pass
        should_continue = False  # assume we're done
        for s_n, d_n in node_mapping:
            if (s_n is not None) and (d_n is not None) and (s_n != source_tree) and (d_n != dest_tree):
                # if nodes are mapped to each other, and are not roots of this iteration of trees
                if s_n.name != d_n.name and len(s_n.children) == 0 and len(d_n.children) == 0 \
                        and ((s_n.parent.index, d_n.parent.index) not in index_mapping
                             or (s_n.key_in_parent != d_n.key_in_parent)):
                    # skip mappings between two leaf nodes which would have to be renamed AND moved
                    # (APTED is a bit too greedy since it doesn't account for the possibility of moving nodes)
                    # TODO: may have some false positives if parent mapping hasn't been processed yet
                    continue
                should_continue = True
                index_mapping.add( (s_n.index, d_n.index) )
        # draw_comparison(source_tree, dest_tree, index_mapping, '', '', f'leftovers{i}.dot')
        # i += 1

        source_tree, dest_tree = _generate_unmapped_subtrees(source_tree, dest_tree, node_mapping)

    return index_mapping


def draw_comparison(source_tree, dest_tree, index_mapping, filename='out/test.dot'):
    source_code = str(source_tree)
    dest_code = str(dest_tree)

    source_justified = source_code.replace('\r\n', '\l')
    dest_justified = dest_code.replace('\r\n', '\l')

    source_justified = source_justified.replace('\n', '\l') + '\l'
    dest_justified = dest_justified.replace('\n', '\l') + '\l'

    map_edges = ''
    for s_i, c_i in index_mapping:
        if (s_i is not None) and (c_i is not None):
            map_edges += f'source{manip_ast.ManipulableAst.gen_short_index(s_i)} -> ' \
                         f'dest{manip_ast.ManipulableAst.gen_short_index(c_i)}\n'

    with open(f'{filename}', 'w') as f:
        f.write(f'''
            digraph G
            {{
                splines="line"
                source{source_tree.short_index} -> dest{dest_tree.short_index}[style=invis] //force vertical offset
                subgraph cluster_source {{
                    {source_tree.generate_dot_notation('source')}
                    label="{source_justified}"
                }}

                subgraph cluster_dest {{
                    {dest_tree.generate_dot_notation('dest')}
                    label="{dest_justified}"
                }}

                subgraph mapping {{
                    edge [constraint=false, arrowhead=none, style=dotted]
                    {map_edges}
                }}
            }}
        ''')


def get_trees_and_mapping(source_text, dest_text):
    source_tree = manip_ast.ManipulableAst(ast.parse(source_text))
    # make sure dest indices are distinct from source indices:
    dest_tree = manip_ast.ManipulableAst(ast.parse(dest_text))

    index_mapping = generate_mapping(source_tree, dest_tree)

    draw_comparison(source_tree, dest_tree, index_mapping, f'../out/test_APTED_gen.dot')
    return source_tree, dest_tree, index_mapping
