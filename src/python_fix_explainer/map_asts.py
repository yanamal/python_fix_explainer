# Logic for creating a "good" mapping between two MutableAst objects
# The mapping mostly tries to optimize for a shorter resulting edit script,
# though there are some heuristics in place to map more "related" parts of the code together.


import ast
import copy
from apted import APTED, Config


from .muast import MutableAst


class CompareConfig(Config):
    def __init__(self, rename_weight=1.9, use_assign_depth=False, careful_statement_map=False):
        self.rename_weight = rename_weight  # this is the weight for renaming between different types of nodes.
        # print('internal rename weight: ', self.rename_weight)
        # renaming nodes of same type but different content (e.g. variable with different name) is always 1.

        self.use_assign_depth = use_assign_depth
        self.careful_statement_map = careful_statement_map  # should we try to avoid mapping statements to expresions?

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
        base_cost = int(node1.name != node2.name)  # 1=different, 0=there is no rename

        # TODO: detect and forbid cases when we are mapping a statement (e.g. if statement, function def, etc.)
        #  onto a place that expects an expression (e.g. as one of the parameters of a function call).
        #  I think node types that are statements can only go into a nodeList that is at key 'body' in some other node.
        #  Note: need access to the original tree to know where the nodes were originally.
        #  or maybe add node metadata that tries to guess if it is a statement?..

        if self.use_assign_depth and \
                (node1.assign_depth is not None) and (node2.assign_depth is not None) and\
                node1.assign_depth != node2.assign_depth:
            base_cost = 1.0
            # These should be considered different even if their names match (e.g. both are the variable x)
            # because they are at different depths with an assigned expression, and at this stage we care about that.

        # TODO: if base_cost is 0 at this stage, return 0 early?

        # TODO: does nothing when using naive_class_function_mapping (key_in_parent broken)
        if self.careful_statement_map and node1.parent and node2.parent:
            # print(node1.isStatement, node2.isStatement)
            # if node1.isStatement or node2.isStatement:
            #     print(node1, node2)
            #     exit()
            if node1.isStatement != node2.isStatement:
                return 10.0*base_cost

        if (node1.isList != node2.isList) or (node1.isLiteral != node2.isLiteral):
            # comparing two different type of nodes - very expensive to rename one into the other!
            # (this is effectively infinitely expensive, since delete + add is cheaper)
            return 10.0*base_cost
        if node1.nodeType != node2.nodeType or (node1.isList and node2.isList and node1.name != node2.name):
            # print('rename with a cost', node1.name, node2.name, base_cost, self.rename_weight)
            # These are different kinds of things.
            # "renaming" in this case is more expensive than just deleting one,
            # but (maybe) less expensive than delete + insert
            # (also consider nodeLists of different kinds of nodes to be different types)
            # TODO: consider Binop a special case where the op has to match?
            #  or some other special weight for changing op?
            # TODO: possibly also account for whether the node is "playing the same role"
            #  even though it's a different type of node/expression.
            #  e.g. a parameter in the same place in the parameter list.
            #  This would be rather complicated and fiddly
            #  (e.g. looking at parent of parent and position in list in the parameter case, but probably not elsewhere)
            #  but it may be worth it to explicitly define what we mean by "similar" node in the context of python
            return self.rename_weight * base_cost
        # TODO: maybe changing literals should be more expensive, to map constants onto each other better?..
        #  But off-by-one errors will be harder to detect then.

        return float(base_cost)


def _generate_subtrees(tree: MutableAst, exclude_set):
    # Generate a MutableAst which contains a subset of the nodes in tree:
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

        # deepcopy of the underlying Python AST ensures that we don't mess up the original underlying tree
        # (e.g. when cutting off children to make the copy shallow)
        # TODO: ensure we pass along isStatement everywhere we're making new nodes
        new_node = MutableAst(copy.deepcopy(tree.ast), tree.index, shallow=True, name=tree.name, isStatement=tree.isStatement)
        # retain name of old node.
        for new_c in child_subtrees:
            # TODO: is this deepcopy necessary? or will all the underlying children be copied by now anyway?
            new_node.add_child_anywhere(copy.deepcopy(new_c))
        return [new_node]  # return list for consistency


def _generate_unmapped_subtrees(source_tree: MutableAst,
                                dest_tree: MutableAst,
                                node_mapping: list):
    # Generate subtrees of both source_tree and dest_tree
    # which consist only of nodes that aren't mapped to each other in node_mapping.
    source_mapped = [s_n for (s_n, c_n) in node_mapping if (c_n is not None and s_n is not None)]
    dest_mapped = [c_n for (s_n, c_n) in node_mapping if (c_n is not None and s_n is not None)]

    # Make new fake root nodes,
    # set a special FakeRoot type to make sure they are mapped to each other (not some arbitrary NodeList)
    source_root = MutableAst([], source_tree.index, shallow=True, name='FakeRoot')
    source_root.nodeType = 'FakeRoot'
    dest_root = MutableAst([], dest_tree.index, shallow=True, name='FakeRoot')
    dest_root.nodeType = 'FakeRoot'

    source_children = _generate_subtrees(source_tree, set(source_mapped))

    for c in source_children:
        source_root.add_child_anywhere(copy.deepcopy(c))

    dest_children = _generate_subtrees(dest_tree, set(dest_mapped))

    for d_c in dest_children:
        dest_root.add_child_anywhere(copy.deepcopy(d_c))

    return source_root, dest_root


def naive_recursive_mapping(source_tree: MutableAst, dest_tree: MutableAst, partial_mapping=None):
    if partial_mapping is None:
        partial_mapping = []
    if source_tree.name == dest_tree.name:
        partial_mapping.append((source_tree, dest_tree))
        # TODO: something different for list nodes?.. (don't use indices as keys)
        for child_key in source_tree.children_dict:
            if child_key in dest_tree.children_dict:
                naive_recursive_mapping(
                    source_tree.children_dict[child_key],
                    dest_tree.children_dict[child_key],
                    partial_mapping)
    return partial_mapping


def naive_class_function_mapping(source_tree: MutableAst, dest_tree: MutableAst,
                                 rename_weight, use_assign_depth, careful_statement_map,
                                 partial_mapping=None, source_index_to_node=None, dest_index_to_node = None):
    types_to_map = {'ClassDef', 'FunctionDef', 'Module'}

    # initialize recursion parameters if they are None
    if partial_mapping is None:
        partial_mapping = []
    if source_index_to_node is None:
        source_index_to_node = source_tree.gen_index_to_node()
    if dest_index_to_node is None:
        dest_index_to_node = dest_tree.gen_index_to_node()

    if source_tree.name == dest_tree.name and (source_tree.nodeType in types_to_map or source_tree.isList):
        partial_mapping.append((source_tree, dest_tree))

        source_children = source_tree.children_dict
        dest_children = dest_tree.children_dict

        if source_tree.isList:
            # These nodes are lists.
            # (we can assume they ae both lists, since their names matched and names include list type)
            # instead of comparing indices as keys, make child dictionaries keyed on the child's name
            # so we can match potential children with same names.
            # TODO: some LCS scheme would be more robust, but for functions/classes this should be sufficient
            #  since there should not be sibling nodes with repeating names (e.g. "FunctionDef name: get_fortune")
            source_children = {}
            for child in source_tree.children:
                source_children[child.name] = child
            dest_children = {}
            for child in dest_tree.children:
                dest_children[child.name] = child
            pass

        for child_key in source_children:
            if child_key in dest_children:
                naive_class_function_mapping(
                    source_children[child_key],
                    dest_children[child_key],
                    rename_weight=rename_weight,
                    use_assign_depth=use_assign_depth,
                    careful_statement_map=careful_statement_map,
                    partial_mapping=partial_mapping,
                    source_index_to_node=source_index_to_node,
                    dest_index_to_node=dest_index_to_node,
                )

        source_unmapped, dest_unmapped = _generate_unmapped_subtrees(source_tree, dest_tree, partial_mapping)
        apted_mapping = APTED(source_unmapped, dest_unmapped,
                                 config=CompareConfig(rename_weight=rename_weight,
                                                      use_assign_depth=use_assign_depth,
                                                      careful_statement_map=careful_statement_map)
                                 ).compute_edit_mapping()
        # apted_mapping is based on copies generated by _generate_unmapped_subtrees.
        # we need to add original nodes with same indices to the partial mapping.
        for s_n, d_n in apted_mapping:
            if s_n is not None and d_n is not None:
                partial_mapping.append((source_index_to_node[s_n.index], dest_index_to_node[d_n.index]))

    return partial_mapping


def generate_mapping(source_tree: MutableAst, dest_tree: MutableAst, do_naive_pass=True):
    # generate mapping between the two trees, represented as a set of pairs of mapped nodes.
    index_mapping = set()
    index_mapping.add((source_tree.index, dest_tree.index))  # add original roots to mapping right away
    should_continue = True
    rename_weight = 1.9  # the first time around, prefer renaming to delete+add (if < 2.0) TODO: make parameter?
    use_assign_depth = True
    careful_statement_map = True
    i = 0

    while should_continue:  # the roots of two trees will always be 'mappable'
        # print('pass', i, 'at rename weight: ', rename_weight)
        if do_naive_pass and i == 0:
            node_mapping = naive_class_function_mapping(source_tree, dest_tree,
                                                        rename_weight=2.0,  # This rename_weight is only used in
                                                        # secondary processing after creating "leftover" trees
                                                        # after the naive mapping.
                                                        # These "leftover" trees should always
                                                        # err on the side of not renaming.
                                                        use_assign_depth=use_assign_depth,
                                                        careful_statement_map=careful_statement_map)
        else:
            node_mapping = APTED(source_tree, dest_tree,
                                 config=CompareConfig(rename_weight=rename_weight,
                                                      use_assign_depth=use_assign_depth,
                                                      careful_statement_map=careful_statement_map)
                                 ).compute_edit_mapping()
        rename_weight = 2.0  # after first APTED pass (in naive_class_function_mapping), don't rename unnecessarily
        # print('changing rename weight: ', rename_weight)
        use_assign_depth = False  # only use it for the first (coarse-grained) pass TODO: should this be in the else?..
        # careful_statement_map = False  # only the first time -
        # shouldn't map different types of things on subsequent passes anyway

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
        # draw_comparison(source_tree, dest_tree, index_mapping, f'leftovers{i}.dot')
        i += 1

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
            map_edges += f'source{MutableAst.gen_short_index(s_i)} -> ' \
                         f'dest{MutableAst.gen_short_index(c_i)}\n'

    with open(f'{filename}', 'w') as f:
        # TODO: use pydot or graphviz library?
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
    source_tree = MutableAst(ast.parse(source_text))
    # make sure dest indices are distinct from source indices:
    dest_tree = MutableAst(ast.parse(dest_text))

    index_mapping = generate_mapping(source_tree, dest_tree)

    return source_tree, dest_tree, index_mapping
