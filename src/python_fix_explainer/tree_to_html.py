import ast
from collections import defaultdict
import asttokens
import logging
import re

from .muast import MutableAst, breadth_first
from .map_asts import generate_mapping, draw_comparison
from .gen_edit_script import EditScript, Action


action_classes = {
    Action.UPDATE: 'rename-node',
    Action.MOVE: 'move-node',
    Action.INSERT: 'insert-node',
    Action.DELETE: 'delete-node'
}


def extract_code_bit(code_str, pos):
    (start_lineno, start_col_offset), (end_lineno, end_col_offset) = pos
    code_lines = code_str.splitlines()
    extract_lines = []
    for line_i in range(start_lineno-1, end_lineno):  # lineno is 1-indexed
        code_to_add = code_lines[line_i]
        if line_i == end_lineno-1:
            code_to_add = code_to_add[:end_col_offset]
        if line_i == start_lineno-1:
            code_to_add = code_to_add[start_col_offset:]
        extract_lines.append(code_to_add)

    return '\n'.join(extract_lines)


def get_f_string_exprs(f_string_str):
    f_contents_match = re.match('f["\'](.*)["\']', f_string_str)
    if f_contents_match:
        f_contents_groups = f_contents_match.groups()
        if len(f_contents_groups) > 0:
            # offset from start of whole expression (vs. the inside of the string)
            f_contents_offset = f_contents_match.start(1)

            f_contents = f_contents_groups[0]
            formatted_values = [g.span() for g in re.finditer('{[^}]*}', f_contents)]
            all_children = []  # list of all expected top-level children of the JoinedStr node in the AST
            cursor = 0
            for start, end in formatted_values:
                if start > cursor:
                    all_children.append((cursor, start, 'Str'))
                all_children.append((start+1, end-1, 'FormattedValue'))
                cursor = end
            if cursor < len(f_contents):
                all_children.append((cursor, len(f_contents), 'Str'))
            return [(start+f_contents_offset, end+f_contents_offset, expected_type)
                    for (start, end, expected_type) in all_children]
    return []


# given a MutableAst object which represents a complete bit of code,
# generate html of the code, marked up with spans annotating code that "belongs" to specific nodes in the AST.
def gen_annotated_html(tree: MutableAst, id_prefix='', edit_script: EditScript = None):
    # We need to use yet another third-party library, asttokens, to be able to go from ast node to positions in code.
    # And in order to make asttokens work correctly
    # (e.g. when the ast has been edited, or when student code is not in canonical form),
    # we need to start from scratch:
    # Regenerate a fresh version of the text representation from the MutableAst, then generate the ast from that,
    # and a new MutableAst from the python ast.
    # Then we try our best to match the nodes of the original and new tree to each other and walk through them
    # together, to get the id/parent/etc. data from the original, and the text positions from the new one.

    node_to_edits = defaultdict(list)
    if edit_script:
        for edit in edit_script.edits:
            node_to_edits[edit.node_id].append(edit)

    # TODO: why didn't str(tree) work?
    # TODO: if compile fails, return flat str?

    txt = tree.to_compileable_str()
    # print(txt)
    py_ast = ast.parse(txt)
    atok = asttokens.ASTTokens(str(txt), tree=py_ast)
    new_tree = MutableAst(py_ast)

    tags = []

    # We use our ast mapping algorithm because tree and new_tree may not be identical,
    # even though they rerpresent identical code.
    #  sometimes the newly generated tree is structured differently from equivalent (original) tree
    #  e.g. by default return statement is not wrapped in Expr, but it is not wrong to have it in an Expr.
    #  and tree which came from edit script may retain Expr for shorter edit script.
    # TODO: maybe just use a single pass of APTED? shouldn't need to move subtrees...
    #  Especially since this can somehow result in thousands of calls to apted?..
    new_index_to_node = new_tree.gen_index_to_node()
    mapping = generate_mapping(tree, new_tree)
    # draw_comparison(tree, new_tree, mapping, 'tree_to_html_map.dot')
    orig_to_new_map = {orig_index: new_index for (orig_index, new_index) in mapping}
    orig_to_pos_map = {}  # map from orig node to position in code string

    # special inferences/calculations for positions of nodes that aren't calculated correctly by python
    # (e.g. inside of f strings)
    inferred_positions_for_lost_nodes = {}

    # at the same time, we still want to traverse the original tree in breadth-first order,
    # to control the order in which we process nodes that start/end in the same place (parents first)
    for i, orig_node in enumerate(breadth_first(tree)):
        if orig_node.index in orig_to_new_map:
            new_index = orig_to_new_map[orig_node.index]
            new_node = new_index_to_node[new_index]
            if not new_node.isList:

                (start_lineno, start_col_offset), (end_lineno, end_col_offset) = \
                    atok.get_text_positions(new_node.ast, padded=False)

                # check for special case where python fails to give back correct position (f string)
                if orig_node.name == 'JoinedStr':
                    f_string = extract_code_bit(txt, ((start_lineno, start_col_offset), (end_lineno, end_col_offset)))
                    for (start, end, expected_type), child_node in \
                            zip(get_f_string_exprs(f_string), orig_node.children_dict['values'].children):
                        # print(f_string[start:end], expected_type, child_node)
                        # TODO: make it correct for multiline f strings
                        inferred_positions_for_lost_nodes[child_node.index] = \
                            ((start_lineno, start_col_offset+start), (start_lineno, start_col_offset+end))

                # check whether appropriate position was found (or did atok return (1,0), (1,0) because it failed?)
                if ((start_lineno, start_col_offset), (end_lineno, end_col_offset)) == ((1, 0), (1, 0)):
                    if orig_node.index in inferred_positions_for_lost_nodes:
                        (start_lineno, start_col_offset), (end_lineno, end_col_offset) = \
                            inferred_positions_for_lost_nodes[orig_node.index]
                        # logging.info(f'Using inferred position for node: {orig_node.name}')
                    else:
                        # couldn't find position. place this node at the beginning of parent node.
                        # logging.warning(f'Couldn\'t find position for node: {orig_node.name}')

                        p = orig_node.parent
                        while p.isList:
                            p = p.parent
                        if p.index in orig_to_pos_map:
                            (parent_lineno, parent_offset), (_, _) = orig_to_pos_map[p.index]
                            ((start_lineno, start_col_offset), (end_lineno, end_col_offset)) = \
                                ((parent_lineno, parent_offset), (parent_lineno, parent_offset))

                orig_to_pos_map[orig_node.index] = ((start_lineno, start_col_offset), (end_lineno, end_col_offset))

                additional_classes = ''
                for e in node_to_edits[orig_node.index]:
                    additional_classes += f' {action_classes[e.action]}'

                attributes = f'class="ast-node{additional_classes}" ' \
                             f'id="{id_prefix}{orig_node.index}" ' \
                             f'data-node-id="{orig_node.index}" ' \
                             f'data-node-name="{orig_node.name}"'
                if orig_node.parent:
                    attributes += f' data-key={orig_node.key_in_parent}'
                    if orig_node.parent.isList:
                        attributes += f' data-parent-list-id="{orig_node.parent.index}"'
                tags.append((start_lineno, start_col_offset, i, f'<span {attributes}>'))
                # Try to ensure that start tags and end tags are ordered correctly when they are in the same spot
                # TODO: this means that several nodes which all start and end on the same character
                #  will always be one after the other, not nested. Need more nuanced logic if we want to retain nesting?
                tags.append((end_lineno, end_col_offset, i+0.5, f'</span>'))

    code_lines = txt.splitlines()
    # sort such that inserts happen from the end to the beginning,
    # and don't mess up string positions where the insert needs to happen
    tags.sort(reverse=True)
    for tag_data in tags:
        lineno, col_offset, _order_helper, tag = tag_data
        lineno -= 1  # line numbers are 1-indexed?!..
        code_lines[lineno] = code_lines[lineno][:col_offset] + tag + code_lines[lineno][col_offset:]

    return '\n'.join(code_lines)
