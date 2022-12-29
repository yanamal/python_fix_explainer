import ast
from collections import defaultdict
import asttokens

from .muast import MutableAst, breadth_first
from .map_asts import generate_mapping
from .gen_edit_script import EditScript, Action


action_classes = {
    Action.UPDATE: 'rename-node',
    Action.MOVE: 'move-node',
    Action.INSERT: 'insert-node',
    Action.DELETE: 'delete-node'
}


# given a MutableAst object which represents a complete bit of code,
# generate html of the code, marked up with spans annotating code that "belongs" to specific nodes in the AST.
def gen_annotated_html(tree: MutableAst, id_prefix='', edit_script: EditScript = None):
    # We need to use yet another third-party library, asttokens, to be able to go from ast node to positions in code.
    # And in order to make asttokens work correctly, we need to start from scratch:
    # Regenerate a fresh version of the text representation from the MutableAst, then generate the ast from that,
    # and a new MutableAst from the python ast.
    # Then we try our best to match the nodes of the original and new tree to each other nd walk through them
    # together, to get the id/parent/etc. data from the original, and the text positions from the new one.

    node_to_edits = defaultdict(list)
    if edit_script:
        for edit in edit_script.edits:
            node_to_edits[edit.node_id].append(edit)

    txt = str(tree)
    py_ast = ast.parse(txt)
    atok = asttokens.ASTTokens(str(txt), tree=py_ast)
    new_tree = MutableAst(py_ast)

    orig_index_to_node = tree.gen_index_to_node()
    new_index_to_node = new_tree.gen_index_to_node()

    tags = []

    # We use our ast mapping algorithm because tree and new_tree may not be identical,
    # even though they rerpresent identical code.
    #  sometimes the newly generated tree is structured differently from equivalent (original) tree
    #  e.g. by default return statement is not wrapped in Expr, but it is not wrong to have it in an Expr.
    #  and tree which came from edit script may retain Expr for shorter edit script.
    for i, (orig_index, new_index) in enumerate(generate_mapping(tree, new_tree)):
        new_node = new_index_to_node[new_index]
        orig_node = orig_index_to_node[orig_index]
        if not new_node.isList:
            (start_lineno, start_col_offset), (end_lineno, end_col_offset) = \
                atok.get_text_positions(new_node.ast, padded=False)

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
            tags.append((end_lineno, end_col_offset, i, f'</span>'))
            # TODO: try to ensure that start tags and end tags are ordered correctly when they are in the same spot

    code_lines = txt.splitlines()
    # sort such that inserts happen from the end to the beginning,
    # and don't mess up string positions where the insert needs to happen
    tags.sort(reverse=True)
    for tag_data in tags:
        lineno, col_offset, _order_helper, tag = tag_data
        lineno -= 1  # line numbers are 1-indexed?!..
        code_lines[lineno] = code_lines[lineno][:col_offset] + tag + code_lines[lineno][col_offset:]

    return '\n'.join(code_lines)
