# Logic for simplifying an edit script
# (removing some subset of edits s.t. the final edited code is still correct according to some set of tests)
# TODO: consider re-introducing constant unrolling option (is it still necessary with the "=" mapping heuristic?)
import logging
import time
from typing import List

from .muast import MutableAst
from .gen_edit_script import EditScript, Edit, Action


# Simplification step 1: find and undo edits that just rename variables
def simplify_var_renames(source_tree: MutableAst, problem_tests: List[str], edit_script: EditScript):
    for old_name, new_name in edit_script.var_renames.items():
        # for each variable name pair

        def rename_filter(e: Edit):
            return e.is_rename and e.new_name == new_name and e.old_name == old_name and e.action == Action.UPDATE

        without_rename = edit_script.filtered_copy(rename_filter)

        # find all var names with new name in the additional nodes taken from "correct" tree
        # and rename them back into the old name.
        inserted_nodes = without_rename.additional_nodes
        for index, node in inserted_nodes.items():
            if type(node.ast).__name__ == 'Name' and node.ast.id == new_name:
                node.ast.id = old_name
            elif type(node.ast).__name__ == 'arg' and node.ast.arg == new_name:
                node.ast.arg = old_name
            # TODO: regenerate MutableAst node with appropriate name

        # Try out the new edit script
        tree_without_rename = without_rename.apply(source_tree)
        if False not in tree_without_rename.test(problem_tests):
            # if all unit tests pass in this new version of the tree
            logging.debug(
                f'Simplified: removed {len(edit_script.edits) - len(without_rename.edits)} '
                f'edits renaming {old_name} -> {new_name}')
            edit_script = without_rename
        else:
            logging.debug(f'Could not remove rename of {old_name} -> {new_name}')
            '''print(source_tree)
            print(dest_tree)
            print(tree_without_rename)'''
    return edit_script


# Simplification step 2: try removing each block of mutually-dependent edits to see if code is still correct without it.
# Repeat until no more blocks can be removed.
def simplify_dependent_blocks(
        source_tree: MutableAst, problem_tests: List[str], edit_script: EditScript):
    potential_removals = edit_script.dependent_blocks

    # sort in descending order based on earliest edit in the block.
    # TODO: This doesn't do anything sensible since indices are no longer sensbily orderered integers.
    #  What was it for again?..
    # edit_str_to_index = {e.short_string: i for i, e in enumerate(edit_script.edits)}
    # potential_removals.sort(key=lambda es: min([edit_str_to_index[e] for e in es]), reverse=True)

    keep_going = True
    i = 0
    while keep_going:
        start_iteration = time.time()
        i += 1
        keep_going = False
        remaining_potential_removals = []
        for remove_set in potential_removals:
            logging.debug(f'Trying to removed edits: {remove_set}')

            # filter function: is this edit in the set of edits to remove
            def is_in_remove_set(e: Edit):
                return e.short_string in remove_set

            edit_script.filtered_copy(is_in_remove_set)
            without_removed = edit_script.filtered_copy(is_in_remove_set)
            tree_without_removed = without_removed.apply(source_tree)
            if len(without_removed.edits) < len(edit_script.edits) and \
                    False not in tree_without_removed.test(problem_tests):
                edit_script = without_removed
                keep_going = True
            else:
                logging.debug(f'could not remove edits.')  # Unit tests:  {tree_without_removed.test(problem_tests)}')
                remaining_potential_removals.append(remove_set)
            potential_removals = remaining_potential_removals
        logging.info(f'Simplification iteration {i} took {time.time() - start_iteration} seconds')

    logging.info(f'{i} simplification iterations')

    # return resulting (shorter) edit script and grouped edits which remain.
    return edit_script, potential_removals


def simplify_edit_script(source_tree: MutableAst, problem_tests: List[str], edit_script: EditScript):
    start_simplify = time.time()
    start_len = len(edit_script.edits)
    # Simplify edit script by trying to remove blocks of connected dependencies

    ### Try skipping variable renames:
    edit_script = simplify_var_renames(source_tree, problem_tests, edit_script)

    ### Get connected components of 'dependent' edits
    # TODO: use runtime improvement information to simplify dependent blocks?..
    edit_script, remaining_blocks = simplify_dependent_blocks(
        source_tree, problem_tests, edit_script)

    for b in remaining_blocks:
        logging.debug('block start:')
        for e_i in b:
            logging.debug(e_i)

    logging.info(f'Simplifying edit script from {start_len} to {len(edit_script.edits)} edits'
                 f' took {time.time() - start_simplify} seconds')
    return edit_script
