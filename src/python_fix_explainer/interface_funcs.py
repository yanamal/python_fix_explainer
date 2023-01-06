import ast
import logging
import time
from typing import List

from . import muast
from . import map_asts
from . import gen_edit_script
from . import simplify
from . import runtime_comparison
from . import tree_to_html
from . import get_runtime_effects
from . import map_bytecode


def generate_edit_scripts(incorrect_code: str, correct_versions: List[str]):
    incorrect_tree = muast.MutableAst(ast.parse(incorrect_code))
    correct_trees = [muast.MutableAst(ast.parse(c)) for c in correct_versions]

    edit_scripts: List[gen_edit_script.EditScript] = []
    for correct_tree in correct_trees:
        start_edit_script_time = time.time()
        index_mapping = map_asts.generate_mapping(incorrect_tree, correct_tree)
        edit_script = gen_edit_script.generate_edit_script(incorrect_tree, correct_tree, index_mapping)
        edit_scripts.append(edit_script)
        logging.info(f'Generating edit script took {(time.time() - start_edit_script_time)} seconds')

    # The  generated edit scripts work in the context of the specific MutableAst object of the original incorrect code
    # or at least one that uses the same uuids for the same nodes (e.g. a clone/copy)
    # so we need to return the tree we generated with those uuids
    return incorrect_tree, edit_scripts


# this simple wrapper around simplify.simplify_edit_script is not particularly necessary, but is here for completeness
# to have a function for each stage of the process
def simplify_and_choose_shortest(incorrect_tree: muast.MutableAst,
                                 problem_unit_tests: List[str],
                                 edit_scripts: List[gen_edit_script.EditScript]):
    simplified_scripts = [
        simplify.simplify_edit_script(incorrect_tree, problem_unit_tests, edit_script)
        for edit_script in edit_scripts
    ]

    return min(simplified_scripts, key=lambda x: x.edit_distance)


def generate_fix_sequence(incorrect_tree: muast.MutableAst,
                          problem_unit_tests: List[str],
                          edit_script: gen_edit_script.EditScript):
    start_generate_sequence = time.time()

    fully_corrected_tree = edit_script.apply(incorrect_tree)

    # each "fix" is an independent part of the edit script which fixes a specific part of the code.
    remaining_fixes = edit_script.dependent_blocks
    remaining_edit_script = edit_script
    ordered_fixes = []
    base_tree = incorrect_tree

    while len(remaining_fixes) > 0:
        # each time through this loop, pick the "best" set of fixes from remaining_fixes,
        # and apply them to make the new base_tree for the next iteration,
        # until no more fixes remain

        # generate runtime comparison between base code and fully corrected code
        base_to_corrected = [
            runtime_comparison.RuntimeComparison(base_tree, fully_corrected_tree, unit_test)
            for unit_test in problem_unit_tests]

        fix_effects = []
        best_tests = []
        for fix in remaining_fixes:
            # get subset of edit script for just applying this fix
            just_this_fix = remaining_edit_script.filtered_copy(lambda e: e.short_string not in fix)

            partial_solution = just_this_fix.apply(base_tree)
            partial_to_corrected = [
                runtime_comparison.RuntimeComparison(partial_solution, fully_corrected_tree, unit_test)
                for unit_test in problem_unit_tests]
            fix_effect, best_test_i = runtime_comparison.compare_comparisons(base_to_corrected, partial_to_corrected)
            fix_effects.append(fix_effect)
            best_tests.append(problem_unit_tests[best_test_i])

        best_effect = max(fix_effects)  # best possible fix effect achived with given base code and fixes
        applied_fixes = []
        for fix, effect, test in zip(remaining_fixes, fix_effects, best_tests):
            if effect == best_effect:
                # apply this fix, and mark it as having been applied
                just_this_fix = remaining_edit_script.filtered_copy(lambda e: e.short_string not in fix)
                base_tree = just_this_fix.apply(base_tree)
                ordered_fixes.append((just_this_fix, test))  # the actual edit script, and the illustrative unit test
                applied_fixes.append(fix)  # the fix "block" that's in remaining_fixes

        for fix in applied_fixes:
            remaining_fixes.remove(fix)

    logging.info(
        f'Generating fix sequence of {len(ordered_fixes)} took {time.time() - start_generate_sequence} seconds')
    return ordered_fixes


def get_run_trace(code_tree, test_string):
    code_trace = get_runtime_effects.run_test(str(code_tree), test_string)
    op_to_node = map_bytecode.gen_op_to_node_mapping(code_tree)
    node_sequence = runtime_comparison.get_runtime_node_sequence(code_trace.ops_list, op_to_node)

    node_trace = []
    for node, op in zip(node_sequence, code_trace.ops_list):
        node_trace.append({
            'node': node,
            'values': op.pushed_values
        })
    return node_trace


def fix_code(incorrect_code: str,
             problem_unit_tests: List[str],
             correct_versions: List[str]):

    # run the 3-stage pipeline to generate the sequence of explainable fixes:
    incorrect_tree, edit_scripts = generate_edit_scripts(incorrect_code, correct_versions)
    shortest_edit_script = simplify_and_choose_shortest(incorrect_tree, problem_unit_tests, edit_scripts)
    fully_corrected_tree = shortest_edit_script.apply(incorrect_tree)
    fix_sequence = generate_fix_sequence(incorrect_tree, problem_unit_tests, shortest_edit_script)

    # generate output for the interface to consume:
    start_calculate_effect = time.time()
    code_sequence = []
    current_tree = incorrect_tree
    for i, (fix, illustrative_unit_test) in enumerate(fix_sequence):
        # Generate annotated html, with the context of the current fix, both before and after the fix
        # some fix data (deletes) will only exist in pre-fix,
        # some (inserts) only in post-fix,
        # and some (moves, renames) in both versions.
        # the interface will be responsible for aggregating this data into a visualization of what changes in the fix
        pre_fix_html = tree_to_html.gen_annotated_html(current_tree, id_prefix=f'before_step_{i}_', edit_script=fix)
        new_tree = fix.apply(current_tree)
        post_fix_html = tree_to_html.gen_annotated_html(new_tree, id_prefix=f'after_step_{i}_', edit_script=fix)

        fix_effect = \
            runtime_comparison.FixEffectComparison(current_tree, new_tree, fully_corrected_tree, illustrative_unit_test)

        current_tree = new_tree
        code_sequence.append({
            'source': pre_fix_html,
            'dest': post_fix_html,
            'unit_test_string': illustrative_unit_test,
            'synced_trace': fix_effect.synced_node_trace,
            'points_of_interest': fix_effect.notable_ops_in_synced,
            'effect_summary': fix_effect.summary_string
        })

    logging.info(f'Generating fix effects and html output took {time.time() - start_calculate_effect} seconds')

    # generate the final code state, without any edit script markup
    # (there are no more edits to apply to this final state)
    final_html = tree_to_html.gen_annotated_html(current_tree, id_prefix=f'step_{len(fix_sequence)}_')

    return {
        'fix_sequence': code_sequence,
        'final_code': final_html
    }
