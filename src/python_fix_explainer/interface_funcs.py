import ast
import copy
import logging
import time
from typing import List, Dict

import doctest

from . import muast
from . import map_asts
from . import gen_edit_script
from . import simplify
from . import runtime_comparison
from . import tree_to_html
from . import get_runtime_effects
from . import map_bytecode
# from .doctest_data import doctestrunner, doctests
from . import doctest_data


# TODO: make a class for "problem specification" which includes correct code, unit tests, pre/post code, other metadata
#  can include things that only need to get generated once, e.g. runtime data including trace, expected ops count, etc.
# TODO: make a class for output format(fix sequence, etc.), then have a to_json conversion for external output?


def generate_edit_scripts(incorrect_code: str, correct_versions: List[str]):
    incorrect_tree = muast.MutableAst(ast.parse(incorrect_code))
    correct_trees = [muast.MutableAst(ast.parse(c)) for c in correct_versions]

    edit_scripts: List[gen_edit_script.EditScript] = []
    for correct_tree in correct_trees:
        start_edit_script_time = time.time()
        index_mapping = map_asts.generate_mapping(incorrect_tree, correct_tree)
        # map_asts.draw_comparison(incorrect_tree, correct_tree, index_mapping, 'mapping.dot')
        edit_script = gen_edit_script.generate_edit_script(incorrect_tree, correct_tree, index_mapping)
        edit_scripts.append(edit_script)
        logging.info(f'Generating edit script took {(time.time() - start_edit_script_time)} seconds')

    # The  generated edit scripts work in the context of the specific MutableAst object of the original incorrect code
    # or at least one that uses the same uuids for the same nodes (e.g. a clone/copy)
    # so we need to return the tree we generated with those uuids
    return incorrect_tree, edit_scripts


def html_from_edit_script(before_tree: muast.MutableAst,
                          edit_script: gen_edit_script.EditScript,
                          before_prefix: str = '', after_prefix: str = ''):
    start_tree_gen_time = time.time()
    after_tree = edit_script.apply(before_tree)
    before_html = tree_to_html.gen_annotated_html(before_tree, before_prefix, edit_script)
    after_html = tree_to_html.gen_annotated_html(after_tree, after_prefix, edit_script)
    logging.info(f'Generating html output took {(time.time() - start_tree_gen_time)} seconds')
    return before_html, after_html


# this simple wrapper around simplify.simplify_edit_script is not particularly necessary, but is here for completeness
# to have a function for each stage of the process
def simplify_and_choose_shortest(incorrect_tree: muast.MutableAst,
                                 problem_unit_tests: List[str],
                                 edit_scripts: List[gen_edit_script.EditScript],
                                 prepend_code: str = '',
                                 append_code: str = ''):
    simplified_scripts = [
        simplify.simplify_edit_script(
            incorrect_tree, problem_unit_tests, edit_script, prepend_code=prepend_code, append_code=append_code)
        for edit_script in edit_scripts
    ]

    return min(simplified_scripts, key=lambda x: x.edit_distance)


def generate_static_fix_sequence(incorrect_tree: muast.MutableAst,
                                 edit_script: gen_edit_script.EditScript):
    ordered_fixes = []
    if len(edit_script.dependent_blocks) >= 15:
        # static fix sequence with too many fixes - omit fix sequence
        # TODO: update static explanation if this happens?..
        print(f'too many fixes - {len(edit_script.dependent_blocks)}')
        return ordered_fixes
    for edit_string_list in edit_script.dependent_blocks:
        fix = edit_script.filtered_copy(lambda e: e.short_string not in edit_string_list)
        ordered_fixes.append((fix, None))
    return ordered_fixes


# TODO: we could/should be merging pairs of fixes where one fix makes things worse, but a separate fix then repairs it
#   e.g. one fix adds a second parameter, the next fix removes the original parameter (instead of replacing).
#   But the logic for detecting that would probably have to look at runtime data in a more elaborate way
#   (these fixes fix/break the same place so we can probably merge them;
#   these fixes change different things in the runtime so they are probably independent)
def generate_fix_sequence(incorrect_tree: muast.MutableAst,
                          problem_unit_tests: List[str],
                          edit_script: gen_edit_script.EditScript,
                          prepend_code: str = '',
                          append_code: str = ''):
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
            runtime_comparison.RuntimeComparison(base_tree, fully_corrected_tree, unit_test,
                                                 prepend_code=prepend_code, append_code=append_code)
            for unit_test in problem_unit_tests
        ]

        fix_effects = []
        best_tests = []
        # calculate effect of each remaining fix (independent of others)
        # TODO: perhaps stop as soon as we find a fix that makes things better, and apply it, then resume main loop
        #  then will only have to evaluate all fixes if there are no good fixes (in that case, merge down/up?)
        for fix in remaining_fixes:

            # get subset of edit script for just applying this fix
            just_this_fix = remaining_edit_script.filtered_copy(lambda e: e.short_string not in fix)

            partial_solution = just_this_fix.apply(base_tree)

            # for e in just_this_fix.edits:
            #     print(e.short_string)
            try:
                partial_to_corrected = [
                    runtime_comparison.RuntimeComparison(partial_solution, fully_corrected_tree, unit_test,
                                                         prepend_code=prepend_code, append_code=append_code)
                    for unit_test in problem_unit_tests
                ]
                fix_effect, best_test_i = runtime_comparison.compare_comparisons(base_to_corrected,
                                                                                 partial_to_corrected)
                fix_effects.append(fix_effect)
                best_tests.append(problem_unit_tests[best_test_i])
            except:  # noqa
                fix_effects.append(runtime_comparison.Effect.BREAKS)
                best_tests.append(None)  # no illustrative unit test for this "fix"

        # apply fix(es) wit the best available effect
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

        # Check whether the current tree already represents correct code (in that case, skip any remaining fixes)
        # TODO: possibly do this for every fix, not every "batch" with same quality? more messy, and maybe no benefit?
        if all(base_tree.test(problem_unit_tests, prepend_code=prepend_code, append_code=append_code)):
            break

    # if there are fixes remaining, we must have stopped early. update edit script
    if len(remaining_fixes) > 0:
        flat_fixes = [fix for fix_set in remaining_fixes for fix in fix_set]
        edit_script = edit_script.filtered_copy(lambda e: e.short_string in flat_fixes)

    logging.info(
        f'Generating fix sequence of {len(ordered_fixes)} took {time.time() - start_generate_sequence} seconds')
    return ordered_fixes, edit_script


def get_run_trace(code_tree: muast.MutableAst, test_string: str, prepend_code: str = '', append_code: str = ''):
    op_to_node = map_bytecode.gen_op_to_node_mapping(code_tree)
    code_trace = get_runtime_effects.run_test(
        str(code_tree), test_string, prepend_code=prepend_code, append_code=append_code)
    node_sequence = runtime_comparison.get_runtime_node_sequence(code_trace.ops_list, op_to_node)

    node_trace = []
    for node, op in zip(node_sequence, code_trace.ops_list):
        node_trace.append({
            'node': node,
            'values': op.pushed_values
        })
    return node_trace


def fix_code_doctest_tests(incorrect_code: str,
                           doctest_tests: Dict[str, str],
                           correct_versions: List[str],
                           prepend_code: str = '',
                           append_code: str = ''):
    # TODO:
    #  2. parse tests and store the calls
    #    2a. make simple function which calls specific test from the datastructure
    #  3. call fix_code with resulting test strings

    doctest_data.doctests = {}
    # parse tests and store in global variable
    for test_name in doctest_tests:
        # first, we have to parse the test string. It is parsed into "a list of alternating Examples and strings"
        #  (https://docs.python.org/3/library/doctest.html#doctestparser-objects)
        parsed_test = doctest.DocTestParser().parse(doctest_tests[test_name], test_name)
        # we only want the Example objects (and we expect exactly one of them per test. This is true for Otter tests.)
        example_only_list = [e for e in parsed_test if isinstance(e, doctest.Example)]
        doctest_data.doctests[test_name] = example_only_list

    # run fix_code and use the tests in doctests via the run_doctest_test function
    test_strings = [f'run_doctest_test("{test_name}", globals())' for test_name in doctest_tests]
    return fix_code(incorrect_code, test_strings, correct_versions, prepend_code, append_code)

# sequence of progressively more detailed analysis:
# 1. run unit tests. if unit tests pass, stop there
# 2. run static comparison and return all edit scripts (note shortest? return html diff?)
# 3. simplify_and_choose_shortest (run intermediate versions, but not instrumented)
# 3a. return sequence of fixes for shortest edit script? (but nobody looks at that anyway)
# 4a. (shortcut) - generate side-by-side runtime analysis for entire shortest script (automatically return with step 3?)
# 4b. generate sequence of fixes with separate side-by-side runtime analysis, as before
# TODO: are intermediate result data types (EditScript, MutableAST) serializable??
#  if not, where do we store them in between requests?
#  In an optimal scenario (more than one worker), wouldn't necessarily need to store intermediate data,
#  just push out parsed results (html, fix effect) to database and continue working.
#  But if the one existing worker is stuck processing one problem anyway,
#  then benefit of having intermediate results may be moot.


def fix_code(incorrect_code: str,
             problem_unit_tests: List[str],
             correct_versions: List[str],
             prepend_code: str = '',
             append_code: str = ''):
    static_only_reason = None  # reason that only static analysis was done (no simplification or runtime analysis)
    zero_fixes_reason = None  # reason that zero fixes were generated

    # run the 3-stage pipeline to generate the sequence of explainable fixes;
    # run abbreviated pipeline in degenerate cases
    incorrect_tree, edit_scripts = generate_edit_scripts(incorrect_code, correct_versions)
    if len(correct_versions) == 0:
        zero_fixes_reason = "There are no correct solutions associated with this problem"
        fix_sequence = []
        fully_corrected_tree = incorrect_tree
        shortest_edit_script = gen_edit_script.EditScript(
            edits=[],
            additional_nodes={},
            var_renames={},
            source_tree=incorrect_tree
        )
        # TODO: could also set correct to student code and run the analysis? less efficient, but less manual
    else:
        # at least one correct code version exists.
        if len(problem_unit_tests) == 0:
            shortest_edit_script = min(edit_scripts, key=lambda x: x.edit_distance)
            static_only_reason = "There are no unit tests associated with this problem"
            fix_sequence = generate_static_fix_sequence(incorrect_tree, shortest_edit_script)
        else:
            shortest_edit_script = simplify_and_choose_shortest(
                incorrect_tree, problem_unit_tests, edit_scripts, prepend_code=prepend_code, append_code=append_code)
            # fix_sequence, shortest_edit_script = generate_fix_sequence(
            #     incorrect_tree, problem_unit_tests, shortest_edit_script)
            # TODO: no more try/except? should be handled upstream now
            try:
                fix_sequence, shortest_edit_script = generate_fix_sequence(
                    incorrect_tree, problem_unit_tests, shortest_edit_script,
                    prepend_code=prepend_code, append_code=append_code)
                if len(fix_sequence) == 0:
                    # TODO: perhaps detect this much earlier
                    zero_fixes_reason = "The student code passes unit tests without any fixes."
            except SyntaxError:
                fix_sequence = generate_static_fix_sequence(incorrect_tree, shortest_edit_script)
                static_only_reason = "Could not compile one of the intermediate versions of the code"
        # TODO: zero fixes reason for static code?.. (too many)

        fully_corrected_tree = shortest_edit_script.apply(incorrect_tree)

    # # Apply final fix sequence to incorrect code to generate final version of correct code:
    # # TODO: also regenerate/re-create shortest edit script (may be even shorter after ordering)
    # fully_corrected_tree = incorrect_tree
    # for fix, _illustrative_unit_test in fix_sequence:
    #     fully_corrected_tree = fix.apply(fully_corrected_tree)

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
        try:
            post_fix_html = tree_to_html.gen_annotated_html(new_tree, id_prefix=f'after_step_{i}_', edit_script=fix)
        except SyntaxError:
            code_sequence.append({
                'source': pre_fix_html,
                'dest': new_tree.to_compileable_str(),
                'disclaimer':
                    'The code generated by this fix does not execute '
                    '(this fix should have probably been combined with another fix)',
            })
            continue

        if illustrative_unit_test:
            fix_effect = \
                runtime_comparison.FixEffectComparison(
                    current_tree, new_tree, fully_corrected_tree, illustrative_unit_test,
                    prepend_code=prepend_code, append_code=append_code)

            code_sequence.append({
                'source': pre_fix_html,
                'dest': post_fix_html,
                'unit_test_string': illustrative_unit_test,
                'synced_trace': fix_effect.synced_node_trace,
                'points_of_interest': fix_effect.notable_ops_in_synced,
                'effect_summary': fix_effect.summary_string
            })
            # print(new_tree)
            # print(illustrative_unit_test)
            # print(fix_effect.summary_string)
        else:
            code_sequence.append({
                'source': pre_fix_html,
                'dest': post_fix_html,
                'disclaimer': 'No runtime analysis was generated for this fix'
            })

        current_tree = new_tree

    logging.info(f'Generating fix effects and per-fix html output took {time.time() - start_calculate_effect} seconds')

    # TODO: why was I using current_tree below for the correct tree?..
    start_gen_html = time.time()
    # generate the final code state, without any edit script markup
    # (there are no more edits to apply to this final state)
    final_html = tree_to_html.gen_annotated_html(fully_corrected_tree, id_prefix=f'step_{len(fix_sequence)}_')

    # also generate both "before" and "after" code state, *with* edit script markup, for side-by-side view
    before_html = tree_to_html.gen_annotated_html(incorrect_tree, id_prefix=f'before_',
                                                  edit_script=shortest_edit_script)
    after_html = tree_to_html.gen_annotated_html(fully_corrected_tree, id_prefix=f'after_',
                                                 edit_script=shortest_edit_script)

    logging.info(f'Generating html for overview took {time.time() - start_gen_html} seconds')

    # print(fully_corrected_tree)
    return {
        'fix_sequence': code_sequence,
        'final_code': final_html,  # TODO: rename so it's clear this is an html representation of the code
        'fixed_code_string': fully_corrected_tree.to_compileable_str(),
        'is_static_analysis_only': (static_only_reason is not None),
        'static_only_reason': static_only_reason,
        'is_zero_fixes': (zero_fixes_reason is not None),
        'zero_fixes_reason': zero_fixes_reason,
        'overall_comparison': {
            'before': before_html,
            'after': after_html
        }
    }


# Generate output that is usable directly in an html interface
# to show the difference between incorrect code and some corrected version
# (without splitting the entire diff into "fixes")
def generate_fix_effect_output(incorrect_tree, edit_script, unit_test,
                               prepend_code: str = '',
                               append_code: str = ''):
    corrected_tree = edit_script.apply(incorrect_tree)
    student_html = tree_to_html.gen_annotated_html(incorrect_tree, 'student_code_only_')
    incorrect_html = tree_to_html.gen_annotated_html(incorrect_tree, 'student_code_', edit_script=edit_script)
    correct_html = tree_to_html.gen_annotated_html(corrected_tree, 'corrected_code_', edit_script=edit_script)

    fix_all_effect = runtime_comparison.FixEffectComparison(
        before_fix=incorrect_tree, after_fix=corrected_tree, fully_correct=corrected_tree,
        test_string=unit_test, prepend_code=prepend_code, append_code=append_code
    )
    return {
                'source': incorrect_html,
                'dest': correct_html,
                'source_no_diff': student_html,
                'dest_string': corrected_tree.to_compileable_str(),
                'source_string': incorrect_tree.to_compileable_str(),
                'unit_test_string': unit_test,
                'synced_trace': fix_all_effect.synced_node_trace,
                'points_of_interest': fix_all_effect.notable_ops_in_synced,
                'effect_summary': fix_all_effect.summary_string
            }


# Generate data for single correction (without splitting into individual fixes)
def generate_correction( incorrect_code: str,
                         problem_unit_tests: List[str],
                         correct_versions: List[str],
                         prepend_code: str = '',
                         append_code: str = ''):
    # Step 1. Run unit tests against student code; check whether it fails any
    test_results = test_all([incorrect_code] + correct_versions, problem_unit_tests,
                            prepend_code=prepend_code, append_code=append_code)
    student_test_results = test_results[0]
    solution_test_results = test_results[1:]  # TODO: warn/error if solutions do not pass unit tests

    if len(student_test_results) > 0 and all(student_test_results):
        print('student solution is correct')
        return {
            'source': incorrect_code,
            'dest': incorrect_code,
            'disclaimer': 'No analysis generated - student solution passes unit tests'
        }

    # Step 2. Run static code diff between student solution and each correct solution

    if len(correct_versions) == 0:
        print('no correct solutions exist')
        return {
            'source': incorrect_code,
            'dest': incorrect_code,
            'disclaimer': 'No analysis generated - no canonical solutions provided'
        }

    incorrect_tree, edit_scripts = generate_edit_scripts(incorrect_code, correct_versions)

    # Step 3. Simplify each edit script and choose the shortest one for runtime analysis.

    if len(problem_unit_tests) == 0:
        print('no unit tests')
        # no runtime analysis is possible; choose closest existing solution
        shortest_len, shortest_i = min([(len(e.edits), i) for i, e in enumerate(edit_scripts)])
        student_code_html, solution_code_html = html_from_edit_script(
            incorrect_tree, edit_scripts[shortest_i], 'student_code_', f'solution_{shortest_i}_code_')
        return {
            'source': student_code_html,
            'dest': solution_code_html,
            'disclaimer': 'Static analysis only - no unit tests provided'
        }

    shortest_script = simplify_and_choose_shortest(incorrect_tree, problem_unit_tests, edit_scripts,
                                                   prepend_code=prepend_code, append_code=append_code)

    # Step 3.5 (generating individual fixes would have been step 4)
    # Generate runtime analysis for full diff between student code and version generated with edit script
    # Use one of the failing unit tests
    for i, result in enumerate(student_test_results):
        if not result:
            return generate_fix_effect_output(incorrect_tree, shortest_script, problem_unit_tests[i],
                                              prepend_code=prepend_code, append_code=append_code)


# Test the (presumed to be correct) solutions against all the unit tests.
# if all tests pass on all solutions, return False.
# otherwise, return index of solution and unit test where the unit test fails on that solution.
def has_failing_unit_test(solutions: List[str], tests: List[str], prepend_code: str = '', append_code: str = ''):
    for s_i, s in enumerate(solutions):
        s_tree = muast.MutableAst(ast.parse(s))
        test_results = s_tree.test(tests, prepend_code=prepend_code, append_code=append_code)
        for r_i, result in enumerate(test_results):
            if not result:
                return s_i, r_i
    return False


# Similar to has_failing_unit_test, but actually return all results of each unit test on each solution.
def test_all(solutions: List[str], tests: List[str], prepend_code: str = '', append_code: str = ''):
    all_results = []
    for s in solutions:
        s_tree = muast.MutableAst(ast.parse(s))
        test_results = s_tree.test(tests, prepend_code=prepend_code, append_code=append_code)
        all_results.append(test_results)
    return all_results
