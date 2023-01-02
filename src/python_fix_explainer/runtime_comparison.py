# logic for comparing the execution of two versions of some code (usually partially-buggy code to corrected version)
# and for comparing these comparisons:
# given two partially-buggy versions (e.g. original student code and student code with partial fix)
# and one totally corrected version, all of the same basic code, how does each buggy version compare to the corrected?
# which buggy-to-correct comparison is better/worse in terms of the buggy version being closer/further away
# from the corrected?
import difflib
from enum import Enum
from functools import total_ordering
from typing import List
from dataclasses import dataclass

from . import muast
from . import get_runtime_effects
from . import map_bytecode


# Given a sequence of ops executed at runtime, and an op-to-node_id mapping,
# return an equivalent sequence using node_ids,
# being careful about ops that aren't mapped to any nodes
def get_runtime_node_sequence(
        runtime_op_sequence: List[get_runtime_effects.TracedOp],
        op_to_node: dict,
        default_prefix: str = 'unmapped_op'):

    return \
        [
            (
                op_to_node[op_trace.op_id]
                if op_trace.op_id in op_to_node
                else str(f'unmapped: {default_prefix} {op_trace.op_id}')
            )
            for op_trace in runtime_op_sequence
        ]


# Helper class for RuntimeComparison below.
# A dataclass for tracking metadata about a particular executed bytecode op in a sequence resulting from
# an execution trace.
# The metadata describes whether (and how) this op is mapped onto an op from another sequence that is being compared to
# this sequence.
@dataclass
class RuntimeOpMappingData:
    is_mapped: bool = False
    mapped_op_index: int = -1
    value_matches: bool = False


def filter_unmapped_ops(op_trace: get_runtime_effects.TracedRunResult, node_trace: List[str]):
    new_ops_list: List[get_runtime_effects.TracedOp] = []
    new_nodes_list: List[str] = []
    for op, node in zip(op_trace.ops_list, node_trace):
        if not node.startswith('unmapped:'):
            new_ops_list.append(op)
            new_nodes_list.append(node)
    return new_ops_list, new_nodes_list


# A class for computing and storing runtime comparison between two versions of some code,
# running against the same unit test.
# in our use case, this is usually a comparison between some buggy version ("source")
# and a fully corrected version("dest").

# This class implements a total ordering, so that we can compare comparisons,
# under the assumption that the "dest" code is the same in both RuntimeComparisons being compared,
# and represents a canonical expected/correct way the code should run.
@total_ordering
class RuntimeComparison:
    def __init__(self, source_tree: muast.MutableAst, dest_tree: muast.MutableAst, test_string: str):
        # Store basic info
        self.source_tree = source_tree
        self.dest_tree = dest_tree
        self.test_string = test_string

        self.source_index_to_node = {n.index: n for n in muast.breadth_first(source_tree)}
        self.source_code = source_tree.to_compileable_str()
        self.dest_code = dest_tree.to_compileable_str()

        # TODO: compute some of these things (e.g. dest static mappings)
        #  less often than once per unit test run per candidate source_tree?
        # compute and store traces of running unit test for each version
        self.source_trace = get_runtime_effects.run_test(self.source_code, test_string)
        self.dest_trace = get_runtime_effects.run_test(self.dest_code, test_string)

        # record run outcomes of source code for easy access:
        self.run_status = self.source_trace.run_outcome
        self.run_completed = (self.source_trace.run_outcome == 'completed')
        self.test_passed = self.source_trace.eval_result

        self.source_op_to_node = map_bytecode.gen_op_to_node_mapping(source_tree)
        self.dest_op_to_node = map_bytecode.gen_op_to_node_mapping(dest_tree)

        ### Find the longest common subsequence (LCS) between the two runtime traces.

        # for matching the sequences, use ids of AST nodes that correspond to the bytecode ops that were executed
        self.source_node_trace = get_runtime_node_sequence(
            self.source_trace.ops_list, self.source_op_to_node, default_prefix='source')
        self.dest_node_trace = get_runtime_node_sequence(
            self.dest_trace.ops_list, self.dest_op_to_node, default_prefix='dest')

        # filter out unmapped nodes from both versions of each trace
        # (original ops trace and list of corresponding nodes)
        # These nodes usually (almost always?) correspond to code that is not part of the student code,
        # e.g. unit test code.
        # seeing them in the trace with no info is just confusing.
        self.source_trace.ops_list, self.source_node_trace = \
            filter_unmapped_ops(self.source_trace, self.source_node_trace)
        self.dest_trace.ops_list, self.dest_node_trace = \
            filter_unmapped_ops(self.dest_trace, self.dest_node_trace)

        # Note: this logic depends on the assumption that source_tree and dest_tree share the same node ids
        # for nodes that are mapped to each other through the AST mapping between them
        # This assumption is true when dest_tree was generated by applying an edit script to the source_tree.

        # use SequenceMatcher to find the LCS based solely on matching AST nodes.
        # TODO: use opcode/name for sequence matcher too? or too restrictive?
        #  maybe label mapped ops in bytecode as (nodeid, i) instead of just nodeid?
        runtime_diff = difflib.SequenceMatcher(None, self.source_node_trace, self.dest_node_trace, autojunk=False).get_opcodes()

        # Now go through the LCS result to:
        # (1) create explicit maps between indices in the two runtime sequences that were matched in the LCS
        # (2) record where *output values* of running matched nodes matched and where they diverged
        # (2.5) find the last op where these values matched
        #       (this is the "deviation point" - after this, the two runs deviate from each other
        #        and are no longer computing the same thing)
        self.total_match_size = 0
        self.last_matching_val_dest = 0  # a measure of how far we got in the dest script while actually matching values
        self.last_matching_val_source = 0  # same for source script

        # lists of RuntimeOpMappingData that track the mapping metadata between the two traces in both directions
        self.source_runtime_mapping_to_dest = [RuntimeOpMappingData() for _ in range(len(self.source_node_trace))]
        self.dest_runtime_mapping_to_source = [RuntimeOpMappingData() for _ in range(len(self.dest_node_trace))]

        for tag, i1, i2, j1, j2 in runtime_diff:
            if tag == 'equal':
                self.total_match_size += (i2 - i1)
                for s_i, d_i in zip(range(i1, i2), range(j1, j2)):
                    # Update metadata to reflect that the ops with these indices in the trace are in fact mapped
                    self.source_runtime_mapping_to_dest[s_i].is_mapped = True
                    self.dest_runtime_mapping_to_source[d_i].is_mapped = True
                    # Record which index maps to which, in both directions
                    self.source_runtime_mapping_to_dest[s_i].mapped_op_index = d_i
                    self.dest_runtime_mapping_to_source[d_i].mapped_op_index = s_i
                    # Find matching values
                    source_vals = self.source_trace.ops_list[s_i].pushed_values
                    dest_vals = self.dest_trace.ops_list[d_i].pushed_values

                    # Note: we do not want to record the absense of values as actively "matching",
                    # because we are looking for the place where the values being calculated actually diverge
                    # and stop matching, and never start matching again.
                    # A "matching" lack of values would mask that moment.
                    # In particular, any function ends with a bytecode RETURN, which does not produce a value,
                    # and therefore any pairs of functions at all would "match" at the end.
                    if len(source_vals) > 0 and source_vals == dest_vals:
                        # Record that the values do in fact match
                        self.source_runtime_mapping_to_dest[s_i].value_matches = True
                        self.dest_runtime_mapping_to_source[d_i].value_matches = True
                        # Record the last matching values found so far
                        self.last_matching_val_source = s_i
                        self.last_matching_val_dest = d_i

    # get the python expression corresponding to the op that was traced in self.last_matching_val_source
    def get_last_matching_expression(self):
        correct_runtime_op = self.source_trace.ops_list[self.last_matching_val_source]
        correct_node_id = self.source_op_to_node[correct_runtime_op.op_id]
        node = self.source_index_to_node[correct_node_id]
        return str(node)

    def __str__(self):
        return f'Unit test: {self.test_string}\n' \
               f'test {"finished" if self.run_completed else f"did not finish ({self.run_status})"}\n' \
               f'test {"passed" if self.test_passed else "did not pass"}\n' \
               f'Deviation point (after this op, calculations in the two versions diverge): ' \
               f'{self.last_matching_val_dest} out of {len(self.dest_trace.ops_list)}\n'

    def __lt__(self, other: 'RuntimeComparison'):
        # This RuntimeComparison is "less than" another RuntimeComparison
        # if the source run doesn't get as close to dest run
        # (assumes both dest and unit test was the same in both)
        if other.run_completed and not self.run_completed:
            # our run did not finish, but the other one did finish
            return True
        if self.run_completed and not other.run_completed:
            # our run finished, but the other one did not finish
            return False
        if other.test_passed and not self.test_passed:
            # our run did not pass the unit test, but the other run did
            return True
        if self.test_passed and not other.test_passed:
            # our run passed the unit test, but the other one did not
            return False
        if self.test_passed and other.test_passed:
            # both are passing this test - one is not better than the other, they are equally good
            return False
        # If we are here, then both runs completed but did not pass the unit test.
        # Consider how far each run got compared to dest run:
        # Is the other run's deviation point further along in the dest run than our run's deviation point?
        return self.last_matching_val_dest < other.last_matching_val_dest

    def __eq__(self, other: 'RuntimeComparison'):
        return (other.test_passed and self.test_passed) or \
               (other.run_completed == self.run_completed and
                other.test_passed == self.test_passed and
                other.last_matching_val_dest == self.last_matching_val_dest)

    # Find the first instance in the sequence comparison where the value deviates permanently:
    # The first instance after the deviation point where an executed op is mapped between the source and dest runs,
    # but the value differs.
    # returns tuple of:
    # - index in source trace;
    # - trace data about the source op (inculding the deviating values);
    # - trace data about the dest op (inculding the deviating values)
    def find_first_wrong_value(self):
        for i in range(self.last_matching_val_source, len(self.source_trace.ops_list)):
            op_trace = self.source_trace.ops_list[i]
            op_trace_mapping = self.source_runtime_mapping_to_dest[i]
            if op_trace_mapping.is_mapped:
                # we found a mapped op trace that's after the last matching value,
                # so we can assume the values don't match
                dest_op_trace = self.dest_trace.ops_list[op_trace_mapping.mapped_op_index]
                if op_trace.pushed_values != dest_op_trace.pushed_values:
                    return i
        # went through the whole loop and didn't find anything - must not be any mapped ops
        # after the last one where values matched.
        return None

    def describe_improvement(self, other: 'RuntimeComparison', self_name: str, other_name: str):
        # assuming there *is* an improvement in this RuntimeComparison over other, describe it in words
        if self.run_completed and not other.run_completed:
            return f'The run completed in {self_name}, but did not complete in {other_name} ' \
                   f'({other.run_status}).'
        if self.test_passed and not other.test_passed:
            return f'The test passed in {self_name}, but not in {other_name}.'
        if self.last_matching_val_dest > other.last_matching_val_dest:
            node = self.get_last_matching_expression()
            return f'The following expression evaluated correctly in {self_name}, ' \
                   f'but {other_name} deviated from the expected evaluation path before this expression:' \
                   f' \n {node}'

    # Describe (in human-readable format) whether the effect of running the code
    # got better, worse, or stayed the same from this current version of the comparison to some new version
    def describe_improvement_or_regression(self, new_version: 'RuntimeComparison'):
        if self == new_version:
            return 'The new version of the code performed the same as the old version.'
        elif self < new_version:
            return 'The new version of the code performed better than the old version: \n' + \
                   new_version.describe_improvement(self, 'the new version', 'the old version')
        else:  # self > new_version
            return 'The new version of the code performed worse than the old version: \n' + \
                   self.describe_improvement(new_version, 'the old version', 'the new version')


class Effect(Enum):
    WORSE = 'worse'
    SAME = 'the same'
    MIXED = 'mixed'
    BETTER = 'better'

    def __lt__(self, other):
        # noinspection PyTypeChecker
        member_list = list(self.__class__)
        return member_list.index(self) < member_list.index(other)


# A class to describe the effect of a given fix to some code in the context of a single unit test.
# Given the unit test, code with fix, without fix, and a fully corrected version:
#  - RuntimeComparison to fully corrected versions for each other one
#    - deviation point: where does the before-fix version deviate from the fully corrected version?
#    - compare the comparisons to decide which version is closer to correct
#      (did the fix make things better, worse, or the same?)
#  - Direct trace mapping between versions with and without fix
#    - A "synced" trace generated from this mapping: ops from both traces paired together while preserving order
class FixEffectComparison:
    def __init__( self,
                  before_fix: muast.MutableAst,
                  after_fix: muast.MutableAst,
                  fully_correct: muast.MutableAst,
                  test_string: str):
        self.before_flat_bytecode = map_bytecode.FlatOpsList(before_fix)
        self.after_flat_bytecode = map_bytecode.FlatOpsList(after_fix)

        #### Compare each intermediate version (before and after the fix) to fully corrected version ####
        self.before_to_correct = RuntimeComparison(before_fix, fully_correct, test_string)
        self.after_to_correct = RuntimeComparison(after_fix, fully_correct, test_string)

        before_trace_last_matching = self.before_to_correct.last_matching_val_source
        before_trace_deviation = self.before_to_correct.find_first_wrong_value()
        # before_trace_deviation may be None if there were no explicitly wrong values found after last_matching
        if before_trace_deviation is None:
            # Try showing the last "correct" state in the after-fix version, if it got further than the before-fix one.
            if self.after_to_correct.last_matching_val_dest >= self.before_to_correct.last_matching_val_dest:
                before_trace_deviation = self.before_to_correct.dest_runtime_mapping_to_source[
                    self.after_to_correct.last_matching_val_dest].mapped_op_index

            # are there any ops after last_matching at all?
            elif before_trace_last_matching + 1 < len(self.before_to_correct.source_node_trace):
                # yes - choose the next available op as the deviation point
                before_trace_deviation = before_trace_last_matching + 1
            # if the last_matching op was the last op period, leave before_trace_deviation as None for now.

        #### Directly compare runtime traces of code before and after the fix, to generate synced trace. ####
        self.before_to_after = RuntimeComparison(before_fix, after_fix, test_string)

        # Get the individual (node-based) trace info for before & after versions
        self.before_node_trace = [
            {'node': node, 'values': op.pushed_values,
             'op': self.before_flat_bytecode.id_to_op[op.op_id].simple_repr()
             if op.op_id in self.before_flat_bytecode.id_to_op else '',
             'op_id': op.op_id,
             'actual_op': op.orig_op_string
             }
            for op, node in
            zip(self.before_to_after.source_trace.ops_list, self.before_to_after.source_node_trace)
        ]
        self.after_node_trace = [
            {'node': node, 'values': op.pushed_values,
             'op': self.after_flat_bytecode.id_to_op[op.op_id].simple_repr()
             if op.op_id in self.after_flat_bytecode.id_to_op else '',
             'op_id': op.op_id,
             'actual_op': op.orig_op_string}
            for op, node in
            zip(self.before_to_after.dest_trace.ops_list, self.before_to_after.dest_node_trace)
        ]

        # TODO: should I actually be making this synced trace inside RuntimeComparison
        #  while processing the SequenceMatcher?
        # Combine these individual traces into one synced list of pairs of operations,
        # using the runtime mapping from the RuntimeComparison objects
        # (mapped ops paired together, unmapped paired with a None, maintaining execution order of both traces)
        after_i = 0  # next index in the after trace we expect to see. if we don't see it mapped, we need to insert it.
        self.synced_node_trace = []
        last_matching_i_in_synced = None
        self.deviation_i_in_synced = None
        for before_i, node_op in enumerate(self.before_node_trace):
            #### add ops to synced_trace as needed ###
            if self.before_to_after.source_runtime_mapping_to_dest[before_i].is_mapped:
                mapped_after_i = self.before_to_after.source_runtime_mapping_to_dest[before_i].mapped_op_index
                # if we skipped any ops in the after trace, add them all now:
                while mapped_after_i > after_i:
                    self.synced_node_trace.append({
                        'before': None,
                        'after': self.after_node_trace[after_i],
                        'value_matches': False
                    })
                    after_i += 1
                self.synced_node_trace.append({
                        'before': self.before_node_trace[before_i],
                        'after': self.after_node_trace[after_i],
                        'value_matches': self.before_to_after.source_runtime_mapping_to_dest[before_i].value_matches
                })
                after_i += 1
            else:
                self.synced_node_trace.append({
                    'before': self.before_node_trace[before_i],
                    'after': None,
                    'value_matches': False
                })
            ### check whether we've reached deviation point(s) ###
            if before_trace_last_matching == before_i:
                last_matching_i_in_synced = len(self.synced_node_trace)-1
            if before_trace_deviation is not None and before_trace_deviation == before_i:
                self.deviation_i_in_synced = len(self.synced_node_trace)-1
        # add on rest of after code (that has't been processed because it's after the last matching before op)
        while len(self.after_node_trace) > after_i:
            self.synced_node_trace.append({
                'before': None,
                'after': self.after_node_trace[after_i],
                'value_matches': False
            })
            after_i += 1

        # if we still don't know deviation_i_in_synced, try to put it right after last_matching_i_in_synced
        if last_matching_i_in_synced + 1 < len(self.synced_node_trace):
            self.deviation_i_in_synced = last_matching_i_in_synced + 1
        else:
            # finally, if nothing else worked, just put it as the last actually matching node.
            self.deviation_i_in_synced = last_matching_i_in_synced


# given sets of comparison data of running two versions of code against the same correct version,
# on the same set of unit tests,
# decide whether the new version is a strict improvement over the old version
# also return a suitable index (presumed to be index of an associated unit test)
#   which represents one of the best available comparisons, for use as a runtime illustration.

# Note: For efficiency reasons, this function takes in lists of pre-generated RuntimeComparison objects,
# rather than ASTs to be compared & unit tests like FixEffectComparison above.
# taking in ASTs and unit tests would encapsulate the logic better,
# but then the RuntimeComparison objects for some specific versions of code would be regenerated multiple times
# (and this would involve re-running instrumented code, which is a huge bottleneck)

def compare_comparisons(orig_comps: List[RuntimeComparison], new_comps: List[RuntimeComparison]):
    new_better = []
    new_worse = []
    same = []
    for test_i, (o_c, n_c) in enumerate(zip(orig_comps, new_comps)):
        if o_c < n_c:
            new_better.append(test_i)
        elif n_c < o_c:
            new_worse.append(test_i)
        else:
            same.append(test_i)

    if len(new_better) + len(new_worse) == 0:
        return Effect.SAME, same[0]
    if len(new_worse) == 0:
        return Effect.BETTER, new_better[0]
    if len(new_better) == 0:
        return Effect.WORSE, new_worse[0]
    return Effect.MIXED, new_better[0]
