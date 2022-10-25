# logic for creating a mapping between:
# (a) the nodes in a MutableAst representation of some code; and
# (b) the instructions in the bytecode representation of the same code

# We achieve this in a somewhat hacky way,
# by deleting nodes from the AST one at a time and seeing what changes in the bytecode.
import ast
import copy
import difflib
import dis
import types
from collections import deque, defaultdict

import muast


# types of possible (resolved) bytecode values which are not pickleable;
# we probably don't care about them when instrumenting tracing
unpickleable = {types.CodeType, types.MethodType, types.ModuleType, types.FunctionType}


# A class for easier tracking of individual bytecode instructions
class Opdata:
    def __init__(self, instr, code_obj):
        self.instr = instr
        self.code_obj = code_obj
        self.co_name = code_obj.co_name
        self.offset = instr.offset
        self.opcode = instr.opcode
        self.arg = instr.arg

    # The op id is a unique and consistent identifier of this op
    # within the context of the specific code that generated the bytecode
    # (so regenerating bytecode from the same code will produce ops that have the same ids)
    @property
    def id(self):
        return self.co_name, self.offset

    def __str__(self):
        return f'{self.co_name}\t{self.offset} {self.instr.opname} {self.instr.argval}'


# A class which represents code as a flat list of bytecode ops
# (rather than the nested Python bytecode where a single op sometimes points to a whole subprogram)
# This is used to compare versions bytecode resulting from slighly different ASTs,
# To capture how changing AST nodes affects the resulting bytecode.
class FlatOpsList:
    def __init__(self, code_tree: muast.MutableAst):
        code_str = code_tree.to_compileable_str()
        # default(blank) ops
        self.ops = []
        self.id_to_op = {}

        # convert a string of code into a flat list of bytecode ops
        try:
            root_code_obj = compile(code_str, '', 'exec')
        except SyntaxError:
            # This code does not compile; so there is no associated bytecode (fields are left blank)
            return

        # things like function definitions are compiled into their own code objects;
        # keep a queue of all code objects to process
        all_code_objs = deque([root_code_obj])

        while len(all_code_objs) > 0:
            curr_code_obj = all_code_objs.popleft()
            for instr in dis.get_instructions(curr_code_obj):
                if isinstance(instr.argval, types.CodeType):
                    all_code_objs.append(instr.argval)
                op_data = Opdata(instr, curr_code_obj)
                self.ops.append(op_data)
                self.id_to_op[op_data.id] = op_data

    def __getitem__(self, item):
        return self.ops[item]

    def __iter__(self):
        return iter(self.ops)

    def __len__(self):
        return len(self.ops)

    def __str__(self):
        return '\n'.join([str(op) for op in self.ops])

    def get_by_id(self, op_id):
        return self.id_to_op[op_id]

    def has_op_id(self, op_id):
        return op_id in self.id_to_op


def compare_op_lists(op_list1: FlatOpsList, op_list2: FlatOpsList, changed_node_id: str):
    # Compare two FlatOpsList objects to identify how the ops changed from the first to the second,
    # and record these changes as changes associated with changed_node_id:
    # (1) find longest commons subsequence of two op lists (based solely on the opcodes, not argvals)
    # (2) find ops that changed: ops that are either unmapped in the LCS, or ones where the argval changed.
    # (3) record which changed_node_id these chaanges are associated with (by modifying in place op_list*)
    # returns mapping between the two ops list (based on unique op identifier of co_name and offset),
    # and the list of ops that changed

    ops1 = [o.opcode for o in op_list1]
    ops2 = [o.opcode for o in op_list2]
    s = difflib.SequenceMatcher(None, ops1, ops2, autojunk=False)
    op_map = {}
    mapped_ops2 = set()
    changed_ops = set()
    matching_blocks = s.get_matching_blocks()
    for start1, start2, n in matching_blocks:
        for off_i in range(n):
            o1: Opdata = op_list1[start1 + off_i]
            o2: Opdata = op_list2[start2 + off_i]
            op_map[o1.id] = o2.id
            mapped_ops2.add(o2.id)
            if (type(o1.instr.argval) not in unpickleable) \
                    and o1.arg not in dis.hasjrel \
                    and o1.arg not in dis.hasjabs \
                    and o1.instr.argval != o2.instr.argval:
                # this op should also count as edited, because argval changed.
                # (for example, the changed AST node represented a constant or literal that was the argument of an op)
                # Don't count arg as having changed for the following kinds of ops:
                #  - anything when argval is a pointer to a complex (unpickleable) object, e.g. code object
                #  - ops whose args are addresses/offsets
                changed_ops.add(o1.id)

    # find all ops that were unmapped, and record them as edited in this edit:
    for o1 in op_list1:
        if o1.id not in op_map:
            changed_ops.add(o1.id)

    return op_map, changed_ops


# the debug_mapping flag maps ops to the code generated by the subtree at the node, instead of to the node id
def gen_op_to_node_mapping(source_tree: muast.MutableAst, debug_mapping=False):
    tree_copy = copy.deepcopy(source_tree)

    # tree_copy.write_dot_file('map_test', 'map_test.dot')

    deletion_order = list(muast.postorder(tree_copy))  # static list of the order in which we want to delete the nodes
    index_to_node = tree_copy.gen_index_to_node()

    # for the debug_mapping option, record the code strings that each node compiles to
    # (before we start deleting nodes and mess up the code)
    index_to_node_str = {}
    if debug_mapping:
        for index in index_to_node:
            index_to_node_str[index] = index_to_node[index].to_compileable_str()

    # TODO: if compilation error, fail the same way as runtime error (earlier than this)?
    orig_ops = FlatOpsList(tree_copy)

    # current ops list (for a given iteration of deleting a node and converting to ops list) - start with orig_ops
    curr_ops = orig_ops
    # map of op ids: original to prev_ops, propagated forward each time through the loop
    orig_op_to_curr_op = {o.id: o.id for o in orig_ops}
    # map of op id (for original op list from original tree) to node in the original tree
    # (nodes normally represented by node ids; except if debug_mapping is on)
    orig_op_to_node = defaultdict(lambda: None)

    for del_node in deletion_order:

        # remove node
        if not del_node.parent:
            # no parent - probably reached root
            # no need to remove 'module' root node, it doesn't have a code representation anyway
            continue
        del_node.parent.remove_child(del_node)

        # TODO: detect if code string is the same as previous time around (node removal makes no visible change)

        # recalculate ops
        next_ops = FlatOpsList(tree_copy)

        # set the annotation that the changed ops will map to (normally node id; when debugging, node code)
        annotation = del_node.index
        if debug_mapping:
            annotation = index_to_node_str[del_node.index]

        # compare new and old ops list, and modify them in place
        # by annotating with index of the deleted node
        curr_to_next_op_map, changed_ops = compare_op_lists(curr_ops, next_ops, annotation)

        # Process each original op that still has a mapping to some op in the current oplist
        for orig_op_id in orig_op_to_curr_op:

            # (1) map changed ops back to the matching ops in the original oplist,
            # and map that original op to the node that changed it (i.e. the current node we are working with)
            curr_op_id = orig_op_to_curr_op[orig_op_id]
            # if this op changed from curr_ops to next_ops, AND it hasn't already been mapped to a node
            # that changed it previously, then map the corresponding original op to this current node.
            if curr_op_id in changed_ops and orig_op_id not in orig_op_to_node:
                orig_op_to_node[orig_op_id] = annotation

            # (2) propagate orig_to_curr forward to point to next_ops (soon to become curr_ops)
            if orig_op_to_curr_op[orig_op_id] in curr_to_next_op_map:
                orig_op_to_curr_op[orig_op_id] = curr_to_next_op_map[orig_op_to_curr_op[orig_op_id]]

        curr_ops = next_ops

    # TODO: it looks like "return" statements in the bytecode do not get correctly mapped to the return AST node
    #  (if one exists)
    #  because the bytecode returns something no matter what. Instead, they get mapped to the function definition, since
    #  they disappear when the function definition node gets deleted.

    # TODO: where do the function parameter definitions get mapped to? do they not have a direct runtime effect?..

    # TODO: what about function names? they seem to get mapped to the FunctionDef node?..
    #   This is because they are "absorbed" into the FunctionDef node by the MutableAst simplification logic.
    #   This normally works well because in the code, there is not much distinction between the literal value and the
    #   ast node that represents it (e.g. Num, Load Identifier, etc.)
    #   maybe certain literals should be whitelisted to remain their own nodes?..

    # TODO: return just the mapping between AST nodes and bytecode instruction ids
    return orig_op_to_node


# Temporary testing code:
code = """
def kthDigit(x, k):
    kthDigLeft = x%(10**(k+1))
    kthDigRight = kthDigLeft//(10**k)
    return kthDigRight
"""

tree = muast.MutableAst(ast.parse(code))
tree_ops = FlatOpsList(tree)
tree_index_to_node = tree.gen_index_to_node()

ops_to_nodes = gen_op_to_node_mapping(tree, debug_mapping=False)

for op in tree_ops:
    print(op.id, op, ops_to_nodes[op.id])
    if ops_to_nodes[op.id]:
        print(tree_index_to_node[ops_to_nodes[op.id]].name)

