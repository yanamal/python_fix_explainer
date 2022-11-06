# Additional metadata about Python bytecode (specifically, 3.7 CPython bytecode)
# which is needed in order to make sense of what is happening in the bytecode,
# but is not generally available as part of standard libraries (as far as I can tell)

import dis
import types


# map of endings from LOAD_*, STORE_* ops to which data structures contain the type of variable being loaded/stored
var_tuples = {
    'NAME': 'co_names',
    'ATTR': 'co_names',
    'GLOBAL': 'co_names',
    'DEREF': 'cell_free',
    'CLASSDEREF': 'cell_free',
    'CLOSURE': 'cell_free',
    'FAST': 'co_varnames',
}

# types of possible (resolved) bytecode values which are not pickleable;
# we probably don't care about them when instrumenting tracing
unpickleable = {types.CodeType, types.MethodType, types.ModuleType, types.FunctionType}


# functions encoding the stack effects (how many things pushed and popped from the stack) of bytecode ops
def get_constant_pop_stack_effect(opname):
    # get how many things are popped off the stack (examined) with a given op

    # effects of individual ops
    # (based on https://docs.python.org/3/library/dis.html#python-bytecode-instructions )

    # Not implemented (too complicated/unclear, probably won't need it):
    # END_ASYNC_FOR BEFORE_ASYNC_WITH WITH_EXCEPT_START SETUP_WITH
    # TODO: BUILD_TUPLE, BUILD_LIST, BUILD_SET, BUILD_MAP, BUILD_CONST_KEY_MAP, BUILD_STRING, BUILD_SLICE (uses arg)
    #  RAISE_VARARGS CALL_FUNCTION CALL_FUNCTION_KW CALL_FUNCTION_EX
    #  MAKE_FUNCTION(?)
    individual_effects = {
        'POP_TOP': 1,
        'ROT_TWO': 2,
        'ROT_THREE': 3,
        'ROT_FOUR': 4,
        'DUP_TOP': 1,
        'DUP_TOP_TWO': 2,
        'GET_ITER': 1,
        'GET_YIELD_FROM_ITER': 1,
        'STORE_SUBSCR': 2,
        'DELETE_SUBSCR': 2,
        'GET_AWAITABLE': 1,
        'GET_AITER': 1,
        'GET_ANEXT': 1,
        'PRINT_EXPR': 1,
        'SET_ADD': 2,
        'LIST_APPEND': 2,
        'MAP_ADD': 2,
        'RETURN_VALUE': 1,
        'YIELD_VALUE': 1,
        'YIELD_FROM': 1,
        'IMPORT_STAR': 1,  # Based on empirical observation (not documented)
        'STORE_NAME': 1,
        'STORE_GLOBAL': 1,
        'LOAD_BUILD_CLASS': 1,
        'LOAD_ASSERTION_ERROR': 1,
        'UNPACK_SEQUENCE': 1,
        'UNPACK_EX': 1,
        'STORE_ATTR': 2,
        'DELETE_ATTR': 1,
        'LIST_TO_TUPLE': 1,
        'LIST_EXTEND': 2,
        'SET_UPDATE': 2,
        'DICT_UPDATE': 2,
        'DICT_MERGE': 2,
        'LOAD_ATTR': 1,
        'COMPARE_OP': 2,  # Based on empirical observation (not documented)
        'IS_OP': 2,  # Based on empirical observation (not documented)
        'CONTAINS_OP': 2,  # Based on empirical observation (not documented)
        'IMPORT_NAME': 2,
        'IMPORT_FROM': 1,
        'POP_JUMP_IF_TRUE': 1,
        'POP_JUMP_IF_FALSE': 1,
        'JUMP_IF_NOT_EXC_MATCH': 2,
        'JUMP_IF_TRUE_OR_POP': 1,  # count looking at TOS as "pop" (and then push back)
        'JUMP_IF_FALSE_OR_POP': 1,
        'FOR_ITER': 1,
        'STORE_FAST': 1,
        'STORE_DEREF': 1,
        'LOAD_METHOD': 1,
        'MAKE_FUNCTION': 2,  # Based on empirical observation (not well-documented)
    }
    if opname.startswith('UNARY_'):
        return 1
    elif opname.startswith('BINARY_'):
        return 2
    elif opname.startswith('INPLACE_'):
        # binary in-place operations (they still look at two things on top of the stack)
        return 2
    elif opname in individual_effects:
        return individual_effects[opname]
    else:
        return 0  # not listed; must not use anything from the stack


def get_pop_push_stack_effect(opcode, oparg=None, jump=None):
    # Get a pair of numbers representing the stack effect for data flow purposes:
    # - number of items popped/accessed
    # - number of items pushed/touched
    opname = dis.opname[opcode]
    sum_effect = dis.stack_effect(opcode, oparg)
    # TODO: figure out how to do jump argument (maybe doesn't exist in < 3.8?.. but tooltips have it)

    # these ops pop a variable number of items off the stack, but always push one on:
    push_for_variable_pop_effect = {op: 1 for op in [
        'BUILD_TUPLE', 'BUILD_LIST', 'BUILD_SET', 'BUILD_MAP', 'BUILD_CONST_KEY_MAP', 'BUILD_STRING', 'BUILD_SLICE',
        'CALL_FUNCTION', 'CALL_FUNCTION_KW', 'CALL_FUNCTION_EX', 'CALL_METHOD', 'MAKE_FUNCTION']}
    # RAISE_VARARGS pops variable number of items, pushes 0
    push_for_variable_pop_effect['RAISE_VARARGS'] = 0

    if opname in push_for_variable_pop_effect:
        push_effect = push_for_variable_pop_effect[opname]
        return -sum_effect + push_effect, push_effect

    # assume it's a constant pop effect
    pop_effect = get_constant_pop_stack_effect(opname)
    return pop_effect, sum_effect + pop_effect


