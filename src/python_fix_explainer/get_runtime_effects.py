# Logic for instrumenting and running code to get runtime information:
# list of bytecode ops that ran, and the output (values pushed/popped on the stack) of each one.

# There is no direct way to examine the execution stack to get the values resulting from the various bytecode ops.
# In order to track what values are pushed onto the stack for each op, we need to inject additional instructions
# that pop the just-pushed values from the stack one at a time, and then immediately push them back again
# using an op (LOAD_FAST) whose value we can examine.

import dis
import logging
import re
import sys
import time

from bytecode import Instr, Bytecode, ConcreteInstr
from dataclasses import dataclass
import types
from typing import List, Dict, Tuple

from . import muast
from . import bytecode_metadata


# small dataclass for tracking data about a bytecode op related to instrumenting those ops to see their stack effect
@dataclass
class OpInstrumentationData:
    op_id: tuple  # op id which uniquely identifies this op within its bytecode context
    is_orig_op: bool = True  # is this op from the original code or is it part of the instrumentation?
    num_pushed: int = 0  # number of things pushed onto the stack by this particular op
    num_popped: int = 0  # number of things popped from the stack by this particular op


# set of instructions which should not be instrumented (by popping and pushing stack after the instruction)
do_not_instrument = {'LOAD_METHOD', 'LOAD_BUILD_CLASS', 'SETUP_WITH',
                     # 'CALL_METHOD', 'STORE_FAST',  # TODO: these break request 354
                     'LOAD_CONST', 'LOAD_FAST'}
# LOAD_METHOD and LOAD_BUILD_CLASS seem to break everything, maybe due to scope issues
#   ('variable referenced before assignment' exception for stack-tracking variable, and/or negative stack size)
# the results of LOAD_CONST and LOAD_FAST will already be picked up by the tracer without additional instrumentation.
# TODO: if any jump instructions actually push something on the stack, the instrumentation will only work
#  in the non-jump case (since they are appended after the instruction)


# Instrument a given python (compiled) code object such that the resulting bytecode produces the same result,
# but can also be used in a subsequent custom tracer to extract all of the output values of each original bytecode op.
# This is done by injecting ops after each original op that pushes anything onto the stack
# (except ones explicitly listed in do_not_instrument).
# These injected ops pop the newly-pushed values from the stack, store them in dummy variables,
# then push them back onto the stack so they can be used in the originally intended way.
#
# Since Python code objects are sometimes nested (e.g. function defs are compiled as separate code objects and then
# linked in the original), this function recurses over the child/referenced code objects
# with the help of instrument_child_code.
# the recursive callls keep adding to a single dictionary mapping instrumented instructions to original ones.
#
# The custom tracer subsequently captures the values of each op that stores something in a variable,
# and traces these values back to the original op that put them on the stack.
def instrument_code_obj(py_code_obj: types.CodeType, instrumented_to_orig: Dict[Tuple, OpInstrumentationData] = None):

    # if the optional argument instrumented_to_orig is left as None, initialize a new instrumented_to_orig dictionary
    # this is a map from op_id in instrumented code to metadata about how it relates to some original op
    # in the non-instrumented code
    if instrumented_to_orig is None:
        instrumented_to_orig = {}

    # recursively instrument any child code objects of this code object:
    py_code_obj, instrumented_to_orig = instrument_child_code(py_code_obj, instrumented_to_orig)

    # first, get the push and pop effects of each instruction in the python code objects,
    # as disassembled by dis.get_instructions.
    dis_instructions = dis.get_instructions(py_code_obj)
    dis_instruction_push_pop_effects: List[OpInstrumentationData] = []
    for code_instr in dis_instructions:
        popped, pushed = bytecode_metadata.get_pop_push_stack_effect(code_instr.opcode, code_instr.arg)
        dis_instruction_push_pop_effects.append(
            OpInstrumentationData(
                op_id=(py_code_obj.co_name, code_instr.offset),
                num_popped=popped,
                num_pushed=pushed
            ))

    # now, use the data above to get the push and pop effects of each op/instruction in the editable Bytecode object
    # produced by Bytecode.from_code:
    #
    # the editable Bytecode object loses important information which we need to obtain the stack effect
    # (bytecode argument, as distinct from the human-readable "argval"). so we need to copy the stack effect of the
    # corresponding dis instruction.
    #
    # But using the same code object as input, Bytecode.from_code produces a superset of the instructions produced by
    # dis.get_instructions because Bytecode.from_code creates additional "dummy" instructions such as
    # labels and SetLineno. So we need to insert these dummy ops, along with dummy stack effects,
    # in the appropriate places, as we copy over the stack effects of non-dummy ops
    # from dis_instruction_push_pop_effects

    byte_code_instructions: Bytecode = Bytecode.from_code(py_code_obj)
    byte_code_instruction_push_pop_effects: List[OpInstrumentationData] = []
    for b_instr in byte_code_instructions:
        if isinstance(b_instr, Instr) or isinstance(b_instr, ConcreteInstr):
            # This is an actual instruction - not a label or a SetLineno
            # we can assume that it corresponds to the next instruction in dis_instruction_push_pop_effects
            next_instr = dis_instruction_push_pop_effects.pop(0)  # take the first remaining instruction
        else:
            # this is a dummy op/instruction
            next_instr = OpInstrumentationData(
                op_id=(py_code_obj.co_name, -1),  # dummy op_id
            )
        byte_code_instruction_push_pop_effects.append(next_instr)

    # Iterate through instructions backwards, and inject instrumentation instructions into byte_code_instructions.
    # (iterating backwards allows us to insert instructions without messing up indices of instructions that are
    # not yet instrumented)
    # also create a list of OpInstrumentationData objects, one per expected actual instruction in the
    # resulting code object, which tracks the op id from the original code to which this instruction belongs.
    orig_op_info: List[OpInstrumentationData] = []
    for i in reversed(range(len(byte_code_instruction_push_pop_effects))):
        instr = byte_code_instructions[i]
        push_pop_effect = byte_code_instruction_push_pop_effects[i]

        # generate the instrumentation bytecode for values pushed onto the stack,
        # and inject it into byte_code_instructions:
        to_inject = []
        if push_pop_effect.num_pushed > 0 and (instr.name not in do_not_instrument):
            # print(instr.name, instr, push_pop_effect.num_pushed)
            for j in range(push_pop_effect.num_pushed):
                # for each value newly pushed onto the stack, add instructions to be injected:
                # (prepend instruction) at the beginning, pop it off the stack
                to_inject.insert(0, Instr('STORE_FAST', f'stack_contents_{j}'))
                # (append instruction) at the end, push it back onto the stack
                to_inject.append(Instr('LOAD_FAST', f'stack_contents_{j}'))
            # Inject all instructions after instruction i:
            # print(to_inject)
            for inject_instr in reversed(to_inject):
                byte_code_instructions.insert(i + 1, inject_instr)

        # if we expect this instruction (and the associated injected instrumentation) to show up in the final
        # code object compiled from this bytecode (i.e. this is not a "dummy" instruction like a label),
        # record metadata which will help us reconstruct what values came from what original instructions.
        if isinstance(instr, Instr) or isinstance(instr, ConcreteInstr):
            # noinspection PyListCreation
            op_info_to_prepend: List[OpInstrumentationData] = []
            op_info_to_prepend.append(push_pop_effect)  # original instrumentation data for this op
            for _ in range(len(to_inject)):
                op_info_to_prepend.append(OpInstrumentationData(op_id=push_pop_effect.op_id, is_orig_op=False))
            orig_op_info = op_info_to_prepend + orig_op_info

    # convert the instrumented Bytecode object back to an actual executable python code object
    # print(byte_code_instructions)
    instrumented_py_code: types.CodeType = byte_code_instructions.to_code()
    disassembled_instrumented = dis.get_instructions(instrumented_py_code)
    # the result of dis.get_instructions is a generator, which unlike an actual list of instructions,
    # can only be iterated through once, and then turns into a pumpkin.
    # but we need to iterate through the instructions twice, and we also need access to indices.
    # So explicitly cast it to a proper list of (index, instruction) tuples.
    disassembled_instrumented_instructions = list(enumerate(disassembled_instrumented))

    # during the conversion from Bytecode object to python code object, the Bytecode library inserts "EXTENDED_ARG"
    # instructions before each jump instruction, which we did not account for
    # when creating the list of expected instructions(orig_op_info) from the Bytecode object itself.
    # we correct this now by inserting "dummy" instructions into orig_op_info
    for i, instr in disassembled_instrumented_instructions:
        if instr.opname == 'EXTENDED_ARG':
            # this instruction does not have a corresponding original op id, but thaat doesn't really matter,
            # because this instruction does not affect the data flow we are trying to capture with the instrumentation.
            # we just need to make sure the list of metadata in orig_op_info lines up with the list of instructions in
            # instrumented_py_code.
            orig_op_info.insert(i, OpInstrumentationData(op_id=(None, None), is_orig_op=True))

    # finally, we can use the list of actual instrumented instructions (disassembled_instrumented)
    # and the list of metadata for each expected instrumented instruction (orig_op_info)
    # to create a mapping between the instrumented instructon ids and the original instruction (plus other metadata)
    for i, instr in disassembled_instrumented_instructions:
        instrumented_to_orig[(py_code_obj.co_name, instr.offset)] = orig_op_info[i]

    return instrumented_py_code, instrumented_to_orig


# (recursion helper for instrument_code_obj)
# Given a python code object, find all child/nested code objects, and run instrument_code_obj to instrument each one.
# Then replace each of these children with the instrumented version.
def instrument_child_code(py_code_obj: types.CodeType, instrumented_to_orig: Dict[Tuple, OpInstrumentationData]):

    # cast code object to ConcreteBytecode in order to be able to change the child code objects
    py_concrete_bytecode = Bytecode.from_code(py_code_obj).to_concrete_bytecode()

    for instr in dis.get_instructions(py_code_obj):
        if isinstance(instr.argval, types.CodeType):
            # instrument this child code object
            instr_child_obj, instrumented_to_orig = instrument_code_obj(instr.argval, instrumented_to_orig)
            # replace reference to child code object with the instrumented version
            py_concrete_bytecode.consts[instr.arg] = instr_child_obj

    # cast back to a regular python code object that can be executed
    with_instrumented_children = py_concrete_bytecode.to_code()
    return with_instrumented_children, instrumented_to_orig


# dataclas representing an op/instructions whose execution was traced.
@dataclass
class TracedOp:
    op_id: tuple  # the unique-within-code-object id of the operation
    pushed_values: list  # list of values which were pushed onto the stack by this op when it was executed this time.
    orig_op_string: str = ''  # string representation of op that actually was traced


# a simple class wrapper for tracking and interpreting instrumented code
class Instrumented_Bytecode:
    def __init__(self, code_str):
        # print('instrumenting code:')
        # print(code_str)
        self.original_code_obj = compile(code_str, '<string>', 'exec')
        self.instrumented_code_obj, self.instrumented_to_orig = instrument_code_obj(self.original_code_obj)
        # self.runtime_ops_list will keep the list of ops that were executed(and values pushed on stack)
        # during most recent run of the instrumented code (through a specialized code tracer)
        self.runtime_ops_list: List[TracedOp] = []

    # add a trace of an op being executed (leave pushed values blank for now)
    def add_op_trace(self, op_id: tuple, orig_op):
        self.runtime_ops_list.append(TracedOp(op_id=op_id, pushed_values=[], orig_op_string=orig_op))

    # add a value that was pushed onto the stack by the last traced op
    def trace_pushed_value(self, value):
        # if type(value) in bytecode_metadata.unpickleable:
        #     # if the value is unpickleable, record a dummy string instead
        #     # value = '<unpickleable object>'

        # noinspection PyBroadException
        try:
            # a circular dependency edge case happens when the __str__ method in a student-defined class is overridden,
            # and the instrumentation tries to get the value of "self" during __init__,
            # when not everything has been initialized yet (including variables that are being used in __str__)
            # In that case, and any other cases where for some reason conversion to a string fails,
            # we try to just document the type of the value.
            value = str(value)
        except Exception as e:
            value = str(type(value))
        value = re.sub(r'<(.*) at .*>', r'<\1>', value)  # cut off things like memory addresses
        last_op = self.runtime_ops_list[-1]
        last_op.pushed_values.append(value)
        # TODO: always converting to string before comparisons could miss subtleties like string/number confusion
        #  maybe convert if it's not a primitive?..


# make and return a tracer function which can be passed to sys.settrace to trace and interpret the ops
# in a particular Instrumented_Bytecode object, mapping the instrumented ops
# back onto the original ops of the code pre-instrumentation.
# record the resulting list of executed ops in tinstr_code.runtime_ops_list.
# If the number of (original) ops exceeds max_ops, assume there is an infinite loop and abort.
# TODO: pass in a suitable per-problem max_ops,
#  based on a multiple of the number of ops that a correct solution tends to take.
def make_ops_tracer(instr_code: Instrumented_Bytecode, max_ops=20000):

    def trace_ops(frame, event, arg):
        frame.f_trace_opcodes = True  # yes, trace execution of each bytecode operation
        if frame.f_code.co_filename != '<string>':
            # we are going into code that wasn't part of the string (problem solution + unit test)
            # we don't care about tracing what happens there.
            return

        if event == 'opcode':
            # we are processing an event of a bytecode op being executed
            this_op_id = (frame.f_code.co_name, frame.f_lasti)
            this_opcode = frame.f_code.co_code[frame.f_lasti]
            this_oparg = frame.f_code.co_code[frame.f_lasti + 1]  # the next byte after opcode is the arg

            if this_op_id in instr_code.instrumented_to_orig:
                # should always be true, unless we added some code which was not instrumented
                orig_op_info = instr_code.instrumented_to_orig[this_op_id]
                if orig_op_info.is_orig_op:
                    # This op in the instrumented code is an original op present in the non-instrumented code
                    # (rather than an op that's part of the insrumentation for some original op)

                    # add this op to the traced ops
                    instr_code.add_op_trace(orig_op_info.op_id,
                                            str(f'{orig_op_info.op_id} {dis.opname[this_opcode]} {this_oparg}'))
                    if len(instr_code.runtime_ops_list) > max_ops:
                        raise muast.CodeTimeoutException()

                    # also, if this is a LOAD_CONST, get the value it pushed onto the stack and add it to the trace
                    if this_opcode == dis.opmap['LOAD_CONST']:
                        # for LOAD_CONST, the op arg is the index of the constant in f_code.co_consts
                        # use it to get the value of the constant (which is the value loaded and pushed onto the stack)
                        const_contents = frame.f_code.co_consts[this_oparg]
                        instr_code.trace_pushed_value(const_contents)

            if this_opcode == dis.opmap['LOAD_FAST']:
                # This operation pushes the value of a local variable onto the stack.
                # It's either an actual variable-load, or part of the instrumentation
                # for inspecting what has been previously pushed onto the stack  in the original op.
                # so, record the loaded variable as one of the pushed values.
                var_name = frame.f_code.co_varnames[this_oparg]
                var_contents = frame.f_locals[var_name]
                instr_code.trace_pushed_value(var_contents)

        return trace_ops  # keep tracing with the same function

    return trace_ops  # return the tracing function we created


@dataclass
class TracedRunResult:
    eval_result: bool  # did the unit test pass?
    ops_list: List[TracedOp]  # sequence of ops that were executed and values they pushed on the stack
    run_outcome: str  # e.g. "completed" or error message if there was an error


# Instrument and run the code in the string, then evaluate the expression in the test string
# Typically, the test string checks that the actual result of executing some function matches the expected result,
# e.g. "secondHalf([1,2,3,4]) == [3,4]"
def run_test_timed(code: str, test_string: str, prepend_code: str = '', append_code: str = ''):
    # instrument studend code string for tracing.
    # (must do it in this function, because we can't pickle code objects for passing through multiprocessing)
    # TODO: we no longer need to do it here, because no longer using multiprocessing.
    code = code + '\n' + test_string
    instr_code = Instrumented_Bytecode(code)

    try:
        # TODO: Try explicitly compiling student code both here and in FlatOpsList, with a matching filename option.
        #  Then use filename as another part of op_id, to avoid spuriously matching <module> ops with same line number
        #  that came from different sources.
        #  (in that case, go back to only instrumenting - and compiling in this way - student code
        #  and eval()ing the unit test after the student code runs)
        # try running and tracing the code together with the unit test
        old_trace = sys.gettrace()  # TODO: do this outside of try, and use old_trace in Exception handling?

        exec(prepend_code, globals())

        sys.settrace(make_ops_tracer(instr_code))
        exec(instr_code.instrumented_code_obj, globals())
        sys.settrace(old_trace)

        exec(append_code, globals())
        unit_test_result = eval(test_string)  # now actually record the unit test result by re-running the unit test

        return TracedRunResult(
            eval_result=unit_test_result,
            ops_list=instr_code.runtime_ops_list,
            run_outcome='completed')

    except Exception as e:  # noqa
        # running the code threw an exception, most likely due to bugs in student code
        sys.settrace(None)
        exception_text = str(e)
        # modify the ops list to capture the exception - in this case, the op that caused the exception
        # will be the last op in the ops_list, and we add the exception as its value.
        if len(instr_code.runtime_ops_list) > 0:
            # sometimes there are no ops yet (e.g. when the exception is actually thrown by the "prefix" code.
            instr_code.runtime_ops_list[-1].pushed_values = [f'Exception: {exception_text}']
        return TracedRunResult(eval_result=False, ops_list=instr_code.runtime_ops_list, run_outcome=exception_text)


def run_test(code: str, test_string: str, prepend_code: str = '', append_code: str = ''):
    start_run = time.time()
    result = run_test_timed(code, test_string, prepend_code=prepend_code, append_code=append_code)
    logging.info(f'Trace consisted of {len(result.ops_list)} non-instrumentation ops')
    logging.info(f'Running instrumented unit test took {time.time() - start_run} seconds')
    return result
