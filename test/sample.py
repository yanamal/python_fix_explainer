import ast
import logging
import sys

import networkx as nx

import gen_edit_script
import map_asts
import muast
import simplify

# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

student_code = '''
def kthDigit(x, k):
    kthDigLeft = x%(10**(k+1))
    kthDigRight = kthDigLeft//(10**k)
    return kthDigRight
'''

fixed_code1 = '''
def kthDigit(x, k):
    kthDigLeft = x%(10**(k))
    kthDigRight = kthDigLeft//(10**(k-1))
    return kthDigRight
'''

fixed_code2 = '''
def kthDigit(x, k):
    kthDigLeft = x%(10**(k))
    y = kthDigLeft//(10**(k-1))
    return y
'''

kt_digit_unit_tests = [
    "kthDigit(4,1)==4",
    "kthDigit(123,2)==2",
    "kthDigit(5003,3)==0",
    "kthDigit(98,1)==8",
]

### Sample 1: transform python code to MutableAst object; print it out, output AST to file
code_tree = muast.MutableAst(ast.parse(student_code))
print('student code (from AST):')
print(code_tree)  # str conversion attempts to convert AST back to code
code_tree.write_dot_file('sample', '../out/sample.dot')  # draw AST of code in graphviz format (.dot file)

### Sample 2: generate mapping aand edit script between pairs of code (student code and each of two fixed versions)
for fixed_code in [fixed_code1, fixed_code2]:
    print()
    print()
    # get MutableAst objects and mapping between them:
    source_tree, dest_tree, index_mapping = map_asts.get_trees_and_mapping(student_code, fixed_code)
    print('fixed version of code:')
    print(dest_tree)
    # generate edit script and additional metadata which is needed to use the edit script.
    edit_script = gen_edit_script.generate_edit_script(source_tree, dest_tree, index_mapping)

    print('The edit distance is:', edit_script.edit_distance)

    # print out edit script
    print('Edits in edit script:')
    for e in edit_script.edits:
        print(e)

    # Draw the dependency graph between edits in the edit script
    # (usually not actually something you need to think about -
    #  this is data that's used by the simplification step to know which edits belong "together")
    deps = edit_script.get_dependencies()
    with open('../out/dependencies.dot', 'w') as dep_file:
        nx.drawing.nx_pydot.write_dot(deps, dep_file)

    print('code after applying edit script:')
    print(edit_script.apply(source_tree))

    print('Code after simplifying:')
    simplified, deps = simplify.simplify_edit_script(source_tree, kt_digit_unit_tests, edit_script)
    print(simplified.apply(source_tree))

