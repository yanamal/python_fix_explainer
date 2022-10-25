import ast
import logging
import sys

import networkx as nx

import gen_edit_script
import map_asts
import muast
import simplify
import map_bytecode

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

### Sample 2: generate mapping and edit script between pairs of code (student code and each of two fixed versions)
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
    deps = edit_script.dependencies
    with open('../out/dependencies.dot', 'w') as dep_file:
        nx.drawing.nx_pydot.write_dot(deps, dep_file)

    print('code after applying edit script:')
    print(edit_script.apply(source_tree))

    simplified = simplify.simplify_edit_script(source_tree, kt_digit_unit_tests, edit_script)
    print('Code after simplifying:')
    print(simplified.apply(source_tree))


### Sample 3: generate (and use) mapping from bytecode op ids to AST nodes that (probably) produced them

student_tree = muast.MutableAst(ast.parse(student_code))
tree_ops = map_bytecode.FlatOpsList(student_tree)
tree_index_to_node = student_tree.gen_index_to_node()

ops_to_nodes = map_bytecode.gen_op_to_node_mapping(student_tree)

for op in tree_ops:
    print(op.id, op, ops_to_nodes[op.id])
    if ops_to_nodes[op.id]:
        print(tree_index_to_node[ops_to_nodes[op.id]].name)  # name property of the node
        print(tree_index_to_node[ops_to_nodes[op.id]])  # node converted to string (usually code)

