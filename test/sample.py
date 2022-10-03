import muast
import map_asts
import gen_edit_script
import ast

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
    edit_script, additional_nodes, var_renames = \
        gen_edit_script.generate_edit_script(source_tree, dest_tree, index_mapping)

    print('The edit distance is:', len(edit_script))

    # print out edit script
    print('Edit script:')
    for e in edit_script:
        print(e)

