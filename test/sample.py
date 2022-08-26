import manip_ast
import map_asts
import edit_script
import ast

code = '''
def kthDigit(x, k):
	kthDigLeft = x%(10**(k+1))
	kthDigRight = kthDigLeft//(10**k)
	return kthDigRight
'''

fixed = '''
def kthDigit(x, k):
	kthDigLeft = x%(10**(k))
	kthDigRight = kthDigLeft//(10**(k-1))
	return kthDigRight
'''

code_tree = manip_ast.ManipulableAst(ast.parse(code))
print(code_tree)

code_tree.write_dot_file('sample', '../out/sample.dot')

source_tree, dest_tree, index_mapping = map_asts.get_trees_and_mapping(code, fixed)

edit_script, additional_nodes, var_renames = edit_script.generate_edit_script(source_tree, dest_tree, index_mapping)

for e in edit_script:
    print(e)



