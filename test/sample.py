import manip_ast
import map_asts
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

map_asts.get_trees_and_mapping(code, fixed)
