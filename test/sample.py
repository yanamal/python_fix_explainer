import manip_ast
import ast

code = '''
def kthDigit(x, k):
	kthDigLeft = x%(10**(k+1))
	kthDigRight = kthDigLeft//(10**k)
	return kthDigRight
'''

code_tree = manip_ast.ManipulableAst(ast.parse(code))
print(code_tree)

code_tree.write_dot_file('sample', '../out/sample.dot')
