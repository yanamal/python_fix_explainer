# Class and helper functions for manipulable AST entities: wrapper around Python ASTs so that they can be:
#  1. manipulated using a specific set of edits
#  2. translated to a human-readable form (usally Python code, or failing that, node description)


# Several types of strings can be generated from a given tree:
# - one emphasizes being able to generate bytecode from resulting code,
# - the other emphasizes literally representing the original code (e.g. no inserting dummy values).

# The native python ast is updated during the manipulations to match the MutableAst object.

import copy
import string
import random
from uuid import uuid4
from typing import List, Dict
import astor.code_gen as cg
import astor
import ast
import multiprocessing
from xml.etree.ElementTree import Element, SubElement, ElementTree


# Utility - canonicalize code
def canonical_code(code_str):
    return astor.to_source(ast.parse(code_str))


class ForbiddenEditException(Exception):
    pass


# TODO: try to prevent (mid-change-state) modification of "x in y" to "x in z is not y"?

# Special cases where a child node's default (blank) value is ambiguous, and depends on parent type
# (so deletion breaks the AST if wrong type is used)
extra_special_cases = {
    'If.orelse': [],
    'IfExp.orelse': None
}


# extend astor's source-from-ast generator to fix a couple of problems with manipulating AST
set_precedence = cg.set_precedence


class CustomSourceGen(astor.SourceGenerator):
    # insert the string 'None' whenever the traversal tries to visit
    # a child node that is None.
    # ordinarily, this would throw an exception; but sometimes move operations leave an unexpected "blank" child,
    # and we would still like to be able to reason about the intermediate tree
    # (e.g. can I skip this deletion but not this move?)
    # TODO: can there be any problems from this?
    def visit_NoneType(self, node):  # noqa
        self.write('None')

    # when visiting Compare operators, fill in missing comparators as needed
    def visit_Compare(self, node):
        if len(node.comparators) < len(node.ops):
            node = copy.deepcopy(node)  # Don't irreversibly change the actual code
            # (in a way that won't be undone by subsequent edits, i.e. by appending to lists)

            node.comparators += [ast.NameConstant(value=None)] * (len(node.ops) - len(node.comparators))
        super(CustomSourceGen, self).visit_Compare(node)

    # fill in empty values for keywords:
    def visit_Call(self, node, len=len):
        for keyword in node.keywords:
            if not keyword.value:
                keyword.value = ast.Constant(value=None, kind=None)
        super(CustomSourceGen, self).visit_Call(node, len)


class RuntimeSourceGen(CustomSourceGen):
    # A custom astor source generator for purposes of bytecode comparison between edits:
    # Fill in blanks which create malformed code and prevent generating bytecode and/or running the code,
    # in such a way that subsequent changes will modify the injected dummy values

    @staticmethod
    def gen_dummy_name():
        # TODO: automatically generate dummy Name replacement when removing a Name node?..
        # generate a (probably unique) dummy name to take the place of some missing token
        return 'dummy_'+''.join(random.choice(string.ascii_letters) for _ in range(8))

    def body(self, statements):
        self.indentation += 1
        for s in statements:
            self.newline()
            self.write(s)
        if len(statements) == 0:
            self.newline()
            self.write('pass')
        self.indentation -= 1

    def visit_BinOp(self, node):
        if not node.left:
            node.left = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        if not node.right:
            node.right = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        super(RuntimeSourceGen, self).visit_BinOp(node)

    def visit_UnaryOp(self, node):
        if not node.operand:
            node.operand = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        super(RuntimeSourceGen, self).visit_UnaryOp(node)

    def visit_Assign(self, node):
        cg.set_precedence(node, node.value, *node.targets)
        self.newline(node)
        for target in node.targets:
            self.write(target, ' = ')
        if len(node.targets) == 0:
            self.write(f'{self.gen_dummy_name()} = ')
        self.visit(node.value)

    def visit_AugAssign(self, node):
        cg.set_precedence(node, node.value, node.target)

        target = node.target or self.gen_dummy_name()
        value = node.value or 'None'

        self.statement(node, target, astor.get_op_symbol(node.op, ' %s= '),
                       value)

    def visit_Attribute(self, node):
        v = node.value or self.gen_dummy_name()
        self.write(v, '.', node.attr)

    def visit_If(self, node):
        if not node.test:
            node.test = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())

        # TODO: why doesn't my self.body work? (or does it? try w/o below)
        if len(node.body) <= 0:
            node.body = ['pass']

        cg.set_precedence(node, node.test)
        self.statement(node, 'if ', node.test, ':')
        self.body(node.body)
        while True:
            else_ = node.orelse
            if len(else_) == 1 and isinstance(else_[0], ast.If):
                node = else_[0]
                if not node.test:
                    node.test = ast.NameConstant(value=False)
                cg.set_precedence(node, node.test)
                self.write(self.newline, 'elif ', node.test, ':')
                self.body(node.body)
            else:
                self.else_body(else_)
                break

    def visit_IfExp(self, node):
        if not node.body:
            node.body = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        if not node.orelse:
            node.orelse = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        if not node.test:
            node.test = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        super(RuntimeSourceGen, self).visit_IfExp(node)

    def visit_Expr(self, node):
        if not node.value:
            node.value = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        super(RuntimeSourceGen, self).visit_Expr(node)

    def visit_While(self, node):
        if not node.test:
            node.test = ast.NameConstant(value=None)
        super(RuntimeSourceGen, self).visit_While(node)

    def visit_For(self, node, is_async=False):  # TODO: deal with async case?..
        if not node.target:
            node.target = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        if not node.iter:
            node.iter = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        super(RuntimeSourceGen, self).visit_For(node)

    def visit_Compare(self, node):
        if not node.left:
            node.left = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        # If there are not enough ops to meaningfully parse the comparison,
        # insert arbitrarily-chosen op that will parse (is not - hopefully the least often used op in student code)
        if len(node.ops) == 0 or len(node.ops) < len(node.comparators):

            node = copy.deepcopy(node)  # Don't irreversibly change the actual code
            # (in a way that won't be undone by subsequent edits, i.e. by appending to lists)

            how_many_more_ops = max(1, len(node.comparators) - len(node.ops))
            node.ops += [ast.IsNot()] * how_many_more_ops
        super(RuntimeSourceGen, self).visit_Compare(node)

    def visit_Import(self, node):
        self.statement(node, 'import ')
        if node.names:
            self.comma_list(node.names)
        else:
            self.write(self.gen_dummy_name())

    def visit_Subscript(self, node):
        cg.set_precedence(node, node.slice)
        if not node.value:
            node.value = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        if node.slice:
            self.write(node.value, '[', node.slice, ']')
        else:
            self.write(node.value)

    def visit_BoolOp(self, node):
        if len(node.values) == 0:
            node.values.append(ast.Name(id=self.gen_dummy_name(), ctx=ast.Load()))
        super(RuntimeSourceGen, self).visit_BoolOp(node)

    def visit_comprehension(self, node):
        if not node.target:
            node.target = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        if not node.iter:
            node.iter = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        super(RuntimeSourceGen, self).visit_comprehension(node)

    def visit_arguments(self, node):
        if node:
            super(RuntimeSourceGen, self).visit_arguments(node)

    def visit_Call(self, node):
        if not node.func:
            node.func = ast.Name(id=self.gen_dummy_name(), ctx=ast.Load())
        super(RuntimeSourceGen, self).visit_Call(node)


def get_node_name(node):
    if isinstance(node, list):
        # this is a list of ast.Node objects
        return 'NodeList'
    elif hasattr(node, '_fields'):
        # this is an ast.Node object - return its type
        return type(node).__name__
    else:
        # this must be a literal, e.g. variable name, const value, etc.
        return f'{str(node)}'


def is_literal(node):
    return (not isinstance(node, list)) and (not hasattr(node, '_fields'))


def ast_children(node):
    if isinstance(node, list):
        # this is a list of nodes. We return an equivalent dictionary, to keep the data type the same
        # regardless of child type
        children = {}
        for i, n in enumerate(node):
            children[i] = n
        return children

    elif hasattr(node, '_fields'):
        # this is a Node object. It has all its children listed in _fields. create a plain dictionary mapping child
        # attribute name to child object (which may be a literal, an ast.Node object, or a list of ast.Nodes)
        children = {}
        for child_name in node._fields:
            children[child_name] = getattr(node, child_name)
        return children
    else:
        # this node itself is a literal; it has no children.
        return {}


def remove_ast_child(ast_node, child_key):
    # remove child from ast node and replace with appropriate empty placeholder
    child_node = getattr(ast_node, child_key)

    placeholder = None
    if isinstance(child_node, list):
        # guess that this is a list-type parameter, and  the empty version is []
        placeholder = []

    # TODO: unit test?
    # if this is a special ambiguous case, make sure we use the right one,
    # based both on the node type and field name
    attr_string = f'{type(ast_node).__name__}.{child_key}'
    if attr_string in extra_special_cases:
        placeholder = extra_special_cases[attr_string]

    setattr(ast_node, child_key, placeholder)


class MutableAst:
    # TODO: separate out List subclass instead of using isList?.. probably would have to have abstract class then?
    #  would it even be possible to have an agnostic constructor?.. does it matter?..
    #  (probably could make AbstractMutableAst, etc.
    #  and MutableAst is just a wrapper which decides whether it's a list)
    # A wrapper class for python ASTs which:
    # (1) plays well with APTED library,
    # (2) generates unique (within tree) indices for nodes to make tracking nodes easier,
    # (3) makes structural tweaks to make AST comparison give more sensible results,
    # (4) allows for changing the AST (and makes corresponding changes to the underlying Python AST)
    # (5) allows for execution of a manipulated AST
    def __init__(self, py_ast, node_index=None, shallow=False, name=None, assign_depth=None):
        # a passed-in chunk of python AST could be one of three things:
        # (1) an actual AST node
        # (2) a 'leaf' literal - e.g. string representing the name of the variable
        # (3) a list of nodes which is a child to some AST node (e.g. list of statements in a block of code)

        # remember if this node is actually a list, because certain things only make sense to do with lists (e.g.
        # insert into 'middle of' list)
        self.isList = isinstance(py_ast, list)
        self.isLiteral = (not self.isList) and (not hasattr(py_ast, '_fields'))

        if self.isList:
            self.nodeType = 'NodeList'
        elif self.isLiteral:
            self.nodeType = 'Literal'
        else:
            self.nodeType = type(py_ast).__name__

        # parent info - will be overwritten by parent as necessary after creation:
        self.parent = None
        self.key_in_parent = None

        self.name = get_node_name(py_ast)
        self.ast = py_ast  # the original ast node
        ignore_children = self.simplify_node()

        # the depth of this node within an assignment (if it is indeed within an assignment)
        if self.name == 'Assign':
            assign_depth = 0
        self.assign_depth = assign_depth

        next_assign_depth = (assign_depth + 1) if (assign_depth is not None) else None

        self.children_dict: Dict[str, 'MutableAst'] = {}
        if not shallow:
            for c_key, c_ast in ast_children(py_ast).items():
                if c_key in ignore_children:
                    # skip certain children (e.g. op type in binary operation node)
                    # because we are "pulling them up" to the name of this node.
                    continue
                c_manip = MutableAst(c_ast, assign_depth=next_assign_depth)
                c_manip.set_parent(self, c_key)
                self.children_dict[c_key] = c_manip

        if shallow:
            # if shallow, "empty out" self.ast so it also has no children.
            if self.isList:
                self.ast = []  # empty list
            else:
                for c_key in ast_children(py_ast).keys():
                    if c_key not in ignore_children:  # the ignore_children are part of this "node"
                        remove_ast_child(self.ast, c_key)

        if name is not None:
            self.name = name

        # finally, assign index to this node
        # use node_index if provided (e.g. for making shallow copie of node)
        # TODO: only allow if actually making a shallow node?..
        self.index = node_index if node_index else str(uuid4())

        # TODO: calculated self.num_children (if needed?)

    def set_parent(self, parent, key):
        # Set the parent on an existing tree node (called as part of parent's init)
        # TODO: could also pass in as parameters to init of children?..
        self.parent = parent
        self.key_in_parent = key
        if self.isList and self.parent:
            # For lists,
            self.name += f': {self.key_in_parent} of {self.parent.nodeType}'

    def simplify_node(self):
        # TODO: adjust cost of "renaming" simplified node based on how many distinct things it actually contains?
        #  e.g. simplified comparison absorbs all the comparison operators
        # for some types of nodes, simplify the tree by "pulling up" known-leaf children,
        # and ignoring them in the subsequent tree structure.
        ignore_children = set()

        # skip "empty" children (no meaningful value) in non-list nodes
        if hasattr(self.ast, '_fields'):
            for child_name in self.ast._fields:
                c_ast = getattr(self.ast, child_name)
                if c_ast is None or c_ast == []:
                    ignore_children.add(child_name)

        if hasattr(self.ast, 'op'):
            # operators - if the node has an 'op' field, include the type of operator in the name of this node
            # (e.g. 'BinOp' -> 'BinOp *').
            # This way, APTED will know that the actual operator needs to match (or be changed)
            self.name += ' ' + (astor.get_op_symbol(self.ast.op))
            ignore_children.add('op')

        if hasattr(self.ast, 'name') and isinstance(self.ast.name, str):
            self.name += ' name: ' + self.ast.name
            ignore_children.add('name')

        if type(self.ast).__name__ == 'Name':
            # this is a variable load/store operation
            self.name = f'{type(self.ast.ctx).__name__} identifier {self.ast.id}'
            ignore_children.add('ctx')
            ignore_children.add('id')
        elif hasattr(self.ast, 'ctx'):
            # even if this is not a variable load/store, ignore ctx parameter (but pull up slightly differently)
            self.name += f'(ctx={type(self.ast.ctx).__name__})'
            ignore_children.add('ctx')

        if type(self.ast).__name__ == 'Compare':
            self.name += f' operators: {str([type(op).__name__ for op in self.ast.ops])}'
            ignore_children.add('ops')

        # find and "pull up" any other children that are literals:
        if hasattr(self.ast, '_fields'):
            for child_name in self.ast._fields:
                c_ast = getattr(self.ast, child_name)
                if child_name not in ignore_children and is_literal(c_ast):
                    self.name += f'({child_name} = {get_node_name(c_ast)})'
                    ignore_children.add(child_name)

        return ignore_children

    @staticmethod
    def gen_short_index(long_index):
        return long_index.split('-')[-1]

    @property
    def short_index(self):
        return self.gen_short_index(self.index)

    @property
    def children(self):
        # return list of children in key-alphabetical order (for comparisons by APTED)
        return [ self.children_dict[c_key] for c_key in sorted(self.children_dict.keys()) ]

    def __str__(self):
        if self.isList or self.isLiteral:
            return self.name
        else:
            # noinspection PyBroadException
            try:
                code = astor.to_source(self.ast, source_generator_class=CustomSourceGen)
            except Exception:
                # If couldn't generate from source code using the "normal" way
                return self.name
            if code.strip() == '':  # ast has no actual string representation
                return self.name
            return code

    def to_compileable_str(self):
        # similar to __str__, but uses a different source generator class which allows for more partially-complete trees
        # to compile to bytecode.
        # Also, if the code string is empty, does not try to come up with some other representation.
        if self.isList or self.isLiteral:
            return self.name
        else:
            code = astor.to_source(self.ast, source_generator_class=RuntimeSourceGen)
            return code

    def exec(self):
        # execute the ast of this object by converting to code string via astor, then executing that string.
        # this works around issues with lineno, context, etc. which would prevent executing a manipulated ast.
        code_string = astor.to_source(self.ast, source_generator_class=CustomSourceGen)
        return exec(code_string, globals())  # not sure if exec actually returns anything

    def test_potential_timeout(self, unit_test_strings):
        # run a set of unit test strings after running the code in this AST.
        # may cuse a Timeout exception (e.g. if the code has an infinite loop)
        try:
            code_string = astor.to_source(self.ast, source_generator_class=CustomSourceGen)
            exec(code_string, globals())
            return [ eval(test) for test in unit_test_strings ]
        except:  # noqa TODO: less broad?
            return [False for test in unit_test_strings]

    def test(self, unit_test_strings: List[str]):
        # run a set of unit test strings, and catch timeout exceptions.
        with multiprocessing.Pool(processes=1) as pool:
            result = pool.apply_async(self.test_potential_timeout, (unit_test_strings,))
            try:
                return result.get(timeout=0.1)
            except multiprocessing.TimeoutError:
                print('timeout')
                return [False for test in unit_test_strings]

    # Generate mapping from index (node id) to actual node object for this MutableAst.
    # Also include any additional nodes in the existing mapping provided by additional_nodes.
    # This needs to be generated on demand as part of reasoning about transformations
    # being applied to this particular tree (which may be a copy of some other tree with identical node ids
    # that was used to come up with the edit script).
    # The (optional) additional nodes often represent some set of nodes that we expect to add to the tree
    # as part of an edit script.
    def gen_index_to_node(self, additional_nodes: Dict[str, 'MutableAst'] = None):
        if additional_nodes is None:
            additional_nodes = {}
        index_to_node = {}
        for node in breadth_first(self):
            index_to_node[node.index] = node

        for n in additional_nodes:
            index_to_node[n] = additional_nodes[n]

        return index_to_node

    def generate_dot_notation(self, tree_name: str):
        root_id = tree_name + self.short_index
        color_string = ''
        if hasattr(self, 'color'):
            color_string = f',color="{self.color}"'
        dot_string = f'{root_id} [label="{self.name}"{color_string}];\n'
        for child_i in self.children_dict:
            child = self.children_dict[child_i]
            child_id = tree_name + child.short_index
            dot_string += child.generate_dot_notation(tree_name)
            dot_string += f'{root_id} -> {child_id} [label="{child_i}"];\n'
        return dot_string

    def write_dot_file(self, tree_name: str, filename: str):
        # TODO: use pydot or graphviz library?
        with open(f'{filename}', 'w') as f:
            f.write(f'''
                        digraph G
                        {{
                            {self.generate_dot_notation(tree_name)}
                        }}
                    ''')

    def generate_xml_for_gumtree(self, parent_elem=None):
        if parent_elem is None:
            my_elem = Element('tree')
        else:
            my_elem = SubElement(parent_elem, 'tree')

        # Dummy values for position and length in code text:
        # TODO: actually calculate?
        my_elem.set('pos', '0')
        my_elem.set('length', '0')

        # node identifiers:
        my_elem.set('type', self.nodeType)
        if self.nodeType != self.name:
            my_elem.set('label', self.name)
        # Note: in the gumtree paper, it seems 'type' is called 'label' and 'label'  is called 'value'.

        for c in self.children:
            c.generate_xml_for_gumtree(my_elem)

        return my_elem

    def generate_xml_file_for_gumtree(self, filename):
        return ElementTree(self.generate_xml_for_gumtree()).write(f'{filename}')

    ### Tree Manipulation functions: ###

    def update_ast_for_child(self, child_ast, key):
        # a child node has changed its ast; replace the old child with the new one in the ast,
        # depending on whether I am a list or a proper node.
        if self.isList:
            self.ast[key] = child_ast
        else:
            setattr(self.ast, key, child_ast)

    def update(self, new_node: 'MutableAst'):
        # update contents of the node to match new_node, keeping/transferring all the same children.

        if self.isList:
            if not new_node.isList:
                # this function makes no sense for lists-and-node pairs
                # APTED should never map two nodes like that because of the cost configuration
                raise Exception(f"Trying to update List node with non-list: {self.name, new_node.name}")
            # if just updating a list with a list, just change the name of this list - the change is superficial
            # (only the name actually differs, no AST changes)
            self.name = new_node.name
            return

        ## update own properties:
        self.name = new_node.name
        old_ast = self.ast
        self.ast = new_node.ast
        # parent, key_in_parent, index, children_dict stay the same - they don't depend on the underlying ast object

        ## update children & parents of actual python asts:

        # assume both are Nodes (or literals, in which case the children stuff is moot)
        for c_key in self.children_dict:
            setattr(self.ast, c_key, self.children_dict[c_key].ast)

        # update parent ast:
        if self.parent is not None:
            self.parent.update_ast_for_child(self.ast, self.key_in_parent)

    def remove_child(self, child_node):
        # remove passed-in node as child.
        # return key/index from which it was removed, or False if child was not present.
        c_key = child_node.key_in_parent
        if (c_key is not None) and (c_key in self.children_dict):
            del self.children_dict[c_key]
            if self.isList:
                # the underlying ast is a list, and the key is the index in the list.
                self.ast.pop(c_key)
                # redo dict keys manually - doesn't seem to be a great alternative,
                # since we have MutableAst nodes that are only linked through this dict.
                new_cdict = {}
                for i in self.children_dict:
                    new_i = i
                    if i > c_key:
                        new_i -= 1
                    new_cdict[new_i] = self.children_dict[i]
                    self.children_dict[i].key_in_parent = new_i
                self.children_dict = new_cdict
            else:
                # underlying ast is not a list - set the corresponding attribute to None or [],
                # depending on what we think Python expects
                # TODO: use function
                placeholder = None
                if child_node.isList:
                    # guess that this is a list-type parameter, and  the empty version is []
                    placeholder = []

                # TODO: unit test?
                # if this is a special ambiguous case, make sure we use the right one,
                # based both on the node type and field name
                attr_string = f'{type(self.ast).__name__}.{c_key}'
                if attr_string in extra_special_cases:
                    placeholder = extra_special_cases[attr_string]

                setattr(self.ast, c_key, placeholder)
            return c_key
        return False

    def add_child_at_key(self, child, key):
        # add child node and use the given key in this (parent) node. On success, return the key (w/o change)
        if self.isList:
            raise ForbiddenEditException("Trying to add_child_at_key on List node")

        # if there's already something at key - "scoot it over" to some other unique key (just in terms of Manip tree)
        # TODO: unit test?
        if key in self.children_dict:
            old_key = f'old_{key} {self.name} {self.nodeType} {self.index}'
            self.children_dict[key].key_in_parent = old_key
            self.children_dict[key].orig_key = key
            self.children_dict[key].displaced_by = child.index
            self.children_dict[old_key] = self.children_dict[key]

        self.children_dict[key] = child
        setattr(self.ast, key, child.ast)

        child.parent = self
        child.key_in_parent = key

        return key

    def add_child_between(self, before_child, after_child, new_child):
        # add child between two other children; return index of insertion (only makes sense for 'list' nodes)
        # assumes before_child and after_child are properly ordered
        if not self.isList:
            raise ForbiddenEditException("Trying to add_child_between on a non-list node")

        ## find desired index
        my_i = None
        if before_child is not None and before_child.parent == self:
            my_i = before_child.key_in_parent + 1
        elif after_child is not None and after_child.parent == self:
            my_i = after_child.key_in_parent  # take the index of after_child, pushing after_child to the index after
        else:
            # both before and after are None - append to end of list
            my_i = 0  # len(self.ast)

        ## insert into children_dict, rewriting dict as needed
        new_dict = {my_i: new_child}
        for j in self.children_dict:
            if j >= my_i:
                new_dict[j+1] = self.children_dict[j]
                self.children_dict[j].set_parent(self, j+1)
            else:
                new_dict[j] = self.children_dict[j]
        self.children_dict = new_dict

        ## insert ast into self.ast
        self.ast.insert(my_i, new_child.ast)

        new_child.parent = self
        new_child.key_in_parent = my_i

        return my_i

    def add_child_anywhere(self, child):
        if self.isList:
            self.add_child_between(None, None, child)
        else:
            new_key = 'key' + str(len(self.children_dict))
            self.add_child_at_key(child, new_key)

    def get_child_neighbors(self, child):
        # get the nodes before and after a child (only makes sense for 'list' nodes)
        if not self.isList:
            raise Exception("Trying to get_child_neighbors on a non-list node")
        child_i = child.key_in_parent
        before_node = self.children_dict.get(child_i-1, None)
        after_node = self.children_dict.get(child_i+1, None)
        return before_node, after_node


def breadth_first(tree: MutableAst):
    # traverse given tree object in breadth-first order
    queue = [tree]
    while len(queue) > 0:
        node = queue.pop(0)
        queue.extend(node.children)
        yield node


def depth_first(tree: MutableAst):
    # traverse given tree object in breadth-first order
    queue = [tree]
    while len(queue) > 0:
        node = queue.pop()
        queue.extend(node.children[::-1])  # append in reverse order, so they get popped in correct order
        yield node


def postorder(tree: MutableAst):
    # traverse given tree object in postorder (children before their parent)
    for c in reversed(tree.children):
        yield from postorder(c)
    yield tree
