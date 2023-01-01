import json

import src.python_fix_explainer.interface_funcs as funcs


student_code = '''
def isEvenPositiveInt(n):
    if n % 2 == 0 and n > 0 and Type(n) == Type(int):
        return True
'''


unit_tests = [
    'isEvenPositiveInt(2) == True',
    'isEvenPositiveInt(2040608) == True',
    'isEvenPositiveInt(21) == False',
    'isEvenPositiveInt(0) == False',
    'isEvenPositiveInt("yikes!") == False',
]


correct = [
    '''
def isEvenPositiveInt(x):
    if x == 0:
        return False
    if type(x) == int and x % 2 == 0 and x > 0:
        return True
    return False
    ''',
    '''
def isEvenPositiveInt(n):
    return type(n) == int and n > 0 and n % 2 == 0 
    '''
]

# student_code = '''
# def helloWorld():
#     print('Hello World!')
# '''
#
# unit_tests = [
#     'helloWorld() == "Hello World!"'
# ]
#
# correct = [
#     '''
# def helloWorld():
#     return 'Hello World!'
#     '''
# ]

out = funcs.fix_code(student_code, unit_tests, correct)
# print(out)
print(json.dumps(out, indent=2))
# for thing in out:
#     for op in thing['synced_trace']:
#         print(op)
