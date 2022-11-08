# python_fix_explainer
Generate fixes for incorrect python code, given data about the problem being solved

A good entry point is [end_to_end.py](end_to_end.py): it's an end-to-end example of how the system is designed to work.

Given:
- An incorrect solution to a programming problem (usually written by a student)
- Some set of correct solutions (often written by other students)
- Some set of correctness checks (unit tests)

Try to generate a (new) corrected solution that is close to the incorrect solution, but fixes all the bugs. 

Present this correction as a series of fixes, and provide explanation about how this fix improves things: how does it change what the code does when it runs?


