# python_fix_explainer
Generate fixes for incorrect python code, given data about the problem being solved.

Given:
- An incorrect solution to a programming problem (usually written by a student)
- Some set of correct solutions (often written by other students)
- Some set of correctness checks (unit tests)

Try to generate a (new) corrected solution that is close to the incorrect solution, but fixes all the bugs. 

Present this correction as a series of fixes, and provide explanation about how each fix improves things: how does it change what the code does when it runs?

Alternatively, don't split up into individual fixes; just generate a comparison (with synced execution trace) between the buggy code and the generated corrected code.

For more information, see these papers:

- [Yana Malysheva and Caitlin Kelleher. 2022. An Algorithm for Generating Explainable Corrections to Student Code. In Proceedings of the 22nd Koli Calling International Conference on Computing Education Research (Koli Calling '22). Association for Computing Machinery, New York, NY, USA, Article 13, 1–11.](https://doi.org/10.1145/3564721.3564731)
- [Yana Malysheva and Caitlin Kelleher. 2022. Assisting Teaching Assistants with Automatic Code Corrections. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems (CHI '22). Association for Computing Machinery, New York, NY, USA, Article 231, 1–18.](https://doi.org/10.1145/3491102.3501820)


This code is deployed on testpypi as python-fix-explainer. Note that **it was written for Python 3.7**; in many cases, it works with Python 3.8, but later versions of Python have incompatible implementations of code-to-bytecode compilation (Look, I'm doing a lot of crazy stuff to get the detailed execution trace. You don't even want to know.) 

To install:

`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple  python-fix-explainer`


A good starting point for testing out this library is the [simple_fix_gen](https://github.com/yanamal/simple_fix_gen/) repository. It's a simple example that generates fixes and then creates a relatively simple html view of the analysis.




