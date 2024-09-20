import doctest

# a global  DocTestRunner object which can run all the doctests.
#  (pretty sure that only one is needed)
doctestrunner = doctest.DocTestRunner()

# initialize global variable which stores doctest objects to be run when needed
# TODO: something less global, e.g. passing this object in to the test interfaces to use in addition to globals()
doctests = {}


def run_doctest_test(test_name, test_globals):
    # generate DocTest object using current globals and run it
    run_results = doctestrunner.run(
        doctest.DocTest(doctests[test_name],
                        test_globals,
                        test_name,
                        None, None, None)
    )
    return run_results.failed == 0
