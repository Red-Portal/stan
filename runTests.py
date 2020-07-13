#!/usr/bin/python

from __future__ import print_function
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import os.path
import platform
import re
import subprocess
import sys
import time

winsfx = ".exe"
testsfx = "_test.cpp"


def processCLIArgs():
    """
    Define and process the command line interface to the runTests.py script.
    """
    cli_description = "Generate and run stan math library tests."
    cli_epilog = "See more information at: https://github.com/stan-dev/stan"

    parser = ArgumentParser(
        description=cli_description,
        epilog=cli_epilog,
        formatter_class=RawTextHelpFormatter,
    )

    # Now define all the rules of the command line args and opts
    parser.add_argument(
        "-j", metavar="N", type=int, default=1, help="number of cores for make to use"
    )

    tests_help_msg = "The path(s) to the test case(s) to run.\n"
    tests_help_msg += "Example: 'src/test/unit', 'src/test/integration', and/or\n"
    tests_help_msg += "         'src/test/unit/version_test'"
    parser.add_argument("tests", nargs="+", type=str, help=tests_help_msg)
    parser.add_argument(
        "-m",
        "--make-only",
        dest="make_only",
        action="store_true",
        help="Don't run tests, just try to make them.",
    )
    parser.add_argument("--only-functions", nargs="+", type=str, default=[],
                        help="Function names to run expression tests for. Default: all functions")
    # And parse the command line against those rules
    return parser.parse_args()


def stopErr(msg, returncode):
    """Report an error message to stderr and exit with a given code."""
    sys.stderr.write("%s\n" % msg)
    sys.stderr.write("exit now (%s)\n" % time.strftime("%x %X %Z"))
    sys.exit(returncode)


def isWin():
    return platform.system().lower().startswith(
        "windows"
    ) or os.name.lower().startswith("windows")


batchSize = 24 if isWin() else 200


def mungeName(name):
    """Set up the makefile target name"""
    if name.startswith("src") or name.startswith("./src"):
        name = name.replace("src/", "", 1)
    if name.endswith(testsfx):
        name = name.replace(testsfx, "_test")
        if isWin():
            name += winsfx
            name = name.replace("\\", "/")
    return name


def doCommand(command, exit_on_failure=True):
    """Run command as a shell command and report/exit on errors."""
    print("------------------------------------------------------------")
    print("%s" % command)
    if isWin() and command.startswith("make "):
        command = command.replace("make ", "mingw32-make ")
    p1 = subprocess.Popen(command, shell=True)
    p1.wait()
    if exit_on_failure and (not (p1.returncode is None) and not (p1.returncode == 0)):
        stopErr("%s failed" % command, p1.returncode)


def modelDependencies(tests):
    dependencies = []
    for filepath in tests:
        filepath = "src/" + filepath + ".cpp"
        if os.path.isfile(filepath) and filepath.endswith(testsfx):
            with open(filepath) as file:
                test_file_content = file.read()
                # look for TEST() and TEST_F()
                matches = re.findall(
                    r"#include <test/test-models/.*hpp>", test_file_content
                )
                for x in matches:
                    x = x.replace("#include <", "").replace(">", "")
                    dependencies.append(x)
    return dependencies


def makeTest(name, j):
    """Run the make command for a given single test."""
    doCommand("make -j%d %s" % (j or 1, name))


def runTest(name):
    executable = mungeName(name).replace("/", os.sep)
    xml = mungeName(name).replace(winsfx, "")
    command = '%s --gtest_output="xml:%s.xml"' % (executable, xml)
    doCommand(command)


def findTests(base_path):
    folders = filter(os.path.isdir, base_path)
    nonfolders = list(set(base_path) - set(folders))
    tests = nonfolders + [
        os.path.join(root, n)
        for f in folders
        for root, _, names in os.walk(f)
        for n in names
        if n.endswith(testsfx)
    ]
    return list(map(mungeName, tests))


def batched(tests):
    return [tests[i: i + batchSize] for i in range(0, len(tests), batchSize)]


def handleExpressionTests(tests, only_functions, n_test_files):
    expression_tests = False
    for n, i in list(enumerate(tests))[::-1]:
        if "test/expressions" in i or "test\\expressions" in i:
            del tests[n]
            expression_tests = True
    if expression_tests:
        import generateExpressionTests
        generateExpressionTests.main(only_functions, n_test_files)
        for i in range(n_test_files):
            tests.append("test/unit/expressions/tests%d_test.cpp"%i)
    elif only_functions:
        stopErr("--only-functions can only be specified if running expression tests (test/expressions)", -1)


def main():
    inputs = processCLIArgs()

    tests = inputs.tests

    handleExpressionTests(tests, inputs.only_functions, inputs.j * 2)

    tests = findTests(tests)
    if not tests:
        stopErr("No matching tests found.", -1)

    for batch in batched(tests):
        modelHpp = modelDependencies(batch)
        if len(modelHpp) > 0:
            makeTest(" ".join(modelHpp), inputs.j)
        makeTest(" ".join(batch), inputs.j)

    if not inputs.make_only:
        for t in tests:
            runTest(t)


if __name__ == "__main__":
    main()
