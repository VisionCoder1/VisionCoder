import os
import shutil

from coder import Coder
from tester import Tester
from analysor import MistralAnalyser
from leader import TeamLeader, ModuleLeader, FunctionCoordinator


def test_coder():
    llama_inst = Coder(work_dir="./test")
    prompt = "Write a Python function that takes a list of integers as input and returns the sum of all the integers."
    llama_inst.coding(prompt)
    llama_inst.gen_filename()
    llama_inst.dump_files()
    llama_inst.release()

def test_coder_refinement():
    refine_dir = "./test/case_refine/"
    llama_inst = Coder(work_dir=refine_dir)
    running_result = """
    F
    ======================================================================
    FAIL: test_sum_integers (__main__.TestSumIntegers.test_sum_integers)
    ----------------------------------------------------------------------
    Traceback (most recent call last):
    File "/work/test_sum_integers.py", line 6, in test_sum_integers
        self.assertEqual(sum_integers([1, 2, 3]), 6)
    AssertionError: 1 != 6

    ----------------------------------------------------------------------
    Ran 1 test in 0.000s

    FAILED (failures=1)
    """
    llama_inst.refine_code(running_result)
    llama_inst.dump_files()
    llama_inst.release()

    tester = Tester(work_dir=refine_dir)
    print(tester.run_test())
    tester.release()


def test_tester():
    tester = Tester(work_dir="./test")
    filename = "sum_integers.py"
    prompt = "Write a Python function that takes a list of integers as input and returns the sum of all the integers."
    with open(f"./test/{filename}", "r") as f:
        code = f.read()
    tester.coding(prompt, code, filename)
    tester.dump_files()
    tb_info = tester.run_test()
    print("*" * 10 + "INFO" + "*" * 10)
    print(tb_info)
    tester.release()

def test_all():
    work_dir = "./test/case_heirarchy/"
    # if os.path.exists(work_dir):
    #     shutil.rmtree(work_dir)

    coder = Coder(agent_name="Coder", work_dir=work_dir)
    prompt = "Write a Python function that takes a list of integers as input and returns the sum of all the integers."
    coder.gen_init_code(prompt)
    filename = coder.get_filename()
    coder.release()

    tester = Tester(agent_name="Tester", work_dir=work_dir)
    with open(f"{os.path.join(work_dir, filename)}", "r") as f:
        code = f.read()
    tester.gen_test_code(prompt, code, filename)
    res = tester.run_test()
    tester.release()

    while "FAILED" in res:
        coder = Coder(agent_name="Coder", work_dir=work_dir)
        coder.refine_code(res)
        coder.release()

        tester = Tester(agent_name="Tester", work_dir=work_dir)
        res = tester.run_test()
        tester.release()
    
    # analyzer = MistralAnalyser(work_dir=work_dir)
    # with open(f"{os.path.join(work_dir, filename)}", "r") as f:
    #     code = f.read()
    # analyzer.gen_analysis(code, filename)


def test_teamleader():
    leader = TeamLeader(work_dir="./test/case_heirarchy/")
    prompt = "Develop a simple web service that allows a user uploads an image and returns the image in grayscale."
    response = leader.split_job(prompt)
    print(response)
    leader.dump_files()
    leader.release()


def test_moduleleader():
    leader = ModuleLeader(work_dir="./test/case_heirarchy/")
    project = "Develop a simple web service that allows a user uploads an image and returns the image in grayscale."
    module_name = "Grayscale Conversion"
    module_description = "This module deals with the conversion logic. It processes the uploaded images and converts them into grayscale."
    response = leader.split_function(project, module_name, module_description)
    print(response)
    leader.dump_files()
    leader.release()


def test_functioncoordinator():
    leader = FunctionCoordinator(work_dir="./test/case_heirarchy/")
    project = "Develop a simple web service that allows a user uploads an image and returns the image in grayscale."
    module_name = "Grayscale Conversion"
    module_description = "This module deals with the conversion logic. It processes the uploaded images and converts them into grayscale."
    function_list = [
        "1. Load_Image: Function to load the uploaded image from the user.",
        "2. Convert_to_Grayscale: Function to convert the loaded image into a grayscale format.",
        "3. Save_Grayscale_Image: Function to save the converted grayscale image.",
        "4. Display_Grayscale_Image: Function to display the grayscale image to the user.",
        "5. Error_Handling: Function to handle any errors that may occur during the grayscale conversion process.",
    ]
    response = leader.refine_function(project, module_name, module_description, function_list)
    print(response)
    leader.dump_files()
    leader.release()

if __name__ == "__main__":
    test_all()