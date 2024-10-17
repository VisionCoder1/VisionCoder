import roles
from pathlib import Path
from base_communication import BaseCommunication
import os

class DG(BaseCommunication):
    def __init__(self, work_dir:Path) -> None:
        self.work_dir = work_dir
        self.function_list = []
        self.dg_list = []
        self.project = None
        self.test_file = None
        super().__init__()

    def pair_programming(self, coder_name:str, tester_name:str, code_filename:str) -> None:
        print(f"DG: Pair Programming...")
        coder = roles.Coder(agent_name=coder_name, work_dir=self.work_dir)
        coder_ori_code = coder.get_code()

        with open(f"{os.path.join(self.work_dir, 'test_'+code_filename)}", "r") as f:
            test_code = f.read()
        coder_prompt = [
            f"Tester has written a test code for your code:\n",
            f"\"\"\"\n{test_code}\n\"\"\"\n",
            f"Check the test code, if there are any errors, fix this test code.\n",
            f"If there are no errors, return the original test code.\n",
        ]
        coder.coding("".join(coder_prompt), store_memory=False)
        new_test_code = coder.get_code()
        coder.role["code"] = coder_ori_code
        # coder.dump_files()
        coder.release()

        tester = roles.Tester(agent_name=tester_name, work_dir=self.work_dir)
        tester.role["code"] = new_test_code
        tester.dump_files()
        tester.release()


    def run_coding(self, function_name:str, module_name:str, coding_language:str, module_description:str, function_description:str) -> None:
        print(f"DG: {function_name} coding...")
        coder_name = function_name + "_coder"
        tester_name = function_name + "_tester"

        coder = roles.Coder(agent_name=coder_name, work_dir=self.work_dir)

        coder.load_knowledge("./RAG/Coder_RAG.json")
        coder_context = coder.retrieve(function_description)
        init_coder_prompt = [
            f"You are a coder working on the project: {self.project}.",
            f"You are assigned to the module {module_name}: {module_description}.\n",
            f"Please finish the following function in {coding_language}:\n{function_description}\n",
            f"Follow strcitly the function signature and the description. Do not change the function signature!!!\n",
            f"You are supposed to include the necessary libraries first and then write the code for the function.\n",
            f"Do not include any GUI modules.\n",
            f"You need to generate the code in the following format:\n```LANGUAGE\n'''\nDOCSTRING\n'''\nCODE\n```\n",
            f"where LANGUAGE is the programming language, DOCSTRING is the description of the function, and CODE is the code.",
            # f"You may use the following context if necessary:\n{coder_context}\n",
        ]
        filename = function_name + ".py"
        coder.gen_init_code("".join(init_coder_prompt), filename)
        coder.release()

        tester = roles.Tester(agent_name=tester_name, work_dir=self.work_dir)
        with open(f"{os.path.join(self.work_dir, filename)}", "r") as f:
            code = f.read()
        tester.load_knowledge("./RAG/Tester_RAG.json")
        tester_context = tester.retrieve(code)
        init_tester_prompt = [
            f"You are a tester working on the project: {self.project}.",
            f"You will be testing the function {function_name}: \n{function_description}.\n",
            f"The coder has already implemented the function:\n{code}\n",
            f"The python file name is {filename}.\n",
            f"You should generate the test code within the following steps:\n",
            f"1. Import the source function 'from xxx import yyy' or 'import xxx'.\n",
            f"2. Import the unittest module. (But don't use unittest.mock or tearDown)\n",
            f"3. Write a simple test case to test the function. Test if the function can execute is enough. Do not make any assumption.\n",
            f"4. Implement 'if __name__ == '__main__':' in your test code.\n",
            f"You need to generate the test code in the following format:\n```LANGUAGE\n'''\nDOCSTRING\n'''\nCODE\n```\n",
            f"where LANGUAGE is the programming language, DOCSTRING is the description of the function, and CODE is the code. ",
            f"You need to return only the test code including the 'import' part."
        ]
        tester.gen_test_code(init_tester_prompt, filename)
        tester.release()

        # self.pair_programming(coder_name, tester_name, filename)
        res = self.run_test(f"test_{filename}")

        refine_conter = 0
        while ("FAILED" or "Traceback") in res and refine_conter < 1:
            coder = roles.Coder(agent_name=coder_name, work_dir=self.work_dir)
            try:
                coder.refine_code(res)
            except Exception as e:
                print(e)
            coder.release()

            res = self.run_test(f"test_{filename}")

            refine_conter += 1
        
        # analyzer = MistralAnalyser(work_dir=work_dir)
        # with open(f"{os.path.join(work_dir, filename)}", "r") as f:
        #     code = f.read()
        # analyzer.gen_analysis(code, filename)

    def run_DG(self) -> None:
        self.project = self.proj_info["project"]
        self.module_list = self.proj_info["TL_modules"]
        for i in range(len(self.module_list)):
            module_name = self.module_list[i]["module_name"]
            module_description = self.module_list[i]["module_description"]
            for function in self.proj_info[f"{module_name}_functions_refined"]:
                function_name = function["function_name"]
                coding_language = function["coding_language"]
                function_description = function["function_description"]
                self.run_coding(function_name, module_name, coding_language, module_description, function_description)
        print("DG: Done.")