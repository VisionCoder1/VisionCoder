from base_agent import BaseGPTAgent
import re
from typing import List
import os

class TeamLeader(BaseGPTAgent):
    def __init__(self, agent_name:str="TeamLeader", model_id:str="gpt-4o-mini", **kwarg) -> None:
        sys_prompt = [
            "You are responsible for virtually managing a development project.\n",
            "You are at the top of the project hierarchy and responsible for overseeing the following roles:\n",
            "1. Module Leaders: Each Module Leader oversees a specific segment of the project. ",
            "Interaction with You: Module Leaders are at the next level of the hierarchy. You will assign modules to Module Leaders and receive final codes from them.\n",
            "2. Function Coordinators: Function Coordinators act as intermediaries between the Module Leaders and the Development Groups. They translate broad project tasks into specific, actionable functions. ",
            "Interaction with You: While you do not typically interact directly with Function Coordinators, they will typically assemble the functions written by Development Groups and prepare them for Module Leaders to review.\n",
            "3. Development Groups: Each group consists of a Junior Coder and a Tester. They are responsible for the hands-on implementation of project functions. Junior Coders write the code, while Testers ensure the code meets quality standards.",
            "Interaction with You: While you do not typically interact directly with Development Groups, the fundamental functions of the project are implemented by them and they report to Function Coordinators.\n\n",
            "Upon Receiving a New Project: You split the project into modules and define the scope and objectives for each. Then You assign modules to respective Module Leaders, providing them with clear instructions and resources. Do not include testing or any GUI development in modules.\n"
        ]
        self.completed_modules = []

        super().__init__(model_id=model_id, sys_prompt=''.join(sys_prompt), agent_name=agent_name, **kwarg)

    def split_modules(self, module_plan:str) -> List[str]:
        modules = re.split(r"\n\d+\. ", module_plan)[1:]
        dict_modules = []
        for module in modules:
            # module_name is the string before colon
            module_name = module.split(":")[0]
            module_name = re.sub(r"[^\w-]", "", module_name)  # remove any symbols in module_name except for letters, numbers, and hyphens

            # coding_language is the string after module name and between "**"
            coding_language = re.search(r"\*\*(.*?)\*\*", module).group(1)
            # module_description is the string after coding_language
            module_description = module.split(coding_language + "**")[1].strip()
            dict_modules.append({"module_name": module_name, "coding_language": coding_language, "module_description": module_description})
        return dict_modules

    def split_job(self, prompt:str) -> None:
        self.load_knowledge("./RAG/TL_RAG.json")
        context = self.retrieve(prompt)
        prompt_text = [
            f"Here is the description of the project:\n{prompt}.\n",
            f"Please split the project into modules and specify the scope and objectives for each module.\n", 
            f"And select proper coding languages for each module from the following list: [Python].\n",
            f"Use as few modules as possible.\n",
            f"List each module strictly in \"NUMBER. MODULE_NAME: **CODING LANGUAGE** MODULE_DESCRIPTION\" format."
            # f"You can use the information from your priori knowledge for inspiration if necessary:\n{context}\n",
            # f"But you need to be based on the current project description.\n"
        ]
        prompt = {"role": "user",
                    "content": "".join(prompt_text)}
        self.combine_prompt(prompt)
        response = self.chat()
        response = "\n" + response.strip()
        modules = self.split_modules(response)
        self.combine_prompt({"role": "assistant", "content": f"{modules}"})
        self.role["modules"] = modules
    
    def select_env(self) -> str:
        prompt_text = [
            f"The project will be implemented using Docker. ",
            f"Now please select the docker image for the project from the following list: [image:1.0, flask:1.0].\n",
            f"image:1.0: A docker image with image processing libs installed.\n",
            f"flask:1.0: A docker image with service development libs installed.\n",
            f"Please provide the docker image name only."
        ]
        prompt = {"role": "user",
                    "content": "".join(prompt_text)}
        self.combine_prompt(prompt)
        response = self.chat().strip()
        response = re.search(r"\w+:\d\.\d", response).group()
        self.combine_prompt({"role": "assistant", "content": f"{response}"})
        self.role["docker_image"] = response
        return response


    def gather_modules(self) -> None:
        for module in self.role["modules"]:
            module_filename = module["module_name"] + ".py"
            with open(f"{os.path.join(self.work_dir, module_filename)}", "r") as f:
                code = f.read()
            self.completed_modules.append({"module_filename": module_filename, "module_code": code})

    def assemble_modules(self) -> None:
        self.gather_modules()
        prompt = [
            f"Now your Module Leaders have finished the assigned modules:\n",
            f"{self.completed_modules}\n",
            f"You are expected to assemble the final codes and return it back to client.\n"
            f"These module files are in the current directory. \n",
            f"Import those modules from the file and call them in the main function.\n",
            f"Please provide the final code in the following format:\n",
            f"```LANGUAGE\n'''\nDOCSTRING\n'''\nCODE\n```\n"
        ]
        prompt = {"role": "user",
                    "content": "".join(prompt)}
        self.combine_prompt(prompt)
        response = self.chat().strip()
        if re.findall(r"```(.*?)\n(.*?)```", response, re.DOTALL):
            code = re.findall(r"```(.*?)\n(.*?)```", response, re.DOTALL)[0][1]
        else:
            code = response
        code = code.strip()
        self.combine_prompt({"role": "assistant", "content": f"{code}"})
        self.role["code"] = code
        self.role["filename"] = "main.py"


class ModuleLeader(BaseGPTAgent):
    def __init__(self, agent_name:str="ModuleLeader", model_id:str="gpt-4o-mini", **kwarg) -> None:
        sys_prompt = [
            "You are responsible for overseeing a specific segment of a development project. ",
            "You are in charge of a module, which is a distinct part of the project. Your role involves managing the module's development, ensuring that it meets the project's objectives, and reporting progress to the Team Leader.\n",
            "Your primary responsibilities and interactions include:\n",
            "1. Team Leader: Team Leader is responsible for virtually managing the development project at the highest level.\n",
            "Interaction with You: You will receive instructions from the Team Leader. After your department finishes the assigned module, you are expected to provide final codes to Team Leader\n",
            "2. Function Coordinators: Function Coordinators act as intermediaries between the Module Leaders and the Development Groups. They translate broad project tasks into specific, actionable functions. ",
            "Interaction with You: You are responsible for providing function descriptions of the current module to the Function Coordinator. After development, you will receive a list of functions from the Function Coordinators. You will be responsible for ensuring that the functions are implemented correctly and meet the module's objectives.\n",
            "Upon Receiving a New Module: You will define the scope and objectives of the module, break down tasks into actionable functions, and assign these functions to Function Coordinators for further delegation to Development Groups.\n"
        ]

        super().__init__(model_id=model_id, sys_prompt=''.join(sys_prompt), agent_name=agent_name, **kwarg)
    
    def retrieve_functions(self, module_plan:str, coding_language:str) -> List[dict]:
        functions = re.split(r"\n\d+\. ", module_plan)[1:]
        dict_functions = []
        for function in functions:
            # function_name is the string before colon
            function_name = function.split(":")[0]
            # module_description is the string after coding_language
            function_description = function.split(function_name+":")[1].strip()
            dict_functions.append({"function_name": function_name, "coding_language": coding_language, "function_description": function_description})
        return dict_functions

    def split_function(self, project:str, all_modules:list, module_name:str, coding_language:str, module_description:str) -> None:
        self.load_knowledge("./RAG/ML_RAG.json")
        context = self.retrieve(module_description)
        prompt = [
            f"Here is the description of the project you are working on: {project}\n",
            f"The Team Leader has split the project into modules: {all_modules}\n",
            f"And you are assigned to work on the {module_name} module: {module_description}\n",
            f"This module will be implemented using {coding_language}.\n",
            f"You are going to break down the tasks into actionable functions. Provide clear inputs, outputs, and descriptions for each function.\n",
            f"Use as few functions as possible.\n",
            f"List each function in \"NUMBER. FUNCTION_NAME: FUNCTION_DESCRIPTION\" format."
            # f"You can use the information from your priori knowledge for inspiration if necessary:\n{context}\n",
            # f"But you need to be based on the current project description.\n"
        ]
        prompt = {"role": "user",
                    "content": "".join(prompt)}
        self.combine_prompt(prompt)
        response = self.chat()
        response = "\n" + response.strip()
        modules = self.retrieve_functions(response, coding_language)
        self.combine_prompt({"role": "assistant", "content": f"{modules}"})
        self.role["modules"] = modules

    def gen_module_test_file(self, module_name:str) -> str:
        module_file = os.path.join(self.work_dir, module_name + ".py")
        with open(module_file, "r") as f:
            module_code = f.read()
        prompt = [
            f"Now your Function Coordinator have finished the assigned {module_name} module: {module_name}.py.\n",
            f"Here is the code for the {module_name} module:\n",
            f"{module_code}\n",
            f"You are expected to write a test code to test the module.\n",
            f"Import the original function from the original file first! Import from file, do not copy it!\n",
            f"Please provide the test code in the following format:\n",
            f"```LANGUAGE\n'''\nDOCSTRING\n'''\nCODE\n```\n"
            f"Return code only."
        ]
        prompt = {"role": "user",
                  "content": "".join(prompt)}
        self.combine_prompt(prompt)
        response = self.chat().strip()
        if re.findall(r"```(.*?)\n(.*?)```", response, re.DOTALL):
            test_code = re.findall(r"```(.*?)\n(.*?)```", response, re.DOTALL)[0][1]
        else:
            test_code = response
        test_code = test_code.strip()
        self.combine_prompt({"role": "assistant", "content": f"{test_code}"})
        filename = f"test_module_{module_name}.py"
        self.role["code"] = test_code
        self.role["filename"] = filename
        return filename



class FunctionCoordinator(BaseGPTAgent):
    def __init__(self, agent_name:str="FunctionCoordinator", model_id:str="gpt-4o-mini", **kwarg) -> None:
        sys_prompt = [
            "You are in charge of a set of functions within a module. You are responsible for translating these simple function requirements into detailed function descriptions, and reporting progress to the Module Leader.\n",
            "Your primary responsibilities and interactions include:\n",
            "1. Module Leader: Module Leader is responsible for overseeing a specific segment of the development project. ",
            "Interaction with You: You will receive rough function requirements from the Module Leader. After your department finishes the assigned functions, you are expected to provide final codes to the Module Leader\n",
            "2. Development Groups: Each group consists of a Junior Coder and a Tester. They are responsible for the hands-on implementation of project functions. Junior Coders write the code, while Testers ensure the code meets quality standards. ",
            "Interaction with You: You will delegate functions to Development Groups. You are responsible for ensuring that the functions are implemented correctly and meet the module's objectives.\n",
            "Upon Receiving a new set of function requirements: You will be responsible for translating these simple function requirements into detailed function descriptions\n"
            "And when Development Groups finish the assigned functions, you are expected to assemble them and provide final codes to the Module Leader.\n"
        ]

        self.list_functions = []
        self.completed_functions = []
        super().__init__(model_id=model_id, sys_prompt=''.join(sys_prompt), agent_name=agent_name, **kwarg)

    def split_functions(self, function_plan:str, coding_language:str) -> None:
        functions = re.split(r"\ndef ", function_plan)[1:]
        for function in functions:
            self.list_functions.append({"function_name": function.split("(")[0], "coding_language": coding_language, "function_description": "def " + function})

    def refine_function(self, module_list:list, project:str, module_name:str, coding_language:str, module_description:str, function_list:list) -> None:
        self.load_knowledge("./RAG/FC_RAG.json")
        context = self.retrieve(str(function_list))
        prompt = [
            f"Here is the description of the project you are working on: {project}.",
            f"The Team Leader has split the project into the following modules: {module_list}.\n",
            f"You are assigned to work on the {module_name} module: {module_description}\n",
            f"Your Module Leader has provided you with a list of functions that need to be implemented: {str(function_list)}.\n\n",
            f"Please refine the function requirements into the function signature in the following format, specify exactly which external lib should be used:\n",
            f"\"\"\"\n",
            f"import LIBRARY\n",
            f"def FUNCTION_NAME(ARGUMENTS:TYPES):\n",
            f"    '''\n",
            f"    FUNCTION_DESCRIPTION\n",
            f"    ARGUMENTS: TYPES\n",
            f"    RETURN: TYPE\n",
            f"    '''\n",
            f"    # TODO\n",
            f"\"\"\"\n",
            f"When you refine each function, pay attention to the connection between the functions, i.e. the output of one function may need to be the input of the next function.\n",
            # f"You can use the information from your priori knowledge for inspiration if necessary:\n{context}\n",
            # f"But you need to be based on the current project description.\n"
        ]
        prompt = {"role": "user",
                    "content": "".join(prompt)}
        self.combine_prompt(prompt)
        response = self.chat()
        response = "\n" + response.strip()
        self.split_functions(response, coding_language)
        self.combine_prompt({"role": "assistant", "content": f"{self.list_functions}"})
        self.role["modules"] = self.list_functions

    def gather_functions(self) -> List[str]:
        assembled_function_files = []
        for function in self.role["modules"]:
            function_filename = function["function_name"] + ".py"
            assembled_function_files.append(function_filename)
            with open(f"{os.path.join(self.work_dir, function_filename)}", "r") as f:
                code = f.read()
            self.completed_functions.append({"function_filename": function_filename, "function_code": code})
        
        return assembled_function_files

    def assemble_functions(self, module_name:str) -> None:
        assembled_function_files = self.gather_functions()
        prompt = [
            f"Now your Development Groups have finished the assigned functions:\n ",
            f"{self.completed_functions}\n",
            f"You are expected to assemble the final codes and provide them to the Module Leader.\n"
            f"These function files are in the current directory. \n",
            f"Import those sub-functions from the file and call them in the main function.\n",
            f"Please provide the final code in the following format:\n",
            f"```LANGUAGE\n'''\nDOCSTRING\n'''\nCODE\n```\n"
        ]
        prompt = {"role": "user",
                    "content": "".join(prompt)}
        self.combine_prompt(prompt)
        response = self.chat().strip()
        if re.findall(r"```(.*?)\n(.*?)```", response, re.DOTALL):
            code = re.findall(r"```(.*?)\n(.*?)```", response, re.DOTALL)[0][1]
        else:
            code = response
        code = code.strip()
        self.combine_prompt({"role": "assistant", "content": f"{code}"})
        self.role["code"] = code
        self.role["filename"] = module_name + ".py"
        self.role["assembled_functions"] = {"module_function":self.role["filename"], "sub_functions":assembled_function_files}

    def fix_module(self, module_name:str, error_message:str) -> None:
        module_file = os.path.join(self.work_dir, self.role["filename"])
        with open(module_file, "r") as f:
            module_code = f.read()
        prompt = [
            f"Your test code for the {module_name} module has failed.\n",
            f"Here is the error message:\n",
            f"{error_message}\n",
            f"You might need to fix the original code: \n",
            f"{module_code}\n",
            f"Please provide the fixed code in the following format:\n",
            f"```LANGUAGE\n'''\nDOCSTRING\n'''\nCODE\n```\n",
            f"Return code only."
        ]
        prompt = {"role": "user",
                  "content": "".join(prompt)}
        self.combine_prompt(prompt)
        response = self.chat().strip()
        if re.findall(r"```(.*?)\n(.*?)```", response, re.DOTALL):
            fixed_code = re.findall(r"```(.*?)\n(.*?)```", response, re.DOTALL)[0][1]
        else:
            fixed_code = response
        fixed_code = fixed_code.strip()
        self.combine_prompt({"role": "assistant", "content": f"{fixed_code}"})
        self.role["code"] = fixed_code



if __name__ == "__main__":
    leader = TeamLeader()
    prompt = "Given an image path, write a Python function that reads the image and converts it to grayscale."
    response = leader.split_job(prompt)