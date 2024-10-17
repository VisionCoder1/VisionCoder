import os
import re

from base_agent import BaseAgent


class Coder(BaseAgent):
    def __init__(self, agent_name:str="Coder", model_id:str="deepseek-ai/deepseek-coder-7b-instruct-v1.5", quantization:bool=False, **kwarg) -> None:
        sys_prompt = [
            "You are a software engineer who is working on a project.\n",
            "You will receive a prompt from user and you need to write a code to fulfill the requirements.\n",
            "Note that your code will be tested by a tester and you need to hand in the code result to your leader."
        ]
        super().__init__(model_id=model_id, sys_prompt=''.join(sys_prompt), quantization=quantization, agent_name=agent_name, **kwarg)

    def coding(self, req:str, store_memory:bool=True) -> None:
        req = req + "\nReturn code only."
        prompt = {"role": "user",
                    "content": f"{req}"}
        # integrate prompt with previous messages
        self.combine_prompt(prompt, store_memory)
        # generate response
        response = self.chat()
        response = response.strip()
        # # codellama: extract code from response delimited by "<s> " and </s>
        # code = response.split("[/INST] ")[-1].split("</s>")[0]
        # Deepseek: extract code from response after req
        code = response.split("<|EOT|>")[0].strip()
        if re.findall(r"```(.*?)\n(.*?)```", code, re.DOTALL):
            code = re.findall(r"```(.*?)\n(.*?)```", code, re.DOTALL)[0][1]
        else:
            code = code
        code = code.strip()
        self.combine_prompt({"role": "assistant", "content": f"{code}"}, store_memory)
        self.role["code"] = code
    
    def set_filename(self, filename:str) -> None:
        self.role["filename"] = filename

    def get_filename(self) -> str:
        assert "filename" in self.role, "Filename is not generated yet."
        return self.role["filename"]

    def get_code(self) -> str:
        assert "code" in self.role, "Code is not generated yet."
        return self.role["code"]
    
    def refine_from_result(self, running_result:str) -> None:
        prompt = [
            f"Forget the test code, focus on your original code:\n",
            f"{self.role['code']}\n",
            f"Here is the execution result: {running_result.rstrip()}\n\n",
            f"Refine your original code based on the result.\n",
            # f"following the format:```LANGUAGE\n'''\nDOCSTRING\n'''\nCODE\n```"
        ]
        self.coding("".join(prompt))

    def gen_init_code(self, prompt:str, filename:str) -> None:
        self.coding(prompt)
        self.set_filename(filename)
        self.dump_files()
    
    def refine_code(self, running_result:str) -> None:
        self.refine_from_result(running_result)
        self.dump_files()



if __name__ == "__main__":
    llama_inst = Coder(work_dir="./test")
    prompt = "Write a Python function that takes a list of integers as input and returns the sum of all the integers."
    llama_inst.coding(prompt)
    llama_inst.gen_filename()
    llama_inst.dump_files()
    llama_inst.release()

