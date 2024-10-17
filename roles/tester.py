import os
import re
import docker

from base_agent import BaseAgent


class Tester(BaseAgent):
    def __init__(self, agent_name:str="Tester", model_id:str="deepseek-ai/deepseek-coder-7b-instruct-v1.5", quantization:bool=False, **kwarg) -> None:
        sys_prompt = [
            "You are a software test engineer who is working on a project.",
            "You will receive a project requirement and a segment of code from user and you need to write a test code to test the input code. ",
            ]
        super().__init__(model_id=model_id, sys_prompt="".join(sys_prompt), quantization=quantization, agent_name=agent_name, **kwarg)
    
    def coding(self, input_prompts:str, filename:str) -> None:
        input_prompts = "".join(input_prompts)
        prompt = {"role": "user",
                  "content": f"{input_prompts}"}
        # integrate prompt with previous messages
        self.combine_prompt(prompt)
        # generate response
        response = self.chat()
        response = response.rstrip()
        # extract code from response delimited by "<s> " and </s>
        # # codellama: extract code from response delimited by "<s> " and </s>
        # test_code = response.split("[/INST] ")[-1].split("</s>")[0]
        # Deepseek: extract code from response after req
        test_code = response.split(input_prompts)[-1].split("<|EOT|>")[0].strip()

        if re.findall(r"```(.*?)```", test_code, re.DOTALL):
            test_code = re.findall(r"```(.*?)\n(.*?)```", test_code, re.DOTALL)[0][1]
        else:
            test_code = test_code
        test_code = test_code.strip()
        
        test_filename = f"test_{filename}"
        self.role["code"] = test_code
        self.role["filename"] = test_filename
        self.combine_prompt({"role": "assistant", "content": f"{test_code}"})

    def run_test(self) -> str:
        client = docker.from_env()
        container = client.containers.run("image:1.0",
                                        command=["python", "-uB", f"/work/{self.role['filename']}"],
                                        detach=True,
                                        volumes={os.path.abspath(self.work_dir): {"bind": "/work", "mode": "rw"}},
                                        working_dir="/work")
        # show container print
        for line in container.logs(stream=True):
            line.strip()
        
        info = container.logs().decode("utf-8")
        container.stop()
        container.remove()
        print(info)
        return info

    def gen_test_code(self, requirements:str, filename:str) -> None:
        self.coding(requirements, filename)
        self.dump_files()

    def get_filename(self) -> str:
        assert "filename" in self.role, "Filename is not generated yet."
        return self.role["filename"]

    def get_code(self) -> str:
        assert "code" in self.role, "Code is not generated yet."
        return self.role["code"]