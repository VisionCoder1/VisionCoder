from base_agent import BaseAgent


class MistralAnalyser(BaseAgent):
    def __init__(self, agent_name:str="Analysor", model_id:str="mistralai/Mistral-7B-Instruct-v0.2", quantization:bool=False, **kwarg) -> None:
        super().__init__(model_id=model_id, sys_prompt=None, quantization=quantization, agent_name=agent_name, **kwarg)

    def analyse(self, code:str, filename:str) -> str:
        prompt = [
            f"You are a software engineer who is working on a project. You will receive a code from your colleague and you need to analyse the code and provide a README.md file with the following information:\n",
            f"- A brief description of the code\n"
            f"- The purpose of the code\n"
            f"- The dependencies of the code\n"
            f"- The input and output of the code\n"
            f"- The expected behaviour of the code\n"
            f"- Any potential issues with the code\n"
            f"- Any additional information that may be useful\n\n",
            f"Please provide the README.md file for the following code:\n",
            f"Filename: {filename}\n",
            f"Code: {code}\n"]
        message = {"role": "user", "content": "".join(prompt)}
        self.combine_prompt(message)
        response = self.chat()
        response = response.rstrip()
        # extract code from response delimited by "<s> " and </s>
        response = response.split("<s> ")[1].split("</s>")[0]
        # extract response after [/INST]
        response = response.split("[/INST] ")[1]

        self.role["code"] = response
        self.role["filename"] = "Readme.md"
        self.combine_prompt({"role": "assistant", "content": f"{response}"})
    
    def gen_analysis(self, code:str, filename:str) -> None:
        self.analyse(code, filename)
        self.dump_files()



if __name__ == "__main__":
    ana_inst = MistralAnalyser(work_dir="./")
    with open("./test/case_normal/sum_integers.py", "r") as f:
        code = f.read()
    ana_inst.gen_analysis(code, "sum_integers.py")




