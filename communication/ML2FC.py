import roles
from pathlib import Path
from base_communication import BaseCommunication
from typing import List

class ML2FC(BaseCommunication):
    def __init__(self, work_dir:Path) -> None:
        self.work_dir = work_dir
        self.module_list = []
        self.ml_list = []
        self.project = None
        super().__init__()

    def set_ML(self) -> None:
        self.module_list = self.proj_info["TL_modules"]
        for i in range(len(self.module_list)):
            ml_name = f"ML_{self.module_list[i]['module_name']}"
            self.ml_list.append(ml_name)

    def ML_split_job(self) -> None:
        self.project = self.proj_info["project"]
        for i in range(len(self.ml_list)):
            module_leader_name = self.ml_list[i]
            module_name = self.module_list[i]["module_name"]
            coding_language = self.module_list[i]["coding_language"]
            module_description = self.module_list[i]["module_description"]

            module_leader = roles.ModuleLeader(agent_name=module_leader_name, model_id= "gpt-4o-mini", work_dir=self.work_dir)
            print(f"ML2FC: {module_leader_name} initialized.")
            print(f"ML2FC: {module_name} Leader splitting job...")
            module_leader.split_function(self.project, self.module_list, module_name, coding_language, module_description)
            print(f"{module_leader.role['modules']}")
            module_leader.dump_files()
            self.insert_proj_info(f"{module_name}_functions", module_leader.role["modules"])
        
    def ML_test_module(self) -> bool:
        module_test_results = []
        for i in range(len(self.ml_list)):
            module_name = self.module_list[i]["module_name"]

            module_leader = roles.ModuleLeader(agent_name=self.ml_list[i], model_id="gpt-4o-mini", work_dir=self.work_dir)
            print(f"ML2FC: {module_name} Leader testing module...")
            test_filename = module_leader.gen_module_test_file(module_name)
            module_leader.dump_files()

            module_test_info = self.run_test(test_filename)
            print(module_test_info)
            validity = False if ("FAILED" or "Error") in module_test_info else True
            module_test_results.append(validity)
            module_test = {
                "validity": validity,
                "info": module_test_info,
            }
            self.insert_proj_info(f"{module_name}_test", module_test)
        
        if all(module_test_results):
            return True
        else:
            return False
