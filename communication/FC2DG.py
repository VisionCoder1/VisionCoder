import roles
from pathlib import Path
from base_communication import BaseCommunication

class FC2DG(BaseCommunication):
    def __init__(self, work_dir:Path) -> None:
        self.work_dir = work_dir
        self.function_list = []
        self.fc_list = []
        self.project = None
        super().__init__()

    def set_FC(self) -> None:
        self.module_list = self.proj_info["TL_modules"]
        for i in range(len(self.module_list)):
            fc_name = f"FC_{self.module_list[i]['module_name']}"
            self.fc_list.append(fc_name)

    def FC_refine_function(self) -> None:
        self.project = self.proj_info["project"]
        for i in range(len(self.fc_list)):
            module_name = self.module_list[i]["module_name"]
            coding_language = self.module_list[i]["coding_language"]
            module_description = self.module_list[i]["module_description"]

            function_coordinator = roles.FunctionCoordinator(agent_name=self.fc_list[i], model_id="gpt-4o-mini", work_dir=self.work_dir)
            init_function_list = self.proj_info[f"{module_name}_functions"]
            print(f"FC2DG: {self.fc_list[i]} initialized.")
            print(f"FC2DG: {module_name} Coordinator refining functions...")
            function_coordinator.refine_function(self.project, self.module_list, module_name, coding_language, module_description, init_function_list)
            print(f"{function_coordinator.role['modules']}")
            function_coordinator.dump_files()
            self.insert_proj_info(f"{module_name}_functions_refined", function_coordinator.role["modules"])

    def FC_assemble_function(self) -> None:
        for i in range(len(self.fc_list)):
            module_name = self.module_list[i]["module_name"]

            function_coordinator = roles.FunctionCoordinator(agent_name=self.fc_list[i], model_id="gpt-4o-mini", work_dir=self.work_dir)
            print(f"FC2DG: {module_name} Coordinator assembling functions...")
            function_coordinator.assemble_functions(module_name)
            function_coordinator.dump_files()
            self.insert_proj_info(f"{module_name}_functions_assembled", function_coordinator.role["assembled_functions"])
    
    def FC_fix_assemble(self) -> None:
        self.load_proj_info()
        for i in range(len(self.fc_list)):
            module_name = self.module_list[i]["module_name"]
            test_result = self.proj_info[f"{module_name}_test"]["validity"]

            if not test_result:
                function_coordinator = roles.FunctionCoordinator(agent_name=self.fc_list[i], model_id="gpt-4o-mini", work_dir=self.work_dir)

                error_message = self.proj_info[f"{module_name}_test"]["info"]
                print(f"ML2FC: {module_name} Leader fixing module...")
                function_coordinator.fix_module(module_name, error_message)
                function_coordinator.dump_files()