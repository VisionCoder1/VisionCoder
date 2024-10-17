import roles
from pathlib import Path
from base_communication import BaseCommunication

class TL2ML(BaseCommunication):
    def __init__(self, work_dir:Path) -> None:
        self.work_dir = work_dir
        self.TL = roles.TeamLeader(agent_name="TL", model_id="gpt-4o-mini", work_dir=self.work_dir)
        print("TL2ML: TeamLeader initialized.")
        super().__init__()

    def TL_split_job(self, req:str) -> None:
        self.TL.split_job(req)
        print("TL2ML: Modules split:")
        for i in self.TL.role["modules"]:
            print(i)
        image_name = self.TL.select_env()
        self.TL.dump_files()
        self.insert_proj_info("project", req)
        self.insert_proj_info("work_dir", self.work_dir)
        self.insert_proj_info("image_name", image_name)
        self.insert_proj_info("TL_modules", self.TL.role["modules"])

    def TL_assemble_modules(self) -> None:
        self.TL.assemble_modules()
        print("TL2ML: Modules assembled.")
        self.TL.dump_files()
        self.insert_proj_info("project_main_file", self.TL.role["filename"])