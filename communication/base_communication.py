import json
from pathlib import Path
import os
import shutil
import docker

class BaseCommunication(object):
    def __init__(self) -> None:
        self.proj_info_file = os.path.join(self.work_dir, "project_info.json")
        if os.path.exists(self.proj_info_file):
            self.load_proj_info()
        else:
            self.proj_info = {}
            open(self.proj_info_file, 'w').close()
    
    def load_proj_info(self) -> None:
        with open(self.proj_info_file, 'r') as f:
            if os.stat(self.proj_info_file).st_size == 0:
                self.proj_info = {}
            else:
                self.proj_info = json.load(f)
    
    def dump_proj_info(self) -> None:
        with open(self.proj_info_file, 'w') as f:
            json.dump(self.proj_info, f)

    def insert_proj_info(self, key:str, value) -> None:
        self.load_proj_info()
        self.proj_info[key] = value
        self.dump_proj_info()

    def run_test(self, test_filename:str) -> str:
        image_name = self.proj_info["image_name"]
        client = docker.from_env()
        container = client.containers.run(image=image_name,
                                        command=["python", "-uB", f"/work/{test_filename}"],
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