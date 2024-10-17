# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import roles

# %%
TeamLeader = roles.TeamLeader(agent_name="TL", model_id="gpt-4o", work_dir="./test/case_heirarchy/")
req = "Develop a simple web service that allows a user uploads an image and returns the image in grayscale."
TeamLeader.rm_cache()
response = TeamLeader.split_job(req)
TeamLeader.dump_files()
TeamLeader.release()

# %%
