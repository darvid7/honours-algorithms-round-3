import sys
import subprocess
import os

# IDK WHT THIS ISNT WORKING

# python3 d2v_suggested_hyperparams_pv/model.py
# export PYTHONPATH=../../
sys.path.append(os.path.join(os.getcwd(), ".."))

# TODO: log out total time?
# TODO: sanitize relations?
# TODO: change hyperparms so it doesnt take 6 days.
# TODO: I can re-evaluate hit@10 based the logs as I've logged out everything that didn't hit (which was everything)
# in which i've logged the top 15

model_files = [
    #"pte_pv_dbow.py", "pte_pv_dm.py",
     "d2v_suggested_pv_dm.py", "d2v_suggested_pv_dbow.py"]
for model_file in model_files:
    cmd_args = ["nohup", "python3", model_file]
    fh_log = open("../../../honours-data-round-2/FB15K/model_out/running/%s.log.txt" % model_file, "w")
    fh_err = open("../../../honours-data-round-2/FB15K/model_out/running/%s.err.txt" % model_file, "w")
    # Makes sync as it waits for the child process to finish, might overcome memory issues.
    subprocess.Popen(cmd_args, stderr=fh_err, stdout=fh_log).wait()
    fh_log.close()
    fh_err.close()
    print("Finished executing %s" % model_file)
    sys.stdout.flush()
