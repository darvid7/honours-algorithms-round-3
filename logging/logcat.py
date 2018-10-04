import argparse
import subprocess
import sys
import time
import os

commands = {
    "macos": "ps -avx",
    "linux": "ps -aux"
}

run_command = None

if sys.platform == "linux" or sys.platform == "linux2":
    run_command = commands["linux"]
elif sys.platform == "darwin":
    run_command = commands["macos"]
else:
    print("You are using a weird ass platform fam %s" % sys.platform, file=sys.stderr)
    sys.exit(1)

print("Logcat - platform %s" % sys.platform)
print("PID: %s" % os.getpid())

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True, help="Name of command to track")

parser = parser.parse_args()

thing_i_care_about = parser.name

cmd_args = run_command.split()

while True:

    shell_output_string = subprocess.check_output(cmd_args)
    shell_output_string = shell_output_string.decode("utf-8")
    output = shell_output_string.split("\n")

    care_about_indexes = [0]
    for index, line in enumerate(output):
        if thing_i_care_about in line and run_command not in line and "logcat.py -n %s" % thing_i_care_about not in line:
            care_about_indexes.append(index)
            print(run_command)
    output_lines = [output[i] for i in care_about_indexes]

    if len(output_lines) <= 1:
        # Only header, no other matches.
        print("No more matches of %s using %s\nTerminating" % (thing_i_care_about, run_command))
        break

    for o in output_lines:  # Re direct to a log file.
        print(o)

    sys.stdout.flush()
    time.sleep(60 * 10)  # Sleep for 10 mins.
