from cut_img import det_text
import subprocess

cmd1 = "cd ./text"
cmd2 = "../../.local/lib/python3.8 detect.py"
cmd = cmd1 + "&&" + cmd2
res = subprocess.run(cmd, shell = True)
