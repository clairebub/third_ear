Test.py

test: # index page

choose: # choose command page

record_command: # record command page

test record: # test coice command page

testdone: # show the test result

rightcommand: # interacting with people to see whether test result is right or not

record: # record voice command
"""
class_number: index of class of command
label: name of command
count: times to record
"""

label_record: # once a voice recored, put it on a txt to make it easy to find
"""
label: label of command
"""

check_string: # use to check if the command exists to not
"""
string: the label of command to check
"""

ssh_scp_put: # send files to server
"""
host: host
port: port
user: your username of server
password: password of server
local_files: files that you want to send
"""