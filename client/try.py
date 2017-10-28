import socket, sys
action_host = '192.168.1.21'
action_port = 50008
action_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
action_s.settimeout(None)
while 1:
    try:
        action_s.connect((action_host, action_port))
        print "Connected to action host"
        break
    except Exception as e:
        print e
        sys.exit()
        print('Unable to connect action host')
import json
 #{"request_id": "2209", "action_id": "qeitsxrhho", "timestamp": 1501281162, "intent": {"name": 201, "slots": {"type": 05, "params":[7]}}}
command_try = {}
command_try['request_id'] = '2209'
command_try['action_id'] = 'qeitsxrhho'
command_try['timestamp'] = 1501281162
command_try['intent'] = {}
command_try['intent']['name'] = 204
command_try['intent']['slots'] = {}
command_try['intent']['slots']['type'] = 01
command_try['intent']['slots']['params'] = ['maps']
data = json.dumps(command_try)
print data
action_s.send(data)


# def train(s, scp_host, scp_port, scp_user, scp_password, selected_command):
#     #label_input = raw_input("input the class you want to add: \n")
#     label_input = selected_command
#     # while label_input in open('label_record.txt', 'r').read():
#     #     print "Your command exists, please add a new one! \n"
#     #     try:
#     #         label_input = raw_input("input the class you want to add: \n")
#     #     except KeyboardInterrupt:
#     #         break
#
#     # class_number = label_record(label_input)
#     # label_input = ''.join(label_input.split())
#     # print "record"
#     if not os.path.exists("commands"):
#         os.makedirs("commands")
#     try:
#         while counter < 5:
#             enter = raw_input("Press <Enter> to start recording \n")
#             record(class_number, label_input)
#     except KeyboardInterrupt:
#         pass
#
#     local_file = ['%s_%s_%s.wav' % (class_number, label_input, str(i)) for i in range(5)]
#     ssh_scp_put(scp_host, scp_port, scp_user, scp_password, local_file)
#     # if_train = raw_input("Input 'yes' or <ENTER> to send your commands and train, 'no' to exit \n")
#     # if if_train != 'no':
#     #     # connect to remote host
#     #     print('Connected to remote host. Start sending files and prepare to train')
#     #     msg = "train"
#     #     s.send(msg)
#     #     while 1:
#     #         train_msg = s.recv(4096)
#     #         if train_msg == "train ok":
#     #             break
#     #     print "You can test now!"
#     #     while 1:
#     #         try:
#     #             test(s, scp_host, scp_port, scp_user, scp_password)
#     #         except KeyboardInterrupt:
#     #             break
#     # else:
#     #     print "Please try again!"
#
#
# def test(s, scp_host, scp_port, scp_user, scp_password):
#     #enter = raw_input("Press <Enter> to start recording \n")
#     #record()
#     testfile = 'testfile.wav'
#     ssh_scp_put(scp_host, scp_port, scp_user, scp_password, [testfile])
#     test_msg = "test prepared"
#     s.send(test_msg)
#     while 1:
#         try:
#             test_done_msg = s.recv(4096)
#             if test_done_msg == "test done":
#                 print test_done_msg
#                 prediction = s.recv(4096)
#                 print prediction
#                 break
#         except KeyboardInterrupt:
#             break
#     result = check_string(prediction)
#     command = result.split('\t')[1]
#     class_number = result.split('\t')[0]
#     print "Your command is: ", command
#
#     right_if = raw_input("Is preidiction right? Press 1:yes 2:no \n")
#     while right_if != '1' and right_if != '2':
#         print "please press 1 or 2"
#         right_if = raw_input("Is preidiction right? Press 1:yes 2:no \n")
#     if right_if == '1':
#         print "Thank you for using! \n"
#
#     if right_if == '2':
#         right_class = raw_input("Please enter the right class label for improving, thank you!\n")
#         result = check_string(right_class)
#         while result is None:
#             print "There is not a command you entered, please retry\n"
#             right_class = raw_input("Please enter the right class label for improving, thank you!\n")
#             result = check_string(right_class)
#         command = result.split('\t')[1]
#         class_number = result.split('\t')[0]
#     s.send('change name')
#     print ''.join(str(datetime.now()).split())
#     s.send(class_number + '_' + ''.join(command.split()) + '_' + ''.join(str(datetime.now()).split()))