import sys
import os
import time
import shutil
import thread
import pyaudio
import wave
import socket, select
import paramiko
from datetime import datetime
from flask import Flask, redirect, render_template, \
     request, url_for

app = Flask(__name__)

audio = pyaudio.PyAudio()
counter = 0
host = ''
port = 5021
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(None)

while 1:
    try:
        s.connect((host, port))
        print "Connected to remote host"
        break
    except Exception as e:
        print e
        print('Unable to connect')


@app.route("/" , methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        select = str(request.form.get('command'))
        if select is not None:
            if select == 'Add a new command':
                print "add a new one"
                return redirect(url_for('choose'))
            if select == 'Test exist commands':
                print "Test one"
                return redirect(url_for('testrecord'))
            print "command exists"
    return render_template('index.html')


@app.route("/choose_command" , methods=['GET', 'POST'])
def choose():
    exist = 0
    if request.method == 'POST':
        select = str(request.form.get('command'))
        if select is not None:
            if select not in open('label_record.txt', 'r').read():
                class_number = label_record(select)
                label_input = ''.join(select.split())
                print label_input
                return redirect(url_for('record_command', class_number=class_number, label=label_input))
            print "command exists"
            exist = 1
            return render_template('command_choose.html', exist = exist)
    return render_template('command_choose.html', exist = exist)


@app.route("/record/<class_number>/<label>" , methods=['GET', 'POST'])
def record_command(class_number, label):
    global counter, s
    # scp_host = ''
    # scp_port = 22
    # scp_user = ''
    # scp_password = ''
    print "receive label", class_number, label
    if counter < 5:
        if request.method == 'POST':
            record(class_number=class_number, label=label, count=counter)
            counter += 1
    if counter >= 5:
        #local_file = ['%s_%s_%s.wav' % (class_number, label, str(i)) for i in range(5)]

        #ssh_scp_put(scp_host, scp_port, scp_user, scp_password, local_file)
        # connect to remote host
        print('Connected to remote host. Start sending files and prepare to train')
        msg = "train"
        s.send(msg)
        while 1:
            train_msg = s.recv(4096)
            if train_msg == "train ok":
                break
        print "You can test now!"
        counter = 0
        return redirect(url_for('testrecord'))
    return render_template('record.html', counter=counter)

@app.route("/testrecord" , methods=['GET', 'POST'])
def testrecord():
    global s
    # local_file = ['%s_%s_%s.wav' % (class_number, label, str(i)) for i in range(5)]
    # ssh_scp_put(scp_host, scp_port, scp_user, scp_password, local_file)
    # connect to remote host
    # print('Connected to remote host. Start sending files and prepare to train')
    # msg = "train"
    # s.send(msg)
    # while 1:
    #     train_msg = s.recv(4096)
    #     if train_msg == "train ok":
    #         break
    # print "You can test now!"
    if request.method == 'POST':
        # scp_host = '105.145.88.51'
        # scp_port = 22
        # scp_user = 'shane.z'
        # scp_password = 'zengxy789'
        record()
        # testfile = 'testfile.wav'
        # ssh_scp_put(scp_host, scp_port, scp_user, scp_password, [testfile])
        test_msg = "test prepared"
        s.send(test_msg)
        while 1:
            try:
                test_done_msg = s.recv(4096)
                if test_done_msg == "test done":
                    print test_done_msg
                    prediction = s.recv(4096)
                    print prediction
                    return redirect(url_for('testdone', prediction = prediction))
                    break
            except KeyboardInterrupt:
                break
        print "Done"
    return render_template('test.html')

@app.route("/testdone/<prediction>" , methods=['GET', 'POST'])
def testdone(prediction):
    global s
    result = check_string(prediction)
    print result
    command = result.split('\t')[1]
    class_number = result.split('\t')[0]
    print class_number, command
    if request.method == 'POST':
        # select = str(request.form.get('command'))
        select_result = str(request.form.get('right_if'))
        if select_result == 'Yes':
            s.send('change name')
            s.send(class_number + '_' + ''.join(command.split()) + '_' + ''.join(str(datetime.now()).split()))
            return redirect(url_for('testrecord'))
        if select_result == 'No':
            return redirect(url_for('rightcommand'))
        print "Done"
    return render_template('testdone.html', prediction = command)


@app.route("/rightcommand" , methods=['GET', 'POST'])
def rightcommand():
    global s
    right = 1
    if request.method == 'POST':
        select = str(request.form.get('right_command'))
        print select
        result = check_string(select, file='label_record.txt')
        if result is None:
            print "You didn't record this command"
            right = 0
            return render_template('right_command.html', right = right)
        command = result.split('\t')[1]
        class_number = result.split('\t')[0]
        s.send('change name')
        s.send(class_number + '_' + ''.join(command.split()) + '_' + ''.join(str(datetime.now()).split()))
        return redirect(url_for('testrecord'))
    return render_template('right_command.html', right = right)


def input_thread(L):
    raw_input()
    L.append(None)


def record(class_number=None, label=None, count=0):
    """
        Records audio until key pressed, then saves to file
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 2
    global counter

    print "Recording... for 3 seconds"
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    L = []
    thread.start_new_thread(input_thread, (L,))
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print "Stopped recording."

    # save the audio data
    if not os.path.exists("commands"):
        os.makedirs("commands")
    if class_number and label:
        wf = wave.open("commands/%s_%s_%s.wav" % (class_number, label, count), 'wb')
    else:
        wf = wave.open('commands/testfile.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print "Recording saved."


def label_record(label):
    try:
        f = open("label.txt", 'r')
        for line in f:
            if label in line:
                class_number = str(line.split('\t')[0])
                #class_number = str(int(linelist[index].split('\t')[0]))
                print "class_number", class_number
                break
    except Exception as e:
        print e

    f_record = open("label_record.txt", 'a')
    f_record.write(class_number)
    f_record.write('\t')
    f_record.write(label)
    f_record.write('\n')
    f_record.close()
    return class_number


def check_string(string, file='label.txt'):
    result = None
    with open(file) as f:
        found = False
        for line in f:  #iterate over the file one line at a time(memory efficient)
            #if re.search("\b{0}\b".format(string), line):    #if string found is in current line then print it
            if string in line:
                # print line.split()[1]
                result = line
                found = True
                break
        if not found:
            print('The command cannot be found!')
        return result

def ssh_scp_put(host, port, user, password, local_files):
    paramiko.util.log_to_file("filename.log")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, user, password)
    sftp = paramiko.SFTPClient.from_transport(ssh.get_transport())
    sftp = ssh.open_sftp()
    for file in local_files:
        local_file = 'C:/Users/shane.z/PycharmProjects/untitled1/commands/' + file
        remote_file = '/SAL_Extended/shane.z/xiangyu/work/dark/data/reddots/commands/' + file
        sftp.put(local_file, remote_file)




if __name__ == "__main__":
    # scp_user = 'shane.z'
    # scp_host = '105.145.88.51'
    # scp_port = 22
    # scp_password = 'zengxy789'

    # host = '105.145.88.62'
    # port = 5021
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.settimeout(None)
    # try:
    #     s.connect((host, port))
    #     print "Connected to remote host"
    # except:
    #     print('Unable to connect')
    #     sys.exit()

    app.run(debug=True)
    #choice_input = raw_input("please choose mode: 1.add a new command 2.test exist command \n")

    # if choice_input == '1':
    #     train(s, scp_host, scp_port, scp_user, scp_password)
    #
    # if choice_input == '2':
    #     while 1:
    #         try:
    #             test(s, scp_host, scp_port, scp_user, scp_password)
    #         except KeyboardInterrupt:
    #             break
