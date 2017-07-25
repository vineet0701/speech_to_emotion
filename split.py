from pydub import AudioSegment
import math
import re

def split():
    audio = AudioSegment.from_file('../audio.wav')

    directory = "../data/"
    lines = list(open('../file.txt'))
    segment_list = []
    for line in lines:
        col = line.split('\t')
        seg = [int(round(float(col[0]))), int(round(float(col[1]))), col[2]]
        segment_list.append(seg)

    for segment in segment_list:
        #print start, end

        start = segment[0]*1000
        end = segment[1]*1000
        tag = segment[2]
        segment = audio[start:end]
        segment.export(directory + "/" + tag.strip('\n') + "/" + str(start) + ".wav", format="wav")


def tagData():
    lines = list(open('../data.txt'))
    logs = []
    for line in lines:
        if line != "" or line != "\n":
            try:
                line = line.strip('"')
                sections = line.split(' ')
                print sections
                start = math.floor(float(sections[0]))
                end = math.ceil(float(sections[1]))
                tag = sections[3].split(',')[0]
                processedLine = str(start) + "\t" + str(end) + "\t" + tag
                logs.append(processedLine)
            except:
                print "Skipping"
    with open('../file.txt' , 'wb') as f:
        for line in logs:
            f.write(line + "\n")

if __name__ == '__main__':
    split()