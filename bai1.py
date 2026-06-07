# bài 1
lines = [line.strip() for line in open("class_scores.txt")]
inside_lines = [line.split(' ') for line in lines]
print("inside lines",inside_lines)
for num in inside_lines:
    num[1]=str(int(num[1])+5)

lines=[" ".join(line) for line in inside_lines]
print("Lines",lines)

f=open('scores2.txt','w')
for line in lines:
    print(line,file=f)

#bài 2
