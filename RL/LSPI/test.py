file = open("testfile.txt", "w")
x = 1
file.write(str(x)+"\n")
file.write("lul")


file.close()

file = open("testfile.txt", "r")
x = file.readline()
y = file.readline()
print(x)
print(y)