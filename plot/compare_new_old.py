import matplotlib.pyplot as plt
import numpy as np
import sys


def read_div_variables(file):
    #cs2_kessence = []
    #N_kessence = []
    z,avg_pi,max_pi,avg_zeta,max_zeta = [],[],[],[],[]
    with open(file, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            elif line.startswith("cs2_kessence"):
                words = line.split()
                cs2_kessence=float(words[1])
                #print(cs2_kessence[-1])
                #print(str(i)+ " cs2 added")
                continue
            elif line.startswith("N_kessence"):
                words = line.split()
                N_kessence=float(words[1])
                #print(cs2_kessence[-1])
                #print(str(i)+ " cs2 added")
                continue
            elif line.startswith("100"):
                #words = line.split()
                #N_kessence=float(words[1])
                #print(cs2_kessence[-1])
                #print(str(i)+ " cs2 added")
                continue
            else:
                words = line.split()
                nan = False
                for word in words:
                    if np.isnan(float(word)):
                        nan = True
                if nan:
                    continue
                        
                #(z,avg_pi,max_pi,avg_zeta,max_zeta).append(float(words[0]),float(words[1]),float(words[2]),float(words[3]),float(words[4]))
                z.append(float(words[0]))
                #print(float(words[0]))
                avg_pi.append(float(words[1]))
                max_pi.append(float(words[2]))
                avg_zeta.append(float(words[3]))
                max_zeta.append(float(words[4]))
    return np.array(z),np.array(avg_pi),np.array(max_pi),np.array(avg_zeta),np.array(max_zeta), cs2_kessence,N_kessence 


def plot(file,y,implementation,color,marker=False,s=False):
    if marker != False and s != False:
        z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file) #read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002/N1/div_variables.txt")

        if y == "avg_pi":
            plt.scatter(z,abs(avg_pi),label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color,marker=marker,s=s)
        elif y == "max_pi":
            plt.scatter(z,abs(max_pi),label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color,marker = marker,s=s)
        elif y == "avg_zeta":
            plt.scatter(z,abs(avg_zeta),label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color,marker = marker,s=s)
        elif y == "max_zeta":
            plt.scatter(z,abs(max_zeta),label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color,marker=marker, s=s)
        else:
            sys.exit(1)
        #marker = marker
        #s = s
    else:
        z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file) #read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002/N1/div_variables.txt")
        #avg_pi = abs(avg_pi)
        #avg_zeta = abs(avg_zeta)
        #max_pi = abs(max_pi)
        #max_zeta = abs(max_zeta)
        if y == "avg_pi":
            plt.scatter(z,abs(avg_pi),label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color)
        elif y == "max_pi":
            plt.scatter(z,abs(max_pi),label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color)
        elif y == "avg_zeta":
            plt.scatter(z,abs(avg_zeta),label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color)
        elif y == "max_zeta":
            plt.scatter(z,abs(max_zeta),label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color)
        else:
            sys.exit(1)


file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002/N1/div_variables.txt"
plot(file,"avg_pi","new","mediumblue")

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002/N5/div_variables.txt"
plot(file,"avg_pi","new","orange")

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002/N10/div_variables.txt"
plot(file,"avg_pi","new","seagreen")

# old impl
file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_test/N1/div_variables.txt"
plot(file,"avg_pi","old","mediumblue",marker = "x",s=70)

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_test/N5/div_variables.txt"
plot(file,"avg_pi","old","orange",marker = "x",s=70)

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_test/N10/div_variables.txt"
plot(file,"avg_pi","old","seagreen",marker = "x",s=70)


plt.gca().invert_xaxis()
plt.yscale('log')
plt.legend()
plt.xlabel("z")
plt.show()
sys.exit(0)

z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002/N1/div_variables.txt")
plt.scatter(z,abs(avg_pi),label="new avg_pi, " + "N_kess = " +str(int(N_kessence)),color="mediumblue")
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002/N5/div_variables.txt")
plt.scatter(z,abs(avg_pi),label="new avg_pi, " + "N_kess = " +str(int(N_kessence)),color="orange")
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002/N10/div_variables.txt")
plt.scatter(z,abs(avg_pi),label="new avg_pi, " + "N_kess = " +str(int(N_kessence)),color="seagreen")
#plt.scatter(z,avg_zeta)


z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_test/N1/div_variables.txt")
plt.scatter(z,abs(avg_pi),label="old avg_pi, " + "N_kess = " +str(int(N_kessence)),color="mediumblue",marker="x",s=70)
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_test/N5/div_variables.txt")
plt.scatter(z,abs(avg_pi),label="old avg_pi, " + "N_kess = " +str(int(N_kessence)),color="orange",marker="x",s=70)
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_test/N10/div_variables.txt")
plt.scatter(z,abs(avg_pi),label="old avg_pi, " + "N_kess = " +str(int(N_kessence)),color="seagreen",marker="x",s=70)
#plt.scatter(z,avg_zeta)
plt.gca().invert_xaxis()
plt.yscale('log')
plt.legend()
plt.xlabel("z")
plt.show()
#sys.exit(0)



z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/new/N_k_1/div_variables.txt")
plt.scatter(z,abs(avg_zeta),label="new avg_zeta, " + "N_kess = " +str(int(N_kessence)),color="mediumblue")
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/new/N_k_5/div_variables.txt")
plt.scatter(z,abs(avg_zeta),label="new avg_zeta, " + "N_kess = " +str(int(N_kessence)),color="orange")
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/new/N_k_10/div_variables.txt")
plt.scatter(z,abs(avg_zeta),label="new avg_zeta, " + "N_kess = " +str(int(N_kessence)),color="seagreen")
#plt.scatter(z,avg_zeta)


z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/old/N_k_1/div_variables.txt")
plt.scatter(z,abs(avg_zeta),label="old avg_zeta, " + "N_kess = " +str(int(N_kessence)),color="mediumblue",marker="x",s=70)
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/old/N_k_5/div_variables.txt")
plt.scatter(z,abs(avg_zeta),label="old avg_zeta, " + "N_kess = " +str(int(N_kessence)),color="orange",marker="x",s=70)
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/old/N_k_10/div_variables.txt")
plt.scatter(z,abs(avg_zeta),label="old avg_zeta, " + "N_kess = " +str(int(N_kessence)),color="seagreen",marker="x",s=70)
#plt.scatter(z,avg_zeta)


plt.gca().invert_xaxis()
plt.yscale('log')
plt.legend()
plt.xlabel("z")
plt.show()



z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/new/N_k_1/div_variables.txt")
plt.scatter(z,max_pi,label="new max_pi, " + "N_kess = " +str(int(N_kessence)),color="mediumblue")
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/new/N_k_5/div_variables.txt")
plt.scatter(z,max_pi,label="new max_pi, " + "N_kess = " +str(int(N_kessence)),color="orange")
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/new/N_k_10/div_variables.txt")
plt.scatter(z,max_pi,label="new max_pi, " + "N_kess = " +str(int(N_kessence)),color="seagreen")
#plt.scatter(z,avg_zeta)


z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/old/N_k_1/div_variables.txt")
plt.scatter(z,max_pi,label="old max_pi, " + "N_kess = " +str(int(N_kessence)),color="mediumblue",marker="x",s=70)
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/old/N_k_5/div_variables.txt")
plt.scatter(z,max_pi,label="old max_pi, " + "N_kess = " +str(int(N_kessence)),color="orange",marker="x",s=70)
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/old/N_k_10/div_variables.txt")
plt.scatter(z,max_pi,label="old max_pi, " + "N_kess = " +str(int(N_kessence)),color="seagreen",marker="x",s=70)
#plt.scatter(z,avg_zeta)
plt.gca().invert_xaxis()
plt.yscale('log')
plt.xlabel("z")
plt.legend()
plt.show()




z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/new/N_k_1/div_variables.txt")
plt.scatter(z,max_zeta,label="new max_zeta, " + "N_kess = " +str(int(N_kessence)),color="mediumblue")
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/new/N_k_5/div_variables.txt")
plt.scatter(z,max_zeta,label="new max_zeta, " + "N_kess = " +str(int(N_kessence)),color="orange")
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/new/N_k_10/div_variables.txt")
plt.scatter(z,max_zeta,label="new max_zeta, " + "N_kess = " +str(int(N_kessence)),color="seagreen")
#plt.scatter(z,avg_zeta)


z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/old/N_k_1/div_variables.txt")
plt.scatter(z,max_zeta,label="old max_zeta, " + "N_kess = " +str(int(N_kessence)),color="mediumblue",marker="x",s=70)
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/old/N_k_5/div_variables.txt")
plt.scatter(z,max_zeta,label="old max_zeta, " + "N_kess = " +str(int(N_kessence)),color="orange",marker="x",s=70)
#plt.scatter(z,avg_zeta)
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/old/N_k_10/div_variables.txt")
plt.scatter(z,max_zeta,label="old max_zeta, " + "N_kess = " +str(int(N_kessence)),color="seagreen",marker="x",s=70)
#plt.scatter(z,avg_zeta)


plt.gca().invert_xaxis()
plt.yscale('log')
plt.xlabel("z")
plt.legend()
plt.show()

