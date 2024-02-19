import matplotlib.pyplot as plt
import numpy as np
import sys



def read_blowup(file):
    blowup_redshift = -2
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
            else:
                words = line.split()
                if "inf" in words[1] or "nan" in words[1] or abs(float(words[1])) >= 1:
                    blowup_redshift = float(words[0])
                    break
    return blowup_redshift

            
          

def read_potentials(file):
    z,phi,psi = [],[],[]
    with open(file, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            else:
                words = line.split()
                z.append(float(words[0]))
                phi.append(float(words[1]))
                psi.append(float(words[2]))
    return np.array(z), np.array(phi),np.array(psi)



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
            plt.scatter(z,max_pi,label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color,marker = marker,s=s)
        elif y == "avg_zeta":
            plt.scatter(z,abs(avg_zeta),label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color,marker = marker,s=s)
        elif y == "max_zeta":
            plt.scatter(z,max_zeta,label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color,marker=marker, s=s)
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
            plt.scatter(z,max_pi,label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color)
        elif y == "avg_zeta":
            plt.scatter(z,abs(avg_zeta),label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color)
        elif y == "max_zeta":
            plt.scatter(z,max_zeta,label= implementation+ " " + y + ", N_kess = " +str(int(N_kessence)),color=color)
        else:
            sys.exit(1)



zb = read_blowup("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig8/not_sourcing_gravity/Ngrid_32/div_variables.txt")
print(zb)
plt.scatter(32**3,1+zb)

zb = read_blowup("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig8/sourcing_gravity/Ngrid_32/div_variables.txt")
print(zb)
plt.scatter(32**3,1+zb,label="sourcing",marker="x",s=80)

zb = read_blowup("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig8/not_sourcing_gravity/Ngrid_64/div_variables.txt")
print(zb)
plt.scatter(64**3,1+zb)

zb = read_blowup("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig8/sourcing_gravity/Ngrid_64/div_variables.txt")
print(zb)
plt.scatter(64**3,1+zb,label="source",marker="x",s=80)

zb = read_blowup("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig8/not_sourcing_gravity/Ngrid_128/div_variables.txt")
print(zb)
plt.scatter(128**3,1+zb)

zb = read_blowup("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig8/sourcing_gravity/Ngrid_128/div_variables.txt")
print(zb)
plt.scatter(128**3,1+zb,label="source",marker="x",s=80)

zb = read_blowup("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig8/not_sourcing_gravity/Ngrid_256/div_variables.txt")
print(zb)
plt.scatter(256**3,1+zb)

zb = read_blowup("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig8/sourcing_gravity/Ngrid_256/div_variables.txt")
print(zb)
plt.scatter(256**3,1+zb,label="source",marker="x",s=80)

zb = read_blowup("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig8/not_sourcing_gravity/Ngrid_512/div_variables.txt")
print(zb)
plt.scatter(512**3,1+zb)

zb = read_blowup("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig8/sourcing_gravity/Ngrid_512/div_variables.txt")
print(zb)
plt.scatter(512**3,1+zb,label="source",marker="x",s=80)

#plt.legend()
plt.yscale("log")
plt.xlim(1e5,5e8)
plt.xscale("log")
plt.ylabel("1+z_b")
plt.grid()
plt.xlabel("N_grid")
plt.show()


sys.exit(0)
plt.title("Relative change",size=15)
###z,phi,psi = read_potentials("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_potentials/sourcing/potentials.txt")
###plt.scatter(z,phi,label="phi_sourcing",color = "mediumblue",marker = "x",s=70)
###plt.scatter(z,psi,label="psi_sourcing",color = "orange",marker = "+",s=80)


z,phi,psi = read_potentials("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_potentials/potentials.txt")
plt.scatter(z,phi,label="phi",color = "mediumblue")
plt.scatter(z,psi,label="psi",color = "orange",s=8)

###plt.plot([100,0],[0.1,0.1],color = "red",label = "10% relative change",linestyle="dashed")

plt.gca().invert_xaxis()
plt.legend()
###plt.yscale("log")
#plt.ylabel("%")
###plt.xlim(20,0)
plt.xlabel("z")
plt.show()
sys.exit(0)





file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_not_source/N1/div_variables.txt"
plot(file,"max_pi","new","mediumblue")

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_not_source/N5/div_variables.txt"
plot(file,"max_pi","new","orange")

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_not_source/N10/div_variables.txt"
plot(file,"max_pi","new","seagreen")

# old impl
file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_test_not_source/N1/div_variables.txt"
plot(file,"max_pi","old","mediumblue",marker = "x",s=70)

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_test_not_source/N5/div_variables.txt"
plot(file,"max_pi","old","orange",marker = "x",s=70)

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_test_not_source/N10/div_variables.txt"
plot(file,"max_pi","old","seagreen",marker = "x",s=70)


#file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002/N1/div_variables.txt"
#plot(file,"avg_zeta","new","mediumblue")
#
#file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002/N5/div_variables.txt"
#plot(file,"avg_zeta","new","orange")
#
#file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002/N10/div_variables.txt"
#plot(file,"avg_zeta","new","seagreen")
#
## old impl
#file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_test/N1/div_variables.txt"
#plot(file,"avg_zeta","old","mediumblue",marker = "x",s=70)
#
#file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_test/N5/div_variables.txt"
#plot(file,"avg_zeta","old","orange",marker = "x",s=70)
#
#file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002_test/N10/div_variables.txt"
#plot(file,"avg_zeta","old","seagreen",marker = "x",s=70)


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

