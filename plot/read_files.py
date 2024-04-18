import matplotlib.pyplot as plt
import numpy as np
import sys

#arr =[]
#a = np.exp(-20)
#b = 1/9*np.log(np.exp(3)/a)
#for i in range(10):
#    arr.append(a*np.exp(i*b))
#
#
#print(arr)
#print(np.logspace(base=np.exp(1),start=-20,stop=3,num=10))
#sys.exit(0)


def read_deta_dt(filename):
    gev  = []
    kess = []
    with open(filename, 'r') as infile:
        for line in infile:
            if line.startswith("#"):
                continue
            else:
                words = line.split()
                if words[0] == "gev":
                    gev.append(float(words[4]))
                if words[0] == "kess":
                    kess.append(float(words[4]))
    return gev, kess




def Omega_func(file):
    Omega_DE = []
    Omega_M = []
    #Omega_b = []
    Omega_Rad = []
    a = []
    z = []
    with open(file, 'r') as infile:
        for line in infile:
            if line.startswith("#"):
                continue
            else:
                words = line.split()
                a.append(float(words[0]))
                z.append(float(words[1]))
                Omega_DE.append(float(words[2]))
                Omega_M.append(float(words[3])+float(words[4]))
                #Omega_b.append(float(words[4]))
                Omega_Rad.append(float(words[5])+float(words[6]))
    return np.array(a), np.array(z), np.array(Omega_DE), np.array(Omega_M), np.array(Omega_Rad)


# reading rho_i/rho_crit_0
def read_DE_a(file):
    a = []
    z = []
    DE = []
    M = []
    Rad = []
    with open(file,"r") as infile:
        for line in infile:
            if line.startswith("#"):
                continue
            else:
                words = line.split()
                a.append(float(words[0]))
                z.append(float(words[1]))
                DE.append(float(words[2]))
                M.append(float(words[3]))
                Rad.append(float(words[4]))
    return np.array(a), np.array(z), np.array(DE), np.array(M), np.array(Rad)

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


print("plotting...")
file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/new/N1/div_variables.txt"
plot(file,"avg_pi","new","mediumblue")

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/new/N5/div_variables.txt"
plot(file,"avg_pi","new","orange")

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/new/N10/div_variables.txt"
plot(file,"avg_pi","new","seagreen")

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/old/N1/div_variables.txt"
plot(file,"avg_pi","old","mediumblue",marker = "x",s=70)

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/old/N5/div_variables.txt"
plot(file,"avg_pi","old","orange",marker = "x",s=70)

file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/old/N10/div_variables.txt"
plot(file,"avg_pi","old","seagreen",marker = "x",s=70)

###file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/new/N1/div_variables.txt"
###plot(file,"max_zeta","new","mediumblue")
###
###file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/new/N5/div_variables.txt"
###plot(file,"max_zeta","new","orange")
###
###file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/new/N10/div_variables.txt"
###plot(file,"max_zeta","new","seagreen")
###
###file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/old/N1/div_variables.txt"
###plot(file,"max_zeta","old","mediumblue",marker = "x",s=70)
###
###file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/old/N5/div_variables.txt"
###plot(file,"max_zeta","old","orange",marker = "x",s=70)
###
###file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/old/N10/div_variables.txt"
###plot(file,"max_zeta","old","seagreen",marker = "x",s=70)

print("Done plotting")
plt.gca().invert_xaxis()
#plt.xlim(20,100)
#plt.ylim(-10,100)
plt.legend()
plt.yscale('log')
plt.show()





def CLASS_file(file):
    z = []
    #z = []
    #DE = []
    #M = []
    #Rad = []
    with open(file,"r") as infile:
        for line in infile:
            if line.startswith("#"):
                continue
            else:
                words = line.split()
                z.append(float(words[0]))
                #z.append(float(words[1]))
                #DE.append(float(words[2]))
                #M.append(float(words[3]))
                #Rad.append(float(words[4]))
    return np.array(z)#, np.array(z), np.array(DE), np.array(M), np.array(Rad)




z = CLASS_file("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/hiclass_tests/test1/class_background.dat")
print(z)
sys.exit(0)





#def Omega_from_friedmann(a,w,Omega_0,rho_crit):
#    Omega = 
#    return np.array(a),np.array(Omega)


def equality(time,a,b): # same length of all lists
    for i in range(len(time)-1):
        diff_old = a[i] - b[i] 
        if ((a[i+1] - b[i+1])/diff_old < 0):
            return (time[i+1] + time[i])/2


a, z, DE, M ,Rad = read_DE_a("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/coincidence_problem/w_1/rho_i_rho_crit_0.txt")
#print(equality(a,DE,M))
#sys.exit(0)
plt.title(r"$\rho_i/\rho_\mathrm{crit,0} \ , \quad c_s^2=1,\ w\approx -1$")
plt.semilogy(np.log(a),DE,label="DE")
plt.semilogy(np.log(a),Rad,label="Radiation")
plt.semilogy(np.log(a),M,label="Matter")
plt.semilogy(np.log(np.ones(2)*equality(a,DE,M)),np.array([1e-10,1e+30]),"--",label="Cosmic time = 10.3 [Gyr]",color="k")
plt.xlabel("ln a")
plt.legend()
plt.xlim(-15,0.1)
plt.ylim(1e-5,1e+23)
plt.savefig("/uio/hume/student-u23/jorgeagl/src/master/master_project/rand_figs/coincidence_w1.pdf")
#plt.gca().invert_xaxis()
plt.show()
#sys.exit(0)



a, z, Omega_DE, Omega_M, Omega_Rad = Omega_func("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/coincidence_problem/w_0/Omega.txt")
sum = np.zeros(len(a))
#sum[:] = Omega_DE[:]+ Omega_CDM[:]+ Omega_b[:]+ Omega_g[:]
plt.title(r"$\Omega_i \ , \quad c_s^2=1,\ w\approx 0$")
plt.plot(np.log(a),Omega_DE,label=r"$\Omega_\mathrm{DE}$")
plt.plot(np.log(a),Omega_Rad,label=r"$\Omega_\mathrm{Rad}$")
plt.plot(np.log(a),Omega_M,label=r"$\Omega_\mathrm{M}$")
#plt.plot(np.log(a),Omega_b,label=r"$\Omega_\mathrm{b}$")
plt.xlabel(r"$\ln (a)$")
#plt.plot(np.log(a),sum)
#plt.plot(z,Omega_DE,label="DE")
#plt.plot(z,Omega_CDM,label="CDM")
#plt.plot(z,Omega_b,label="b")
#plt.plot(z,Omega_g,label="g")

plt.legend()
#plt.gca().invert_xaxis()
plt.xlim(-15,0.1)
#plt.savefig("/uio/hume/student-u23/jorgeagl/src/master/master_project/rand_figs/Omega_w0.pdf")
plt.show()
sys.exit()




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

