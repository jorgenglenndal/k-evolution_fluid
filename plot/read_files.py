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

import os
def blowup_redshifts_func(dir_path,dt,Ngrid):
    #temp_blowup = -1
    if Ngrid:
        Ngrid_list =[]
    if (dt == True):
        dt_list = []
    test_list = []
    last_avg_pi = 0
    last_max_zeta = 0
    last_z = -1
    for path, dirnames, filenames in os.walk(dir_path):
        test_list.append(dirnames)
    subdirs = test_list[0]
    for i in range(len(subdirs)):
        
        subdirs[i] = dir_path + subdirs[i] + "/"
    #print(subdirs)
    #sys.exit(0)

    blowup_redshift = []
    cs2_kessence = []
    for i in range(len(subdirs)):
        #if (subdirs[i] == dir_path + "original_imp/"):
        #    continue
        #print(i)
        #if len(cs2_kessence) > len(blowup_redshift): # if no blowup, we do not want sound speed
        #    cs2_kessence.pop()
            #print(str(i)+ " cs2 popped")
            #print("popped")

        if (dt == True):
            with open(subdirs[i] + "file_settings_used.ini", 'r') as file:
                #print(str(subdirs[i]))
                for line in file:
                    if line.startswith("time step limit ="):
                        words = line.split()
                        dt_list.append(float(words[4]))
                        break

        if Ngrid:
            with open(subdirs[i] + "file_settings_used.ini", 'r') as file:
                #print(str(subdirs[i]))
                for line in file:
                    if line.startswith("Ngrid ="):
                        words = line.split()
                        Ngrid_list.append(float(words[2]))
                        break


        with open(subdirs[i] + "div_variables.txt", 'r') as file:
            #print(str(subdirs[i]))
            z_temp = []
            zeta_temp = []
            
            loop_broken = False
            for line in file:
                #if line.startswith("### The blowup criteria are met"):
                #    blowup_redshift.append(temp_blowup)
                #    break

                if line.startswith("#"):
                    continue
                elif line.startswith("cs2_kessence"):
                    words = line.split()
                    cs2_kessence.append(float(words[1]))
                    #print(cs2_kessence[-1])
                    #print(str(i)+ " cs2 added")
                    continue
                elif line.startswith("N_kessence"):
                    continue
                else:
                    words = line.split()
                    #if "inf" in words[-1] or "nan" in words[-1]:

                    #z_temp.append(float(words[0]))
                    #zeta_temp.append(float(words[-1]))
                    #temp_blowup = float(words[0])
                    if abs(float(words[1])) > 1 or "inf" in words[1] or "nan" in words[1]:
                        blowup_redshift.append(float(words[0]))
                        loop_broken = True
                        break

                    #if "inf" in words[1] or "nan" in words[1] or abs(float(words[1])) > 1 or float(words[-1])>1:
                    #    blowup_redshift.append(float(words[0]))
                    #    break
                        
                    last_avg_pi = words[1]
                    last_max_zeta = words[-1]
                    last_z = float(words[0])


                    #if abs(float(words[1])) > 1000:
                    #    blowup_redshift.append(float(words[0]))
                    #    break
            #if "inf" in last_avg_pi or "nan" in last_avg_pi or abs(float(words[1])) > 1 or float(words[-1])>1:
            #            blowup_redshift.append(float(words[0]))
            if (abs(float(last_max_zeta)) > 1 or "inf" in last_max_zeta or "nan" in last_max_zeta) and loop_broken==False:
                print("Blowup not detected in pi")
                blowup_redshift.append(last_z)

            
        if len(cs2_kessence) > len(blowup_redshift): # if no blowup, we do not want sound speed
                print("Blowup did not happen for " +str(cs2_kessence[-1])+". However, last avg_pi is " + str(last_avg_pi))
                cs2_kessence.pop()
                if (dt == True):
                    dt_list.pop()
                if Ngrid:
                    Ngrid_list.pop()

    if (dt == True) and Ngrid:
        return np.array(blowup_redshift),np.array(cs2_kessence), np.array(dt_list), np.array(Ngrid_list)
    if dt and Ngrid==False:
        return blowup_redshift,cs2_kessence, dt_list
    
    if dt==False and Ngrid:
        return blowup_redshift,cs2_kessence, Ngrid_list

    if dt==False and Ngrid==False:
        return blowup_redshift,cs2_kessence
          

def read_potentials(file):
    z,rel_phi,rel_psi,max_phi,max_psi,avg_rel_phi = [],[],[],[],[],[]
    with open(file, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            else:
                words = line.split()
                z.append(float(words[0]))
                rel_phi.append(float(words[1]))
                rel_psi.append(float(words[2]))
                max_phi.append(float(words[3]))
                max_psi.append(float(words[4]))
                avg_rel_phi.append(float(words[5]))
    return np.array(z), np.array(rel_phi),np.array(rel_psi),np.array(max_phi),np.array(max_psi),np.array(avg_rel_phi)



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
            plt.title("$\mathrm{Average}\ \mathcal{H}\pi$")
            plt.scatter(z,abs(avg_pi),label= implementation+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color=color,marker=marker,s=s)
        elif y == "max_pi":
            plt.title("$\mathrm{Max}\ \mathcal{H}\pi$")
            plt.scatter(z,max_pi,label= implementation+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color=color,marker = marker,s=s)
        elif y == "avg_zeta":
            plt.title("$\mathrm{Average}\ \zeta$")
            plt.scatter(z,abs(avg_zeta),label= implementation+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color=color,marker = marker,s=s)
        elif y == "max_zeta":
            plt.title("$\mathrm{Max}\ \zeta$")
            plt.scatter(z,max_zeta,label= implementation+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color=color,marker=marker, s=s)
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
            plt.title("$\mathrm{Average}\ \mathcal{H}\pi$")
            plt.scatter(z,abs(avg_pi),label= implementation+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color=color)
        elif y == "max_pi":
            plt.title("$\mathrm{Max}\ \mathcal{H}\pi$")
            plt.scatter(z,max_pi,label= implementation+ ", "  + r"N$_\mathrm{kess} =$" +str(int(N_kessence)),color=color)
        elif y == "avg_zeta":
            plt.title("$\mathrm{Average}\ \zeta$")
            plt.scatter(z,abs(avg_zeta),label= implementation+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color=color)
        elif y == "max_zeta":
            plt.title("$\mathrm{Max}\ \zeta$")
            plt.scatter(z,max_zeta,label= implementation+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color=color)
        else:
            sys.exit(1)





print("plotting...")
"""
blowup, cs2,dt,Ngrid = blowup_redshifts_func("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig8/source/",dt=True,Ngrid = True)
A = np.vstack([np.log10(Ngrid**3), np.ones(len(np.log10(Ngrid**3)))]).T
c1, c2 = np.linalg.lstsq(A, np.log10(blowup+1), rcond=None)[0]
Ngrid_test = np.array([32,1024])
print(c1,c2)
plt.plot(Ngrid_test**3,Ngrid_test**(3*c1)*10**c2 ,color='grey',linestyle="dashed", label=r'Fitted line: $0.29 log(\mathrm{N}_\mathrm{grid})-1.38=log(1+z_b)$')
plt.scatter(Ngrid**3,1+blowup,label="DE sources gravity",s=100)
#A = np.vstack([np.log10(Ngrid**3), np.ones(len(np.log10(Ngrid**3)))]).T
#c1, c2 = np.linalg.lstsq(A, np.log10(blowup+1), rcond=None)[0]
#
#plt.plot(Ngrid**3,(1+blowup)/((10**c2)*Ngrid**c1) , 'r', label='Fitted line')
blowup, cs2,dt,Ngrid = blowup_redshifts_func("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig8/not_source/",dt=True,Ngrid = True)
plt.scatter(Ngrid**3,1+blowup,label="DE not sourcing gravity",marker="*",s=100)

#plt.plot([10**4,10**9],[((10**4)**0.22),((10**9)**0.22)])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'N$_\mathrm{grid}$',size=14)
plt.ylabel(r'$1+z_b$',size=14)

plt.legend()
plt.title(r'$N_\mathrm{grid}=N_\mathrm{particles}, \ L = \ 300\mathrm{Mpc/h},  w = -0.9,   c_s^2=10^{-7}$',size=12)
#plt.show()

#A = np.vstack([np.log10(Ngrid**3), np.ones(len(np.log10(Ngrid**3)))]).T
#c1, c2 = np.linalg.lstsq(A, np.log10(blowup+1), rcond=None)[0]
#
#plt.plot(Ngrid**3,(1+blowup)/((10**c2)*Ngrid**c1) , 'r', label='Fitted line')
plt.tight_layout()
plt.show()
"""


blowup, cs2,dt = blowup_redshifts_func("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig7/N5_new/source/",True,False)
plt.title(r'$N_\mathrm{grid}=N_\mathrm{particles}=256^3, \ L = \ 300\mathrm{Mpc/h}, \ w = -0.9, \ \   c_s^2=10^{-7}$',size=12)
print(blowup)
print(dt)
plt.scatter(blowup,dt,label="DE sourcing gravity",s=100)
blowup, cs2,dt = blowup_redshifts_func("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig7/N5_new/not_source/",True,False)
print(blowup)
print(dt)
plt.scatter(blowup,dt,marker="*",label="DE not sourcing gravity",s=100)
#plt.gca().invert_xaxis()
#plt.title(r'$N_\mathrm{grid}=N_\mathrm{particles}=256^3, \ L = \ 300\mathrm{Mpc/h},  w = -0.9$',size=12)
#plt.yscale('log')
plt.xlabel(r'$z_b$',size=14)
#blowup, cs2 = blowup_redshifts_func("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig7/N5/not_source/")
#plt.scatter(blowup,cs2,marker="*")
plt.ylabel(r'$d\tau$',size=14)
plt.tight_layout()
plt.legend()
#plt.savefig("fig7_N1.pdf")
plt.show()
sys.exit(0)
"""
blowup, cs2 = blowup_redshifts_func("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig2/not_source/",False,False)
plt.scatter(blowup,cs2,s=100)
plt.title(r'$N_\mathrm{grid}=N_\mathrm{particles}=256^3, \ L = \ 300\mathrm{Mpc/h},  w = -0.9$',size=12)
plt.yscale('log')
plt.xlabel(r'$z_b$',size=14)
plt.ylabel(r'$c_s^2$',size=14)
plt.tight_layout()
#plt.savefig("blowup_cs2.pdf")
plt.show()
"""


#z,rel_phi,rel_psi,max_phi,max_psi,avg_rel_phi = read_potentials("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig7/not_source/N512/potentials.txt")


"""
fig,axs = plt.subplots(1,2,figsize=(6.5,6.5),sharey=True)

z,rel_phi,rel_psi,max_phi,max_psi,avg_rel_phi = read_potentials("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig7/N1/not_source/dt_04/potentials.txt")
axs[0].scatter(z,rel_phi,label="Max " + r"|$\frac{\Phi_\mathrm{New}-\Phi_\mathrm{Old}}{\Phi_\mathrm{Old}}}$|")
axs[0].scatter(z,avg_rel_phi,label= r"$<|\frac{\Phi_\mathrm{New}-\Phi_\mathrm{Old}}{\Phi_\mathrm{Old}}|>$")
axs[0].scatter(z,max_phi,label="Max " + r"$|\Phi |$")
axs[0].set_yscale('log')
axs[0].invert_xaxis()
axs[0].set_xlabel('z')
axs[0].set_title(r"$d\tau = 0.04$")


z,rel_phi,rel_psi,max_phi,max_psi,avg_rel_phi = read_potentials("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fig7/N1/not_source/dt_0025/potentials.txt")

axs[1].scatter(z,rel_phi)#,label="Max relative change")
axs[1].scatter(z,avg_rel_phi)#,label="Average relative change")
axs[1].scatter(z,max_phi)#,label="Max " + r"$\Phi$")
axs[1].set_yscale('log')
axs[1].invert_xaxis()
axs[1].set_xlabel('z')
axs[1].set_title(r"$d\tau = 0.0025$")
#z,rel_phi,rel_psi,max_phi,max_psi = read_potentials("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/test_implementation/new/variable_dtau/N1test2/potentials.txt")
#plt.plot(z,max_phi)
#plt.scatter(z,max_psi)
#plt.scatter(z,rel_psi)

#plt.yscale('log')
#plt.gca().invert_xaxis()
#plt.ylim(10**(-5),10**7)
#plt.xlabel('z')
#plt.legend()
#plt.tight_layout()
fig.legend(loc=(0.09,0.01),fancybox=True, shadow=True, ncol=3,fontsize=13)
plt.subplots_adjust(top=0.965,
bottom=0.16,
left=0.075,
right=0.99,
hspace=0.275,
wspace=0.075)
plt.subplot_tool()

plt.setp(axs,ylim=(1e-5,1e+7),xlim=(101,0))

plt.show()

"""
#sys.exit(0)



"""
 #read_div_variables("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/123002/N1/div_variables.txt")
fig,axs = plt.subplots(2,2,figsize=(7.5,7.5))
marker = "x"
s=70
root = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/comparing_versions/"
file = root + "new_1em7/N1/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[0,0].scatter(z,abs(avg_pi),label= "New"+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color="mediumblue")
axs[0,0].set_title("|$\mathrm{Average}\ \mathcal{H}\pi$|")
axs[0,0].set_yscale('log')
axs[0,0].set_xlabel('z')
axs[0,0].invert_xaxis()

file = root + "old_1em7/N1/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[0,0].scatter(z,abs(avg_pi),label= "Old"+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color="mediumblue",marker = marker,s=s)

file = root + "new_1em7/N5/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[0,0].scatter(z,abs(avg_pi),label= "New"+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color="orange")

file = root + "old_1em7/N5/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[0,0].scatter(z,abs(avg_pi),label= "Old"+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color="orange",marker = marker,s=s)

file = root + "new_1em7/N10/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[0,0].scatter(z,abs(avg_pi),label= "New"+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color="seagreen")





file = root + "old_1em7/N10/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[0,0].scatter(z,abs(avg_pi),label= "Old"+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color="seagreen",marker = marker,s=s)
#axs[0,0].legend(loc='upper center')

#####################################################################################

file = root + "new_1em7/N1/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[1,0].scatter(z,abs(max_pi),color="mediumblue")
axs[1,0].set_title("$\mathrm{Max}\ |\mathcal{H}\pi |$")
axs[1,0].set_yscale('log')
axs[1,0].set_xlabel('z')
axs[1,0].invert_xaxis()

file = root + "new_1em7/N5/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[1,0].scatter(z,abs(max_pi),color="orange")

file = root + "new_1em7/N10/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[1,0].scatter(z,abs(max_pi),color="seagreen")

file = root + "old_1em7/N1/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[1,0].scatter(z,abs(max_pi),color="mediumblue",marker = marker,s=s)

file = root + "old_1em7/N5/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[1,0].scatter(z,abs(max_pi),color="orange",marker = marker,s=s)

file = root + "old_1em7/N10/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[1,0].scatter(z,abs(max_pi),color="seagreen",marker = marker,s=s)
#axs[1,0].legend()

#####################################################################################

file = root + "new_1em7/N1/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[1,1].scatter(z,abs(max_zeta),color="mediumblue")
axs[1,1].set_title("$\mathrm{Max}\ |\zeta |$")
axs[1,1].set_yscale('log')
axs[1,1].set_xlabel('z')
axs[1,1].invert_xaxis()

file = root + "new_1em7/N5/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[1,1].scatter(z,abs(max_zeta),color="orange")

file = root + "new_1em7/N10/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[1,1].scatter(z,abs(max_zeta),color="seagreen")

file = root + "old_1em7/N1/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[1,1].scatter(z,abs(max_zeta),color="mediumblue",marker = marker,s=s)

file = root + "old_1em7/N5/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[1,1].scatter(z,abs(max_zeta),color="orange",marker = marker,s=s)

file = root + "old_1em7/N10/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[1,1].scatter(z,abs(max_zeta),color="seagreen",marker = marker,s=s)
#axs[1,1].legend()
#####################################################################################


file = root + "new_1em7/N1/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[0,1].scatter(z,abs(avg_zeta),color="mediumblue")
axs[0,1].set_title("|$\mathrm{Average}\ \zeta$|")
axs[0,1].set_yscale('log')
axs[0,1].set_xlabel('z')
axs[0,1].invert_xaxis()

file = root + "new_1em7/N5/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[0,1].scatter(z,abs(avg_zeta),color="orange")

file = root + "new_1em7/N10/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[0,1].scatter(z,abs(avg_zeta),color="seagreen")

file = root + "old_1em7/N1/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[0,1].scatter(z,abs(avg_zeta),color="mediumblue",marker = marker,s=s)

file = root + "old_1em7/N5/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[0,1].scatter(z,abs(avg_zeta),color="orange",marker = marker,s=s)

file = root + "old_1em7/N10/div_variables.txt"
z,avg_pi,max_pi,avg_zeta,max_zeta, cs2_kessence,N_kessence = read_div_variables(file)
axs[0,1].scatter(z,abs(avg_zeta),color="seagreen",marker = marker,s=s)
#axs[0,1].legend()

#####################################################################################
#fig.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=3, borderaxespad=0.)
#loc=(0.19,0)
########fig.legend(loc="lower center",fancybox=True, shadow=True, ncol=3)
########plt.subplot_tool()
########plt.subplots_adjust(top=0.965,
########bottom=0.145,
########left=0.065,
########right=0.99,
########hspace=0.275,
########wspace=0.2)
########plt.setp(axs, xlim=(1.9,1.7))#,ylim=(1e-8,1e+5))

#plt.tight_layout()
plt.show()
"""




"""
file = root + "new_1em7/N1/div_variables.txt"
plot(file,"max_zeta","New","mediumblue")

file = root + "new_1em7/N5/div_variables.txt"
plot(file,"max_zeta","New","orange")

file = root + "new_1em7/N10/div_variables.txt"
plot(file,"max_zeta","New","seagreen")

file = root + "old_1em7/N1/div_variables.txt"
plot(file,"max_zeta","Old","mediumblue",marker = "x",s=70)

file = root + "old_1em7/N5/div_variables.txt"
plot(file,"max_zeta","Old","orange",marker = "x",s=70)

file = root + "old_1em7/N10/div_variables.txt"
plot(file,"max_zeta","Old","seagreen",marker = "x",s=70)

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
plt.xlabel('z')
#plt.ylabel('')
plt.show()
"""









#sys.exit()


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




#z = CLASS_file("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/hiclass_tests/test1/class_background.dat")
#print(z)
#sys.exit(0)





#def Omega_from_friedmann(a,w,Omega_0,rho_crit):
#    Omega = 
#    return np.array(a),np.array(Omega)


def equality(time,a,b): # same length of all lists
    for i in range(len(time)-1):
        diff_old = a[i] - b[i] 
        if ((a[i+1] - b[i+1])/diff_old < 0):
            return (time[i+1] + time[i])/2



fig,axs = plt.subplots(2,2,figsize=(7.5,7.5),sharex='col', sharey='row')
marker = "x"
s=70
a, z, DE, M ,Rad = read_DE_a("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/coincidence_problem/w_1/rho_i_rho_crit_0.txt")
#axs[0,0].scatter(z,abs(avg_pi),label= "New"+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color="mediumblue")
axs[0,0].set_title(r"$\rho_i/\rho_\mathrm{crit,0} \ , \quad c_s^2=1,\ w\approx -1$")
axs[0,0].plot(np.log(a),DE,label="DE")
axs[0,0].plot(np.log(a),Rad,label="Radiation")
axs[0,0].plot(np.log(a),M,label="Matter")
axs[0,0].plot(np.log(np.ones(2)*equality(a,DE,M)),np.array([1e-10,1e+30]),"--",label="Cosmic time = 10.3 [Gyr]",color="k")
axs[0,0].set_xlim(-15,0.1)
axs[0,0].set_ylim(1e-5,1e+23)
axs[0,0].set_yscale('log')
axs[0,0].set_xlabel(r"$\ln (a)$")



a, z, DE, M ,Rad = read_DE_a("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/coincidence_problem/w_0/rho_i_rho_crit_0.txt")
#axs[0,0].scatter(z,abs(avg_pi),label= "New"+ ", "  + r"N$_\mathrm{kess} =$ " +str(int(N_kessence)),color="mediumblue")
axs[0,1].set_title(r"$\rho_i/\rho_\mathrm{crit,0} \ , \quad c_s^2=1,\ w\approx 0$")
axs[0,1].plot(np.log(a),DE)#,label="DE")
axs[0,1].plot(np.log(a),Rad)#,label="Radiation")
axs[0,1].plot(np.log(a),M)#,label="Matter")
#axs0110].plot(np.log(np.ones(2)*equality(a,DE,M)),np.array([1e-10,1e+30]),"--",label="Cosmic time = 10.3 [Gyr]",color="k")
axs[0,1].set_xlim(-15,0.1)
axs[0,1].set_ylim(1e-5,1e+23)
axs[0,1].set_yscale('log')
axs[0,1].set_xlabel(r"$\ln (a)$")

#######################################################################################
a, z, Omega_DE, Omega_M, Omega_Rad = Omega_func("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/coincidence_problem/w_1/Omega.txt")

axs[1,0].set_title(r"$\Omega_i \ , \quad c_s^2=1,\ w\approx -1$")
axs[1,0].plot(np.log(a),Omega_DE)#,label=r"$\Omega_\mathrm{DE}$")
axs[1,0].plot(np.log(a),Omega_Rad)#,label=r"$\Omega_\mathrm{Rad}$")
axs[1,0].plot(np.log(a),Omega_M)#,label=r"$\Omega_\mathrm{M}$")
#axs[1,0].plot(np.log(np.ones(2)*equality(a,DE,M)),np.array([1e-10,1e+30]),"--",label="Cosmic time = 10.3 [Gyr]",color="k")
axs[1,0].set_xlim(-15,0.1)
#axs[1,0].set_ylim(1e-5,1e+23)
#axs[1,0].set_yscale('log')
axs[1,0].set_xlabel(r"$\ln (a)$")

a, z, Omega_DE, Omega_M, Omega_Rad = Omega_func("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/coincidence_problem/w_0/Omega.txt")

axs[1,1].set_title(r"$\Omega_i \ , \quad c_s^2=1,\ w\approx 0$")
axs[1,1].plot(np.log(a),Omega_DE)#,label=r"$\Omega_\mathrm{DE}$")
axs[1,1].plot(np.log(a),Omega_Rad)#,label=r"$\Omega_\mathrm{Rad}$")
axs[1,1].plot(np.log(a),Omega_M)#,label=r"$\Omega_\mathrm{M}$")
#axs[1,0].plot(np.log(np.ones(2)*equality(a,DE,M)),np.array([1e-10,1e+30]),"--",label="Cosmic time = 10.3 [Gyr]",color="k")
axs[1,1].set_xlim(-15,0.1)
#axs[1,0].set_ylim(1e-5,1e+23)
#axs[1,0].set_yscale('log')
axs[1,1].set_xlabel(r"$\ln (a)$")

#loc=(0.19,0)
fig.legend(loc="lower center",fancybox=True, shadow=True, ncol=4)

plt.subplots_adjust(top=0.965,
bottom=0.11,
left=0.06,
right=0.985,
hspace=0.205,
wspace=0.04)
#plt.setp(axs, xlim=(1.9,1.7))#,ylim=(1e-8,1e+5))
#axs[0,0].invert_xaxis()
plt.subplot_tool()
plt.show()



#a, z, DE, M ,Rad = read_DE_a("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/coincidence_problem/w_1/rho_i_rho_crit_0.txt")
##print(equality(a,DE,M))
##sys.exit(0)
#plt.title(r"$\rho_i/\rho_\mathrm{crit,0} \ , \quad c_s^2=1,\ w\approx -1$")
#plt.semilogy(np.log(a),DE,label="DE")
#plt.semilogy(np.log(a),Rad,label="Radiation")
#plt.semilogy(np.log(a),M,label="Matter")
#plt.semilogy(np.log(np.ones(2)*equality(a,DE,M)),np.array([1e-10,1e+30]),"--",label="Cosmic time = 10.3 [Gyr]",color="k")
#plt.xlabel("ln a")
#plt.legend()
#plt.xlim(-15,0.1)
#plt.ylim(1e-5,1e+23)
##plt.savefig("/uio/hume/student-u23/jorgeagl/src/master/master_project/rand_figs/coincidence_w1.pdf")
##plt.gca().invert_xaxis()
#plt.show()
#sys.exit(0)



#a, z, Omega_DE, Omega_M, Omega_Rad = Omega_func("/mn/stornext/d5/data/jorgeagl/kevolution_output/results/coincidence_problem/w_0/Omega.txt")
#sum = np.zeros(len(a))
##sum[:] = Omega_DE[:]+ Omega_CDM[:]+ Omega_b[:]+ Omega_g[:]
#plt.title(r"$\Omega_i \ , \quad c_s^2=1,\ w\approx 0$")
#plt.plot(np.log(a),Omega_DE,label=r"$\Omega_\mathrm{DE}$")
#plt.plot(np.log(a),Omega_Rad,label=r"$\Omega_\mathrm{Rad}$")
#plt.plot(np.log(a),Omega_M,label=r"$\Omega_\mathrm{M}$")
##plt.plot(np.log(a),Omega_b,label=r"$\Omega_\mathrm{b}$")
#plt.xlabel(r"$\ln (a)$")
##plt.plot(np.log(a),sum)
##plt.plot(z,Omega_DE,label="DE")
##plt.plot(z,Omega_CDM,label="CDM")
##plt.plot(z,Omega_b,label="b")
##plt.plot(z,Omega_g,label="g")
#
#plt.legend()
##plt.gca().invert_xaxis()
#plt.xlim(-15,0.1)
##plt.savefig("/uio/hume/student-u23/jorgeagl/src/master/master_project/rand_figs/Omega_w0.pdf")
#plt.show()
#sys.exit()




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

