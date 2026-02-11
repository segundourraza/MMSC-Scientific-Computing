import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import numpy as np



beta = 1
th_c = 1
th_ratio = np.linspace(0.8, 0.2, 4)[1:]

def psi_log(c, thr):
    th = thr*th_c
    return th/2*((1+c)*np.log((1+c)/2) + (1-c)*np.log((1-c)/2)) + th_c/2*(1-c**2)

def psi_pol(c):
    return 1/4*(c**2-beta**2)**2


cc = np.linspace(-1.5, 1.5, 100)


fig, ax = plt.subplots()
for thr in th_ratio:
    ax.plot(cc, psi_log(cc,thr), label = r"$\phi_{log}(c), \theta/\theta_c$ = " + "{:.1f}".format(thr))
ax.plot(cc, psi_pol(cc),'k', label = r"$\phi_{pol}(c)$")
ax.grid()
ax.legend(fontsize = 12)
ax.set_xlabel('c')
ax.set_ylabel('$\\phi(c)$', rotation = 0, labelpad=20)
fig.tight_layout()
# ax.axis('equal')
# ax.set_xlim(-1.5, 1.5)

plt.show()

