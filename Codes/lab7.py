
# caracterizacion del control
def u_nueva(L1,L4, S, u):
    res = (L1-L4)*S/2
    for i in range(N_max):
        res[i] = min(M2, max(M1, res[i]))
    res = (res+u)/2.0  #Actualización del control
    return res    


###_____Lado derecho de la EDO de variables de estado
def f_1(N,S,I,u):
    return b*N-d*S-c*S*I-u*S

from matplotlib import pyplot as plt
import numpy as np


def f_2(S,I,E):
    return c*S*I-(e+d)*E

def f_3(E,I):
    return e*E-(g+a+d)*I

def f_4(I,R,S,u):
    return g*I-d*R+u*S

def f_5(N, I):
    return (b-d)*N-a*I

##__Lado derecho de la EDO adjunta__

def l_1(L1,I,L2,L4,u):
    return L1*(d+c*I+u)-L2*c*I-L4*u

def l_2(L2,E,L3):
    return L2*(e+d)*E-L3*e

def l_3(L1,S,L2,L3,L4,L5):
    return -A + L1*c*S-L2*c*S+L3*(g+a+d)-L4*g + L5*a

def l_4(L4):
    return L4*d

def l_5(L1, L5):
    return -L1*b-L5*(b-d)

# forward
def runge_forward():
    for i in range(N_max-1):
        
        K1S=f_1(N[i],S[i],I[i],u[i])
        K1E=f_2(S[i],I[i],E[i])
        K1I=f_3(E[i],I[i])
        K1R=f_4(I[i],R[i],S[i],u[i])
        K1N=f_5(N[i],I[i])
        
        u_medio=0.5*(u[i]+u[i+1])
        
        K2S=f_1(N[i]+h*K1N/2,S[i]+h*K1S/2,I[i]+h*K1I/2,u_medio)
        K2E=f_2(S[i]+h*K1S/2,I[i]+h*K1I/2,E[i]+h*K1E/2)
        K2I=f_3(E[i]+h*K1E/2,I[i]+h*K1I/2)
        K2R=f_4(I[i]+h*K1I/2,R[i]+h*K1R/2,S[i]+h*K1S/2,u_medio)
        K2N=f_5(N[i]+h*K1N/2,I[i]+h*K1I/2)
        
        K3S=f_1(N[i]+h*K2N/2,S[i]+h*K2S/2,I[i]+h*K2I/2,u_medio)
        K3E=f_2(S[i]+h*K2S/2,I[i]+h*K2I/2,E[i]+h*K2E/2)
        K3I=f_3(E[i]+h*K2E/2,I[i]+h*K2I/2)
        K3R=f_4(I[i]+h*K2I/2,R[i]+h*K2R/2,S[i]+h*K2S/2,u_medio)
        K3N=f_5(N[i]+h*K2N/2,I[i]+h*K2I/2)
        
        K4S=f_1(N[i]+h*K3N,S[i]+h*K3S,I[i]+h*K3I,u[i+1])
        K4E=f_2(S[i]+h*K3S,I[i]+h*K3I,E[i]+h*K3E)
        K4I=f_3(E[i]+h*K3E,I[i]+h*K3I)
        K4R=f_4(I[i]+h*K3I,R[i]+h*K3R,S[i]+h*K3S,u[i+1])
        K4N=f_5(N[i]+h*K3N,I[i]+h*K3I)
        
        S[i+1]=S[i]+ h/6*(K1S+2*K2S+2*K3S+K4S)
        E[i+1]=E[i]+ h/6*(K1E+2*K2E+2*K3E+K4E)
        I[i+1]=I[i]+ h/6*(K1I+2*K2I+2*K3I+K4I)
        R[i+1]=R[i]+ h/6*(K1R+2*K2R+2*K3R+K4R)
        N[i+1]=N[i]+ h/6*(K1N+2*K2N+2*K3S+K4N)  
    return S,E,I,R,N

def runge_backward():
    
    for i in range(N_max-1,0,-1):
               
               
        u_medio = 0.5*( u[i]+u[i-1] )
        S_medio = 0.5*( S[i]+S[i-1] )
        E_medio = 0.5*( E[i]+E[i-1] )
        I_medio = 0.5*( I[i]+I[i-1] )
        R_medio = 0.5*( R[i]+R[i-1] )
        N_medio = 0.5*( N[i]+N[i-1] )
               
        #_____      
        K1L1=l_1(L1[i],I[i],L2[i],L4[i],u[i])
        K1L2=l_2(L2[i], E[i], L3[i])
        K1L3=l_3(L1[i], S[i], L2[i], L3[i], L4[i], L5[i])
        K1L4=l_4(L4[i])
        K1L5=l_5(L1[i],L5[i])
        #____
        K2L1=l_1(L1[i]-h*K1L1/2,I_medio,L2[i]-h*K1L2/2,L4[i]-h*K1L4/2,u_medio)
        K2L2=l_2(L2[i]-h*K1L2/2, E_medio, L3[i]-h*K1L3/2)
        K2L3=l_3(L1[i]-h*K1L1/2, S_medio, L2[i]-h*K1L2/2,
                 L3[i]-h*K1L3/2, L4[i]-h*K1L4/2, L5[i]-h*K1L5/2)
        K2L4=l_4(L4[i]-h*K1L4/2)
        K2L5=l_5(L1[i]-h*K1L1/2,L5[i]-h*K1L5/2)
        #____
        K3L1=l_1(L1[i]-h*K2L1/2,I_medio,L2[i]-h*K2L2/2,L4[i]-h*K2L4/2,u_medio)
        K3L2=l_2(L2[i]-h*K2L2/2, E_medio, L3[i]-h*K2L3/2)
        K3L3=l_3(L1[i]-h*K2L1/2, S_medio, L2[i]-h*K2L2/2,
                 L3[i]-h*K2L3/2, L4[i]-h*K2L4/2, L5[i]-h*K2L5/2)
        K3L4=l_4(L4[i]-h*K2L4/2)
        K3L5=l_5(L1[i]-h*K2L1/2,L5[i]-h*K2L5/2)
        #______
        K4L1=l_1(L1[i]-h*K3L1,I_medio,L2[i]-h*K3L2,L4[i]-h*K3L4,u[i-1])
        K4L2=l_2(L2[i]-h*K3L2, E_medio, L3[i]-h*K3L3)
        K4L3=l_3(L1[i]-h*K3L1, S_medio, L2[i]-h*K3L2,
                 L3[i]-h*K3L3, L4[i]-h*K3L4, L5[i]-h*K3L5)
        K4L4=l_4(L4[i]-h*K3L4)
        K4L5=l_5(L1[i]-h*K3L1,L5[i]-h*K3L5)
        
        L1[i-1]=L1[i]- h/6*(K1L1+2*K2L1+2*K3L1+K4L1)
        L2[i-1]=L2[i]- h/6*(K1L2+2*K2L2+2*K3L2+K4L2)
        L3[i-1]=L3[i]- h/6*(K1L3+2*K2L3+2*K3L3+K4L3)
        L4[i-1]=L4[i]- h/6*(K1L4+2*K2L4+2*K3L4+K4L4)
        L5[i-1]=L5[i]- h/6*(K1L5+2*K2L5+2*K3L5+K4L5) 
    return L1, L2, L3, L4, L5

# metodo forward-backward
def forward_backward():
    global u
    
               #Variables de estados
    S_iter = np.zeros((N_max, ITER_MAX))
    E_iter = np.zeros((N_max, ITER_MAX))
    I_iter = np.zeros((N_max, ITER_MAX))
    R_iter = np.zeros((N_max, ITER_MAX))
    N_iter = np.zeros((N_max, ITER_MAX))
               #Variables adjuntas
    L1_iter = np.zeros((N_max, ITER_MAX))
    L2_iter = np.zeros((N_max, ITER_MAX))
    L3_iter = np.zeros((N_max, ITER_MAX))
    L4_iter = np.zeros((N_max, ITER_MAX))
    L5_iter = np.zeros((N_max, ITER_MAX))
               #Variable control
    u_iter =np.zeros((N_max, ITER_MAX))

    for i in range(ITER_MAX):
        u_iter[:,i]  = u
        S_iter[:,i],E_iter[:,i],I_iter[:,i],R_iter[:,i],N_iter[:,i]  = runge_forward()
        L1_iter[:,i],L2_iter[:,i],L3_iter[:,i],L4_iter[:,i],L5_iter[:,i]  = runge_backward()
        
        uu = u_nueva(L1_iter[:,i],L4_iter[:,i],S_iter[:,i], u)
        u = uu
    
    return S_iter, E_iter, I_iter, R_iter, N_iter, u_iter


#--------------

# parametros de integracion
h = .01
tf = 20
N_max = int(np.floor(tf/h))

# variables
t = np.linspace(0,tf,N_max)
# De estado
S = np.zeros(N_max)
E = np.zeros(N_max)
I = np.zeros(N_max)
R = np.zeros(N_max)
N = np.zeros(N_max)
# Adjuntas
L1 = np.zeros(N_max)
L2 = np.zeros(N_max)
L3 = np.zeros(N_max)
L4 = np.zeros(N_max)
L5 = np.zeros(N_max)
# Control
u = np.zeros(N_max)
               
# parametros de modelo
b=0.525
d=0.5
c=0.001
e=0.5
g=0.1
a=0.2
A=0.1
#__ cotas del control__
M1 = 0.0
M2 = 0.9

# condiciones  de frontera
S[0]= 1000
E[0]= 100
I[0]= 50
R[0]= 15
N[0]=1165 #=S[0]+E[0]+I[0]+R[0]
L1[N_max-1] = 0.0
L2[N_max-1] = 0.0
L3[N_max-1] = 0.0
L4[N_max-1] = 0.0
L5[N_max-1] = 0.0

# parametros de forward-backward
ITER_MAX = 50
TOL = 1e-10

#--------------

# resultados
S_iter, E_iter, I_iter, R_iter, N_iter, u_iter = forward_backward()

fig = plt.figure()
ax = fig.add_subplot(111)
string = 'Número de infectados'
plt.title(string)
plt.plot(t,I_iter[:,1],linewidth=1.0,color='k',label='Iteraciones')
plt.plot(t,I_iter[:,2:ITER_MAX-2],linewidth=1.0,color='k')
plt.plot(t,I_iter[:,0],linewidth=2.0,color='magenta',label='sin vacunación')
plt.plot(t,I_iter[:,ITER_MAX-1],linewidth=2.0,color='cyan',label='con vacunación')
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
fig.savefig('L7_infectados.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(111)
string = 'Población Total'
plt.title(string)
plt.plot(t,N_iter[:,1],linewidth=1.0,color='k',label='Iteraciones')
plt.plot(t,N_iter[:,2:ITER_MAX-2],linewidth=1.0,color='k')
plt.plot(t,N_iter[:,0],linewidth=2.0,color='magenta',label='sin vacunación')
plt.plot(t,N_iter[:,ITER_MAX-1],linewidth=2.0,color='cyan',label='con vacunación')
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
fig.savefig('L7_poblacion.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(111)
string = 'Población latente'
plt.title(string)
plt.plot(t,E_iter[:,1],linewidth=1.0,color='k',label='Iteraciones')
plt.plot(t,E_iter[:,2:ITER_MAX-2],linewidth=1.0,color='k')
plt.plot(t,E_iter[:,0],linewidth=2.0,color='magenta',label='sin vacunación')
plt.plot(t,E_iter[:,ITER_MAX-1],linewidth=2.0,color='cyan',label='con vacunación')
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
fig.savefig('L7_expuestos.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(111)
string = 'Población recuperada (o inmune)'
plt.title(string)
plt.plot(t,R_iter[:,1],linewidth=1.0,color='k',label='Iteraciones')
plt.plot(t,R_iter[:,2:ITER_MAX-2],linewidth=1.0,color='k')
plt.plot(t,R_iter[:,0],linewidth=2.0,color='magenta',label='sin vacunación')
plt.plot(t,R_iter[:,ITER_MAX-1],linewidth=2.0,color='cyan',label='con vacunación')
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
fig.savefig('L7_recuperados.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(111)
string = 'Población suceptible'
plt.title(string)
plt.plot(t,S_iter[:,1],linewidth=1.0,color='k',label='Iteraciones')
plt.plot(t,S_iter[:,2:ITER_MAX-2],linewidth=1.0,color='k')
plt.plot(t,S_iter[:,0],linewidth=2.0,color='magenta',label='sin vacunación')
plt.plot(t,S_iter[:,ITER_MAX-1],linewidth=2.0,color='cyan',label='con vacunación')
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
fig.savefig('L7_suceptible.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('control u')
plt.plot(t,u_iter[:,1],linewidth=1.0,color='k',label='Iteraciones')
plt.plot(t,u_iter[:,2:ITER_MAX-2],linewidth=1.0,color='k')
plt.plot(t,u_iter[:,0],linewidth=2.0,color='magenta',label='Sin control')
plt.plot(t,u_iter[:,ITER_MAX-1],linewidth=2.0,color='cyan',label='control óptimo')
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
fig.savefig('L7_icontrol.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
