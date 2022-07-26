from matplotlib import pyplot as plt
import numpy as np

# caracterizacion del control
def u_nueva(X1,L1,L2,P01,I,L3,u):
    res = (X1*L1 + L2*(2*P01-I+X1) +L3*I)/(2*BT)
    for i in range(N_max):
        res[i] = min(MT, max(0.0, res[i]))
    res = (res+u)/2.0  #Actualización del control
    return res    

#Lado derecho de las variables de estado 
def f_1(X1,P01,I,u):
    return (sigma+mu)*I-(2*mu+rho+sigma)*X1-(gamma+u)*X1

def f_2(X1,P01,I,u):
    return rho*(1-h1)*(X1-X1**2/Xast)-(sigma+phi*h1+2*mu)*P01+(gamma+u)*(I-X1-2*P01)
def f_3(X1,P01,I,u):
    return rho*h1*(X1-X1**2/Xast)+phi*h1*P01-mu*I-(gamma+u)*I

#Lado derecho de las variables adjuntas
def l_1(L1,X1,L2,L3,u):
    return (2*mu+rho+sigma+gamma+u)*L1+(gamma+u-rho*(1-h1))*(1-2*X1/Xast)*L2-h1*rho*(1-2*X1/Xast)*L3

def l_2(L2,L3, u):
    return (2*mu+sigma+2*(gamma+u)+h1*phi)*L2-h1*phi*L3

def l_3(L1,L2,L3,u):
    return -L1*(mu+sigma)-L2*(gamma+u)+L3*(mu+gamma+u)-1

# forward
def runge_forward():
    
    for i in range(N_max-1):        
        #_____________
        K1X1=f_1(X1[i],P01[i],I[i],u[i])
        K1P01=f_2(X1[i],P01[i],I[i],u[i])
        K1I=f_3(X1[i],P01[i],I[i],u[i])
        #__________
        u_medio=0.5*(u[i]+u[i+1])
        
        K2X1=f_1(X1[i]+h*K1X1/2,P01[i]+h*K1P01/2,I[i]+h*K1I/2,u_medio)
        K2P01=f_2(X1[i]+h*K1X1/2,P01[i]+h*K1P01/2,I[i]+h*K1I/2,u_medio)
        K2I=f_3(X1[i]+h*K1X1/2,P01[i]+h*K1P01/2,I[i]+h*K1I/2,u_medio)
        
        #________
        K3X1=f_1(X1[i]+h*K2X1/2,P01[i]+h*K2P01/2,I[i]+h*K2I/2,u_medio)
        K3P01=f_2(X1[i]+h*K2X1/2,P01[i]+h*K2P01/2,I[i]+h*K2I/2,u_medio)
        K3I=f_3(X1[i]+h*K2X1/2,P01[i]+h*K2P01/2,I[i]+h*K2I/2,u_medio)
        
        #________
        K4X1=f_1(X1[i]+h*K3X1,P01[i]+h*K3P01,I[i]+h*K3I,u_medio)
        K4P01=f_2(X1[i]+h*K3X1,P01[i]+h*K3P01,I[i]+h*K3I,u_medio)
        K4I=f_3(X1[i]+h*K3X1,P01[i]+h*K3P01,I[i]+h*K3I,u[i+1])
        
        X1[i+1]=X1[i]+ h/6*(K1X1+2*K2X1+2*K3X1+K4X1)
        P01[i+1]=P01[i]+ h/6*(K1P01+2*K2P01+2*K3P01+K4P01)
        I[i+1]=I[i]+ h/6*(K1I+2*K2I+2*K3I+K4I) 
    return X1, P01, I

def runge_backward():
    for i in range(N_max-1,0,-1):
               
               
        u_medio = 0.5*( u[i]+u[i-1] )
        X1_medio = 0.5*( X1[i]+X1[i-1] )
        P01_medio = 0.5*( P01[i]+P01[i-1] )
        I_medio = 0.5*( I[i]+I[i-1] )
               
        #______       
        K1L1=l_1(L1[i],X1[i],L2[i],L3[i],u[i])
        K1L2=l_2(L2[i],L3[i], u[i])
        K1L3=l_3(L1[i],L2[i],L3[i],u[i])
        
        #_____
        K2L1=l_1(L1[i]-h*K1L1/2,X1_medio,L2[i]-h*K1L2/2,
                 L3[i]-h*K1L3/2,u_medio)
        K2L2=l_2(L2[i]-h*K1L2/2,L3[i]-h*K1L3/2, u_medio)
        K2L3=l_3(L1[i]-h*K1L1/2,L2[i]-h*K1L2/2,L3[i]-h*K1L3/2,u_medio)
        
        #_____
        K3L1=l_1(L1[i]-h*K2L1/2,X1_medio,L2[i]-h*K2L2/2,
                 L3[i]-h*K2L3/2,u_medio)
        K3L2=l_2(L2[i]-h*K2L2/2,L3[i]-h*K2L3/2, u_medio)
        K3L3=l_3(L1[i]-h*K2L1/2,L2[i]-h*K2L2/2,L3[i]-h*K2L3/2,u_medio)
        
        #_____
        K4L1=l_1(L1[i]-h*K3L1,X1_medio,L2[i]-h*K3L2,
                 L3[i]-h*K3L3,u_medio)
        K4L2=l_2(L2[i]-h*K3L2,L3[i]-h*K3L3, u_medio)
        K4L3=l_3(L1[i]-h*K3L1,L2[i]-h*K3L2,L3[i]-h*K3L3,u[i-1])
        
        
        L1[i-1]=L1[i]- h/6*(K1L1+2*K2L1+2*K3L1+K4L1)
        L2[i-1]=L2[i]- h/6*(K1L2+2*K2L2+2*K3L2+K4L2)
        L3[i-1]=L3[i]- h/6*(K1L3+2*K2L3+2*K3L3+K4L3)
    return L1, L2, L3

# metodo forward-backward
def forward_backward():
    global u
    
               #Variables de estados
    X1_iter = np.zeros((N_max, ITER_MAX))
    P01_iter = np.zeros((N_max, ITER_MAX))
    I_iter = np.zeros((N_max, ITER_MAX))
               #Variables adjuntas
    L1_iter = np.zeros((N_max, ITER_MAX))
    L2_iter = np.zeros((N_max, ITER_MAX))
    L3_iter = np.zeros((N_max, ITER_MAX))
               #Variable control
    u_iter = np.ones((N_max, ITER_MAX))

    for i in range(ITER_MAX):
        u_iter[:,i]  = u
        X1_iter[:,i],P01_iter[:,i],I_iter[:,i]  = runge_forward()
        L1_iter[:,i],L2_iter[:,i],L3_iter[:,i]  = runge_backward()
        
        uu = u_nueva(X1_iter[:,i],L1_iter[:,i],L2_iter[:,i],P01_iter[:,i],I_iter[:,i],L3_iter[:,i],u_iter[:,i])
        u = uu
    
    return X1_iter, P01_iter, I_iter, u_iter

#--------------

# parametros de integracion
h = 1
tf = 5 # en años
N_max = int(np.floor(tf/h))


#discretización del tiempo
t = np.linspace(0,tf,N_max)

# De estado
X1 = np.zeros(N_max)
P01 = np.zeros(N_max)
I = np.zeros(N_max)

# Adjuntas
L1 = np.zeros(N_max)
L2 = np.zeros(N_max)
L3 = np.zeros(N_max)

# Control
u = np.ones(N_max)
               
# parametros de modelo
rho=5.0
sigma=2.0
mu= 1/9
nu=1.111e5
phi=52.0
BT=10e5
#______ Caso Clamidia
gamma= 0.855
h1=0.129
Xast=(nu/mu)*(sigma+2*mu)/(sigma+2*mu+rho)

#__ cotas del control__
MT = 1

# condiciones  de frontera
X1[0]= 1e5
P01[0]= 3e5
I[0]=7e5
L1[N_max-1] = 0.0
L2[N_max-1] = 0.0
L3[N_max-1] = 0.0


# parametros de forward-backward
ITER_MAX = 20

#--------------

# resultados
X1_iter, P01_iter, I_iter, u_iter = forward_backward()

fig = plt.figure()
ax = fig.add_subplot(111)
string = 'Individuos con Gonorrea'
plt.title(string)
plt.plot(t,P01_iter[:,1],linewidth=1.0,color='k',label='Iteraciones')
plt.plot(t,P01_iter[:,2:ITER_MAX-2],linewidth=1.0,color='k')
plt.plot(t,P01_iter[:,0],linewidth=2.0,color='magenta',label='sin tratamiento')
plt.plot(t,P01_iter[:,ITER_MAX-1],linewidth=2.0,color='cyan',label='con tratamiento')
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
fig.savefig('Ariel_individuos.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Plan de acción:Tratamiento')
plt.plot(t,u_iter[:,1],linewidth=1.0,color='k',label='Iteraciones')
plt.plot(t,u_iter[:,2:ITER_MAX-2],linewidth=1.0,color='k')
plt.plot(t,u_iter[:,0],linewidth=2.0,color='magenta',label='Sin control')
plt.plot(t,u_iter[:,ITER_MAX-1],linewidth=2.0,color='cyan',label='control óptimo')
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
fig.savefig('Ariel_control.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
