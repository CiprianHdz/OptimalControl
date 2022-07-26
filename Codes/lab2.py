from matplotlib import pyplot as plt
import numpy as np

# modelo

## Ecuación de estado x'=g(x,u) en u^{*}
def gx(x_vec, u, t):
    x = x_vec
    dx = r*(M-x)-u*x #r(M-x)-ux
    return np.array([dx])

## Ecuación adjunta 
def glambda(x_vec, u, l_vec, t):
    x = x_vec
    l = l_vec
    
    dl = -2*A+l*(r+u) #-2Ax+\lambda(r+u)
             
    return np.array([dl])


##_________________________________________
# caracterizacion del control
def u_nueva(x, u, l):
    res = (l*x)/2
    
    #Actualizacion
    res = (res+u)/2.0
    return res

###_____________________________________ Método_____________________
# forward
def runge_forward():
    for i in range(N-1):
        u_medio = 0.5*(u[i]+u[i+1])
        
        k1 = gx( x[i], u[i], i )
        k2 = gx( x[i]+h*0.5*k1, u_medio, i+h/2.0 )
        k3 = gx( x[i]+h*0.5*k2, u_medio, i+h/2.0 )
        k4 = gx( x[i]+h*k3, u[i+1], i+h )
        
        x[i+1] = x[i] + h*(k1+2.0*k2+2.0*k3+k4)/6.0
    return x

# backward
def runge_backward():
    for i in range(N-1,0,-1):
        u_medio = 0.5*( u[i]+u[i-1] )
        x_medio = 0.5*( x[i]+x[i-1] )
        k1 = glambda( x[i], u[i], l[i], i )
        k2 = glambda( x_medio, u_medio, l[i]-h*0.5*k1, i )
        k3 = glambda( x_medio, u_medio, l[i]-h*0.5*k2, i )
        k4 = glambda( x_medio, u_medio, l[i]-h*k3, i )
        l[i-1] = l[i] - h*(k1+2.0*k2+2.0*k3+k4)/6.0
    return l

# metodo forward-backward
def forward_backward():
    global u
    x_iter = np.zeros((N, ITER_MAX))
    l_iter = np.zeros((N, ITER_MAX))
    u_iter = np.zeros((N, ITER_MAX))

    for i in range(ITER_MAX):
        u_iter[:,i]  = u
        x_iter[:,i]  = runge_forward()
        l_iter[:,i]  = runge_backward()
        uu = u_nueva(x_iter[:,i], u, l_iter[:,i])
        u = uu
    
    return x_iter, u_iter, l_iter

#--------------

# parametros de integracion
h = 0.01
tf = 5
N = int(np.floor(tf/h))

# variables
t = np.linspace(0,tf,N) #discretización del tiempo
x = np.zeros(N)
u = np.zeros(N)
l = np.zeros(N)

# parametros de modelo
r = 0.6
M = 5
A = 10

# condiciones de frontera
x[0]   = 5.0
l[N-1] = 0.0

# parametros de forward-backward
ITER_MAX = 50
#TOL = 1e-9

#--------------

# resultados
x_iter, u_iter, l_iter = forward_backward()

# grafica de estado
fig = plt.figure()
ax = fig.add_subplot(111)
string = 'Concentración de moho'
plt.title(string)
plt.plot(t,x_iter[:,1],linewidth=1.0,color='k',label='Iteraciones')
plt.plot(t,x_iter[:,2:ITER_MAX-2],linewidth=1.0,color='k')
plt.plot(t,x_iter[:,0],linewidth=2.0,color='magenta',label='Sin control')
plt.plot(t,x_iter[:,ITER_MAX-1],linewidth=2.0,color='cyan',label='control óptimo (fungicida)')
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
fig.savefig('L2_caso3_estado.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

# grafica de control
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Fungicida')
plt.plot(t,u_iter[:,1],linewidth=1.0,color='k',label='Iteraciones')
plt.plot(t,u_iter[:,2:ITER_MAX-2],linewidth=1.0,color='k')
plt.plot(t,u_iter[:,0],linewidth=2.0,color='magenta',label='Sin control')
plt.plot(t,u_iter[:,ITER_MAX-1],linewidth=2.0,color='cyan',label='control óptimo')
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(True)
fig.savefig('L2_caso3_control.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
