import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

radioTerreste = 6378.0 # km	
parametroGravitacionalTerrestre = 3.9860043543609598E+5 # km^3/s^2

# Modelos matemáticos
def ecuacionDiferencialParaDosCuerpos(t, estado):
    # vector posición con 3 componentes
    r = estado[:3]
    
    # Aceleracion obtenida con la Ley de gravitación universal de Newton
    a = -parametroGravitacionalTerrestre * r / np.linalg.norm(r)**3
    
    return np.array([estado[3], estado[4], estado[5], a[0], a[1], a[2]])

def rungeKutta4Paso(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + k3 * dt )
    
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Modelos de visualización
def simularOrbita(rs, args):
    _args = {
        'figsize': (10, 8),
        'labels': [''] * len(rs),
        'colors': ['m', 'c', 'r', 'C3'],
        'traj_lws': 3,
        'dist_unit': 'km',
        'groundtracks': False,
        'cb_radius': 6378.0,
        'cb_SOI': None,
        'cb_SOI_color': 'c',
        'cb_SOI_alpha': 0.7,
        'cb_axes': True,
        'cb_axes_mag': 2,
        'cb_cmap': 'Blues',
        'cb_axes_color': 'w',
        'axes_mag': 0.8,
        'axes_custom': None,
        'title': 'Trajectories',
        'legend': True,
        'axes_no_fill': True,
        'hide_axes': False,
        'azimuth': False,
        'elevation': False,
        'show': False,
        'filename': False,
        'dpi': 300
    }
    for key in args.keys():
        _args[key] = args[key]

    fig = plt.figure(figsize=_args['figsize'])
    ax = fig.add_subplot(111, projection='3d')

    max_val = 0
    n = 0

    for r in rs:
        ax.plot(r[:, 0], r[:, 1], r[:, 2],
                color=_args['colors'][n], label=_args['labels'][n],
                zorder=10, linewidth=_args['traj_lws'])
        ax.plot([r[0, 0]], [r[0, 1]], [r[0, 2]], 'o',
                color=_args['colors'][n])

        if _args['groundtracks']:
            rg = r[:] / np.linalg.norm(r, axis=1).reshape((r.shape[0], 1))
            rg *= _args['cb_radius']

            ax.plot(rg[:, 0], rg[:, 1], rg[:, 2], cs[n], zorder=10)
            ax.plot([rg[0, 0]], [rg[0, 1]], [rg[0, 2]], cs[n] + 'o', zorder=10)

        max_val = max([r.max(), max_val])
        n += 1

    _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
    _x = _args['cb_radius'] * np.cos(_u) * np.sin(_v)
    _y = _args['cb_radius'] * np.sin(_u) * np.sin(_v)
    _z = _args['cb_radius'] * np.cos(_v)
    ax.plot_surface(_x, _y, _z, cmap=_args['cb_cmap'], zorder=1)

    if _args['cb_axes']:
        l = _args['cb_radius'] * _args['cb_axes_mag']
        x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        u, v, w = [[l, 0, 0], [0, l, 0], [0, 0, l]]
        ax.quiver(x, y, z, u, v, w, color=_args['cb_axes_color'])

    xlabel = 'X (%s)' % _args['dist_unit']
    ylabel = 'Y (%s)' % _args['dist_unit']
    zlabel = 'Z (%s)' % _args['dist_unit']

    if _args['axes_custom'] is not None:
        max_val = _args['axes_custom']
    else:
        max_val *= _args['axes_mag']

    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_box_aspect([1, 1, 1])
    ax.set_aspect('auto')

    if _args['azimuth'] is not False:
        ax.view_init(elev=_args['elevation'],
                     azim=_args['azimuth'])

    if _args['axes_no_fill']:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    if _args['hide_axes']:
        ax.set_axis_off()

    if _args['legend']:
        plt.legend()

    if _args['filename']:
        plt.savefig(_args['filename'], dpi=_args['dpi'])
        print('Saved', _args['filename'])

    if _args['show']:
        plt.show()

    plt.close()
    
def graficarCinematica(t, estados):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Extracting acceleration, velocity, and position
    accels = np.zeros((len(t), 3))
    for i in range(len(t)):
        accels[i] = ecuacionDiferencialParaDosCuerpos(t[i], estados[i])[3:]

    vels = estados[:, 3:6]
    poss = estados[:, :3]

    anorms = np.linalg.norm(accels, axis=1)
    vnorms = np.linalg.norm(vels, axis=1)
    rnorms = np.linalg.norm(poss, axis=1)

    # Acceleration
    axs[0].plot(t, accels[:, 0], 'r', label=r'$a_x$')
    axs[0].plot(t, accels[:, 1], 'g', label=r'$a_y$')
    axs[0].plot(t, accels[:, 2], 'b', label=r'$a_z$')
    axs[0].plot(t, anorms, 'm', label=r'$Norms$')
    axs[0].grid(linestyle='dotted')
    axs[0].set_ylabel(r'Aceleracion $(\dfrac{km}{s^2})$', fontsize=20)
    axs[0].legend(loc='upper center')

    # Velocity
    axs[1].plot(t, vels[:, 0], 'r', label=r'$v_x$')
    axs[1].plot(t, vels[:, 1], 'g', label=r'$v_y$')
    axs[1].plot(t, vels[:, 2], 'b', label=r'$v_z$')
    axs[1].plot(t, vnorms, 'm', label=r'$Norms$')
    axs[1].grid(linestyle='dotted')
    axs[1].set_ylabel(r'Velocidad $(\dfrac{km}{s})$', fontsize=20)
    axs[1].legend(loc='upper center')

    # Position
    axs[2].plot(t, poss[:, 0], 'r', label=r'$r_x$')
    axs[2].plot(t, poss[:, 1], 'g', label=r'$r_y$')
    axs[2].plot(t, poss[:, 2], 'b', label=r'$r_z$')
    axs[2].plot(t, rnorms, 'm', label=r'$Normas vectoriales$')
    axs[2].grid(linestyle='dotted')
    axs[2].set_ylabel(r'Posicion $(km)$', fontsize=20)
    axs[2].set_xlabel('Time (seconds)')
    axs[2].legend(loc='upper right')

    plt.tight_layout()
    plt.show()     

# Bloque de ejecución
if __name__ == '__main__':
    r0_norma = radioTerreste + 2500.0 # km
    v0_norma = (parametroGravitacionalTerrestre / r0_norma) ** 0.5 # km/s    
    # Definir estado inicial con posición y velocidad normalizadas
    estadoInicial = np.array([r0_norma, 0.0, 0.0, 0.0, v0_norma*1.1, 4.0])
    # Definir tiempo de propagación
    tiempoPropagacion = 100.0 * 360.0 # s
    dt = 100.0 # s
    # Calcular numero de pasos de propagación
    pasos = int(tiempoPropagacion / dt)
    # Inicializar vector de tiempo y matriz de estados
    ets = np.zeros((pasos, 1))
    estados = np.zeros((pasos, 6))
    estados[0] = estadoInicial
    
    # Realizar propagación de la órbita con numero de pasos para metodo de Runge-Kutta 4
    for paso in range(pasos - 1):
        # Se calcula el siguiente estado
        estados[paso + 1] = rungeKutta4Paso(
            ecuacionDiferencialParaDosCuerpos, ets[paso], estados[paso], dt)
        # Actualizar tiempo
        ets[paso + 1] = ets[paso] + dt
    
    print(estados)
    simularOrbita([estados], {'show': True})
    graficarCinematica(ets.flatten(), estados)
