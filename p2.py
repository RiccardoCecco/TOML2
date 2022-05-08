import random

from header import *  # IMPORTING HEADER FILE
import matplotlib.pyplot as plt
import gpkit
import cvxpy
import gpkit.nomials

def calc_n_d(d):
    n_d = (2 * d - 1) * C
    if d == 0:
        n_d = 1
    return n_d


def calc_i_d(d):
    if d == 0:
        i_d = C
    elif d == D:
        i_d = 0
    else:
        i_d = (2 * d + 1) / (2 * d - 1)
    return i_d


def calc_f_out(d):
    f_out = Fs * ((D ** 2 - (d ** 2) + 2 * d - 1) / (2 * d - 1))
    if d == D:
        f_out = Fs
    return f_out


def calc_f_b(d):
    return (C - abs(calc_i_d(d))) * calc_f_out(d)


def calc_f_i(d):
    f_i = Fs * ((D ** 2 - (d ** 2)) / (2 * d - 1))
    if d == 0:
        f_i = Fs * (D ** 2) * C
    return f_i


def calc_alphas(d):
    alpha1 = Tcs + Tal + 3 / 2 * Tps * ((Tps + Tal) / 2 + Tack + Tdata) * calc_f_b(d)
    alpha2 = calc_f_out(d) / 2
    alpha3 = ((Tps + Tal) / 2 + Tcs + Tal + Tack + Tdata) * calc_f_out(d) + (
            3 / 2 * Tps + Tack + Tdata) * calc_f_i(d) + 3 / 4 * Tps * calc_f_b(d)

    return alpha1, alpha2, alpha3


def calc_betas(d):
    beta1 = sum([1 / 2] * d)
    beta2 = sum([Tcw / 2 + Tdata] * d)

    return beta1, beta2


def energy_fun(tw):  # ENERGY FUNCTION
    return alpha_1 / tw + alpha_2 * tw + alpha_3


def delay_fun(tw):
    return beta_1 * tw + beta_2


if __name__ == "__main__":
    # PART 1 #
    time_1 = [1, 5, 10, 15, 20, 25]
    time_2 = [5, 10, 15, 20, 25]
    alpha_1, alpha_2, alpha_3 = 0.0, 0.0, 0.0
    x = np.linspace(Tw_min, Tw_max)

    prob1_solves = []
    np_L = np.linspace(100, 5000, 50)
    list_Lmax = [500, 750, 1000, 2500, 5000]
    tw_np = np.linspace(Tw_min, Tw_max)

    colours_plot = ['green', 'red', 'orange', 'purple', 'blue']
    size_colours = len(colours_plot)
    colour_index = 0

    for l_element in list_Lmax:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for t in time_2:
            x = gpkit.Variable("x")

            Fs = 1.0 / (t * 60 * 1000)
            alpha_1, alpha_2, alpha_3 = calc_alphas(1)
            beta_1, beta_2 = calc_betas(D)
            Tt_x = (x / (Tps + Tal)) * ((Tps + Tal) / 2) + Tack + Tdata
            E1_tx = (Tcs + Tal + Tt_x) * calc_f_out(1)

            obj_fun1 = alpha_1 / x + alpha_2 * x + alpha_3
            cons1 = beta_1 * x + beta_2
            cons2 = x
            cons3 = abs(calc_i_d(0)) * E1_tx
            constraints = [cons1 <= l_element, cons2 >= Tw_min, cons3 <= 1 / 4]
            prob1 = gpkit.Model(obj_fun1, constraints)
            solution = prob1.solve()
            #print(solution['variables']['x'], solution['cost'])
            prob1_solves.append(solution['cost'])
            plt.plot(tw_np, energy_fun(tw_np), color=colours_plot[colour_index % size_colours], label='E(Tw) for Fs('+str(t)+'min)')
            colour_index += 1
            ax.scatter(solution['variables'][x], solution['cost'], color="red")
            print("T(max)=", t)
            print("L(max)=", l_element)
            print("optimal value p* = ", solution['cost'])
            print("optimal var: T_w = ", solution['variables'][x], "\n\n")

        plt.xlabel('Tw (ms)')
        plt.ylabel('Energy (J)')
        plt.legend(loc='upper right')
        plt.title("L_max="+str(l_element))
        plt.savefig("2-"+str(l_element)+".jpg")
        plt.show()

    colour_index = 0
    list_Ebudget = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    prob2_solves = []

    for e_element in list_Ebudget:
        x = gpkit.Variable("x")
        Fs = 1.0 / (5 * 60 * 1000)
        alpha_1, alpha_2, alpha_3 = calc_alphas(1)
        beta_1, beta_2 = calc_betas(D)
        Tt_x = (x / (Tps + Tal)) * ((Tps + Tal) / 2) + Tack + Tdata
        E1_tx = (Tcs + Tal + Tt_x) * calc_f_out(1)

        obj_fun2 = beta_1 * x + beta_2
        cons1 = alpha_1 / x + alpha_2 * x + alpha_3
        cons2 = x
        cons3 = abs(calc_i_d(0)) * E1_tx
        constraints = [cons1 <= e_element, cons2 >= Tw_min, cons3 <= (1 / 4)]
        prob2 = gpkit.Model(obj_fun2, constraints)
        solution = prob2.solve()
        #print(solution['variables'][x], solution['cost'])
        prob2_solves.append([solution['variables'][x], solution["cost"]])
        plt.plot(tw_np, delay_fun(tw_np), color=colours_plot[colour_index % size_colours])
        colour_index += 1
        ax.scatter(solution['variables'][x], solution['cost'], color="red")
        print("L(max)=", e_element)
        print("optimal value p* = ", solution['cost'])
        print("optimal var: T_w = ", solution['variables'][x], "\n\n")


    plt.xlabel('Tw (ms)')
    plt.ylabel('Delay')
    plt.title("All Ebudgets")
    plt.savefig("allebudgets.jpg")
    plt.show()


