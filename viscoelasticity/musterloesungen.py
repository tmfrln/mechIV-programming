import numpy as np
import random
from scipy.optimize import fsolve


def test_stress(stress_student, n=5):
    """
    Vergleich von Lösung der Studierenden mit Musterlösung für n zufällige Eingaben.

    Dieser Vergleich findet in einer Funktion statt um den globalen Namespace möglichst nicht zu ändern.

    """

    # Musterlösung für die Implementierung der Funktion
    def stress_solution(eps, epsvn, dt, Einf, E, eta):
        # Lösung der Evolutionsgleichung für die viskosen Dehnung
        epsv = (eta*epsvn + dt*E*eps)/(eta+dt*E)

        # Resultierende Spannung
        sig = Einf*eps+E*(eps-epsv)

        return sig, epsv

    E = 200.
    Einf = 200.
    eta = 100.

    numErrs = 0

    for i in range(n):
        eps = random.random()
        epsvn = random.random()
        dt = random.random()

        sig_sol, epsv_sol = stress_solution(eps, epsvn, dt, Einf, E, eta)

        try:
            sig, epsv = stress_student(eps, epsvn, dt, Einf, E, eta)

            assert np.isclose(sig, sig_sol)
            assert np.isclose(epsv, epsv_sol)
        except AssertionError:
            numErrs += 1

    if numErrs == 0:
        print('Die Funktion ist korrekt.')
    else:
        print('Die Funktion ist nicht korrekt.')


def test_residuum(res_student, n=5):
    """
    Vergleich von Lösung der Studierenden mit Musterlösung für n zufällige Eingaben.

    Dieser Vergleich findet in einer Funktion statt um den globalen Namespace möglichst nicht zu ändern.

    """

    # Musterlösung für die Implementierung der Funktion
    def res_solution(eps, sig, epsv, epsvn, dt, Einf, E, eta):

        # Hilfswerte
        siginf = Einf*eps
        sigv = E*(eps-epsv)

        # Residuum
        r1 = sig - siginf - sigv
        r2 = (epsv - epsvn)/dt - sigv/eta

        return np.array([r1, r2])

    E = 200.
    Einf = 200.
    eta = 100.

    numErrs = 0

    for i in range(n):
        eps = random.random()
        sig = random.random()
        epsv = random.random()
        epsvn = random.random()
        dt = random.random()

        r_sol = res_solution(eps, sig, epsv, epsvn, dt, Einf, E, eta)

        try:
            r = res_student(eps, sig, epsv, epsvn, dt, Einf, E, eta)

            assert np.allclose(r, r_sol)
        except AssertionError:
            numErrs += 1

    if numErrs == 0:
        print('Die Funktion ist korrekt.')
    else:
        print('Die Funktion ist nicht korrekt.')


def test_stress_res(stress_res_student, n=5):
    def res(eps, sig, epsv, epsvn, dt, Einf, E, eta):

        # Hilfswerte
        siginf = Einf*eps
        sigv = E*(eps-epsv)

        # Residuum
        r1 = sig - siginf - sigv
        r2 = (epsv - epsvn)/dt - sigv/eta

        return np.array([r1, r2])

    def stress_res_solution(eps, epsvn, dt, Einf, E, eta, x0=np.array([0.,0.])):
        """Lösung der Residuumsgleichung nach den Spannungen und viskosen Verzerrungen"""

        # f ist die Funktion, die das Residuum für verschiedene Werte von
        # sigma und epsv auswertet.
        f = lambda x: res(eps, x[0], x[1], epsvn, dt, Einf, E, eta)

        # Mit der Funktion fsolve wird eine Lösung des Gleichungssystems
        # gesucht.
        x = fsolve(f, x0)

        # Wir überprüfen noch einmal, dass die gefundene Lösung x die
        # Residuumsgleichung tatsächlich löst
        assert np.allclose(f(x), np.zeros_like(x))

        # Wie zuvor geben wir hier Spannung und interne Variable einzeln zurück
        return x[0], x[1]


    E = 200.
    Einf = 200.
    eta = 100.

    numErrs = 0

    for i in range(n):
        eps = random.random()
        epsvn = random.random()
        dt = random.random()

        sig_sol, epsv_sol = stress_res_solution(eps, epsvn, dt, Einf, E, eta)

        try:
            sig, epsv = stress_res_student(eps, epsvn, dt, Einf, E, eta)

            assert np.isclose(sig, sig_sol)
            assert np.isclose(epsv, epsv_sol)

        except AssertionError:
            numErrs += 1

    if numErrs == 0:
        print('Die Funktion ist korrekt.')
    else:
        print('Die Funktion ist nicht korrekt.')


def test_strain_res(strain_res_student, n=5):
    def res(eps, sig, epsv, epsvn, dt, Einf, E, eta):

        # Hilfswerte
        siginf = Einf*eps
        sigv = E*(eps-epsv)

        # Residuum
        r1 = sig - siginf - sigv
        r2 = (epsv - epsvn)/dt - sigv/eta

        return np.array([r1, r2])

    def strain_res_solution(sig, epsvn, dt, Einf, E, eta, x0=np.array([0.,0.])):
        """Lösung der Residuumsgleichung nach Dehnung und interner Variable"""

        # f ist die Funktion, die das Residuum für verschiedene Werte von
        # eps und epsv auswertet.
        f = lambda x: res(x[0], sig, x[1], epsvn, dt, Einf, E, eta)

        # Mit der Funktion fsolve wird eine Lösung des Gleichungssystems
        # gesucht.
        x = fsolve(f, x0)

        # Wir überprüfen noch einmal, dass die gefundene Lösung x die
        # Residuumsgleichung tatsächlich löst
        assert np.allclose(f(x), np.zeros_like(x))

        # Wie zuvor geben wir hier Dehnung und interne Variable einzeln zurück
        return x[0], x[1]


    E = 200.
    Einf = 200.
    eta = 100.

    numErrs = 0

    for i in range(n):
        sig = random.random()
        epsvn = random.random()
        dt = random.random()

        eps_sol, epsv_sol = strain_res_solution(sig, epsvn, dt, Einf, E, eta)

        try:
            eps, epsv = strain_res_student(sig, epsvn, dt, Einf, E, eta)

            assert np.isclose(eps, eps_sol)
            assert np.isclose(epsv, epsv_sol)

        except AssertionError:
            numErrs += 1

    if numErrs == 0:
        print('Die Funktion ist korrekt.')
    else:
        print('Die Funktion ist nicht korrekt.')
