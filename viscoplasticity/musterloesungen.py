"""
Diese Datei enthält Musterlösungen und Tests für die Übung zu Viskoelastizität.

Die Tests werden im Übungsnotebook importiert und ausgeführt. So erhalten
die Studierenden direkt Feedback zu ihrer Lösung.

"""

import numpy as np
import random
from scipy.optimize import fsolve
import warnings

# Filter out the specific RuntimeWarning from scipy.optimize.minpack
warnings.filterwarnings("ignore", message="The iteration is not making good progress")

def test_stress(stress_student, n=5):
    """
    Teste die Funktion für die Bestimmung der Spannungen.

    Es wird überprüft, ob die Funktion stress_student für gegebene Dehnung die
    korrekte Spannung und interne Variable bestimmt. Dazu werden n Sätze von
    zufälligen Eingaben generiert und die Ausgaben der Funktion mit der
    hinterlegten Musterlösung abgeglichen.

    Rundungsfehler sollten nicht zu einer Ablehnung der Lösung führen.

    """

    # Musterlösung für die Implementierung der Funktion
    def stress_solution(eps, epsvp_n, dt, E, sigy, eta):
        epsvp_tr = epsvp_n
        sig_tr = E*(eps-epsvp_tr)
        phi_tr = np.abs(sig_tr)-sigy

        if phi_tr <= 0:
            epsvp = epsvp_tr
            sig = sig_tr
            dlambda = 0 # Nur für Sanity check
        else:
            dlambda = dt/(eta+E*dt)*phi_tr
            epsvp = epsvp_n + dlambda*np.sign(sig_tr)
            sig = E*(eps-epsvp)

        # # Sanity check
        #assert np.isclose(epsvp, epsvp_n + dlambda*np.sign(sig))

        return sig, epsvp

    # Tests - Materialparameter
    E = 200.
    eta = 50.
    sigy = 10.

    # Anzahl der fehlgeschlagenen Tests
    numErrs = 0

    # Schleife über n Checks
    for i in range(n):
        # Zufällige Eingaben
        eps = random.random()
        epsvn = random.random()
        dt = random.random()

        # Musterlösung
        sig_sol, epsv_sol = stress_solution(eps, epsvn, dt, E, sigy, eta)

        try:
            # Lösung der Studierenden - ein Fehler entsteht zB auch wenn das
            # Interface falsch implementiert wurde
            sig, epsv = stress_student(eps, epsvn, dt, E, sigy, eta)

            assert np.isclose(sig, sig_sol)
            assert np.isclose(epsv, epsv_sol)
        except BaseException:
            numErrs += 1

    if numErrs == 0:
        print('Die Funktion ist korrekt.')
    else:
        print('Die Funktion ist nicht korrekt.')


def test_residuum(res_student, n=5):
    """
    Teste die Funktion für die Bestimmung des Residuums.

    Es wird überprüft, ob die Funktion res_student für gegebene Dehnung,
    Spannung und interne Variable das korrekte Residuum bestimmt. Dazu werden
    n Sätze von zufälligen Eingaben generiert und die Ausgaben der Funktion mit
    der hinterlegten Musterlösung abgeglichen.

    Rundungsfehler sollten nicht zu einer Ablehnung der Lösung führen.

    """

    # Musterlösung für die Implementierung der Funktion
    # Residuum für gegebene Dehnungen, Spannungen und viskose Verzerrungen
    def res_solution(eps, sig, epsvp, dlambda, phi, epsvp_n, dt, E, sigy, eta):

        r1 = sig - E*(eps-epsvp)
        r2 = epsvp - epsvp_n - dlambda*np.sign(sig)
        r3 = dlambda*eta - dt*np.max(np.array([0, phi]))
        r4 = phi-np.abs(sig)+sigy

        return np.array([r1, r2, r3, r4])


    # Tests - Materialparameter
    E = 200.
    eta = 50.
    sigy = 10.

    # Anzahl der fehlgeschlagenen Tests
    numErrs = 0

    # Schleife über n Checks
    for i in range(n):
        # Zufällige Eingaben
        eps = random.random()
        sig = random.random()
        epsvp = random.random()
        dlambda = random.random()
        phi = random.random()
        epsvp_n = random.random()
        dt = random.random()

        # Musterlösung
        r_sol = res_solution(eps, sig, epsvp, dlambda, phi, epsvp_n, dt, E, sigy, eta)

        try:
            # Lösung der Studierenden - ein Fehler entsteht zB auch wenn das
            # Interface falsch implementiert wurde
            r = res_student(eps, sig, epsvp, dlambda, phi, epsvp_n, dt, E, sigy, eta)

            assert np.allclose(r, r_sol)
        except BaseException:
            numErrs += 1

    if numErrs == 0:
        print('Die Funktion ist korrekt.')
    else:
        print('Die Funktion ist nicht korrekt.')


def test_stress_res(stress_res_student, n=5):
    """
    Teste die Funktion für die Bestimmung der Spannung aus dem Residuum.

    Es wird überprüft, ob die Funktion stress_res_student für gegebene Dehnung
    die korrekte Spannung und interne Variable bestimmt. Dazu werden n Sätze
    von zufälligen Eingaben generiert und die Ausgaben der Funktion mit der
    hinterlegten Musterlösung abgeglichen.

    Rundungsfehler sollten nicht zu einer Ablehnung der Lösung führen.

    """

    # Korrektes Residuum ist Voraussetzung für Lösung
    def res(eps, sig, epsvp, dlambda, phi, epsvp_n, dt, E, sigy, eta):

        r1 = sig - E*(eps-epsvp)
        r2 = epsvp - epsvp_n - dlambda*np.sign(sig)
        r3 = dlambda*eta - dt*np.max(np.array([0, phi]))
        r4 = phi-np.abs(sig)+sigy

        return np.array([r1, r2, r3, r4])

    # Musterlösung für die Implementierung der Funktion
    # Lösung der Residuumsgleichung nach den Spannungen und viskosen Verzerrungen
    def stress_res_solution(eps, epsvp_n, dt, E, sigy, eta, x0 = np.zeros(4)):
        f = lambda x: res(eps, x[0], x[1], x[2], x[3], epsvp_n, dt, E, sigy, eta)
        sol = fsolve(f, x0)

        # assert np.allclose(f(sol), np.zeros_like(sol))

        return sol[0:2]

    # Tests - Materialparameter
    E = 200.
    eta = 50.
    sigy = 10.

    # Anzahl der fehlgeschlagenen Tests
    numErrs = 0

    # Schleife über n Checks
    for i in range(n):
        # Zufällige Eingaben
        eps = random.random()
        epsvn = random.random()
        dt = random.random()

        # Musterlösung
        sig_sol, epsv_sol = stress_res_solution(eps, epsvn, dt, E, sigy, eta)

        try:
            # Lösung der Studierenden - ein Fehler entsteht zB auch wenn das
            # Interface falsch implementiert wurde
            sig, epsv = stress_res_student(eps, epsvn, dt, E, sigy, eta)

            assert np.isclose(sig, sig_sol)
            assert np.isclose(epsv, epsv_sol)
        except AssertionError as e:
            numErrs += 1
            print("Die Funktion hat die korrekte Lösung nicht gefunden. Dies kann an den zufälligen Ausgangswerten liegen. Führe die Zelle noch einmal aus.")
        except BaseException:
            numErrs += 1

    if numErrs == 0:
        print('Die Funktion ist korrekt.')
    else:
        print('Die Funktion ist nicht korrekt.')


def test_strain_res(strain_res_student, n=5):
    """
    Teste die Funktion für die Bestimmung der Dehnung aus dem Residuum.

    Es wird überprüft, ob die Funktion strain_res_student für gegebene Dehnung
    die korrekte Spannung und interne Variable bestimmt. Dazu werden n Sätze
    von zufälligen Eingaben generiert und die Ausgaben der Funktion mit der
    hinterlegten Musterlösung abgeglichen.

    Rundungsfehler sollten nicht zu einer Ablehnung der Lösung führen.

    """

    # Korrektes Residuum ist Voraussetzung für Lösung
    def res(eps, sig, epsvp, dlambda, phi, epsvp_n, dt, E, sigy, eta):

        r1 = sig - E*(eps-epsvp)
        r2 = epsvp - epsvp_n - dlambda*np.sign(sig)
        r3 = dlambda*eta - dt*np.max(np.array([0, phi]))
        r4 = phi-np.abs(sig)+sigy

        return np.array([r1, r2, r3, r4])

    # Musterlösung für die Implementierung der Funktion
    # Lösung der Residuumsgleichung nach den Spannungen und viskosen Verzerrungen
    def strain_res_solution(sig, epsvp_n, dt, E, sigy, eta, x0 = np.zeros(4)):
        f = lambda x: res(x[0], sig, x[1], x[2], x[3], epsvp_n, dt, E, sigy, eta)
        sol = fsolve(f, x0)

        # assert np.allclose(f(sol), np.zeros_like(sol))

        return sol[0:2]

    # Tests - Materialparameter
    E = 200.
    eta = 50.
    sigy = 10.

    # Anzahl der fehlgeschlagenen Tests
    numErrs = 0

    # Schleife über n Checks
    for i in range(n):
        # Zufällige Eingaben
        sig = random.random()
        epsvn = random.random()
        dt = random.random()

        # Musterlösung
        eps_sol, epsv_sol = strain_res_solution(sig, epsvn, dt, E, sigy, eta)

        try:
            # Lösung der Studierenden - ein Fehler entsteht zB auch wenn das
            # Interface falsch implementiert wurde
            eps, epsv = strain_res_student(sig, epsvn, dt, E, sigy, eta)

            assert np.isclose(eps, eps_sol)
            assert np.isclose(epsv, epsv_sol)

        except AssertionError as e:
            numErrs += 1
            print("AssertionError - die Funktion hat die korrekte Lösung nicht gefunden. Dies kann an den zufälligen Ausgangswerten liegen, versuch es noch einmal.")

        except BaseException:
            numErrs += 1

    if numErrs == 0:
        print('Die Funktion ist korrekt.')
    else:
        print('Die Funktion ist nicht korrekt.')
