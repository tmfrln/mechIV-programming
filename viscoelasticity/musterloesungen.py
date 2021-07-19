"""
Diese Datei enthält Musterlösungen und Tests für die Übung zu Viskoelastizität.

Die Tests werden im Übungsnotebook importiert und ausgeführt. So erhalten
die Studierenden direkt Feedback zu ihrer Lösung.

"""

import numpy as np
import random
from scipy.optimize import fsolve


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
    def stress_solution(eps, epsvn, dt, Einf, E, eta):
        # Lösung der Evolutionsgleichung für die viskosen Dehnung
        epsv = (eta*epsvn + dt*E*eps)/(eta+dt*E)

        # Resultierende Spannung
        sig = Einf*eps+E*(eps-epsv)

        return sig, epsv

    # Tests - Materialparameter
    E = 200.
    Einf = 150.
    eta = 100.

    # Anzahl der fehlgeschlagenen Tests
    numErrs = 0

    # Schleife über n Checks
    for i in range(n):
        # Zufällige Eingaben
        eps = random.random()
        epsvn = random.random()
        dt = random.random()

        # Musterlösung
        sig_sol, epsv_sol = stress_solution(eps, epsvn, dt, Einf, E, eta)

        try:
            # Lösung der Studierenden - ein Fehler entsteht zB auch wenn das
            # Interface falsch implementiert wurde
            sig, epsv = stress_student(eps, epsvn, dt, Einf, E, eta)

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
    def res_solution(eps, sig, epsv, epsvn, dt, Einf, E, eta):

        # Hilfswerte
        siginf = Einf*eps
        sigv = E*(eps-epsv)

        # Residuum
        r1 = sig - siginf - sigv
        r2 = (epsv - epsvn)/dt - sigv/eta

        return np.array([r1, r2])

    # Tests - Materialparameter
    E = 200.
    Einf = 150.
    eta = 100.

    # Anzahl der fehlgeschlagenen Tests
    numErrs = 0

    # Schleife über n Checks
    for i in range(n):
        # Zufällige Eingaben
        eps = random.random()
        sig = random.random()
        epsv = random.random()
        epsvn = random.random()
        dt = random.random()

        # Musterlösung
        r_sol = res_solution(eps, sig, epsv, epsvn, dt, Einf, E, eta)

        try:
            # Lösung der Studierenden - ein Fehler entsteht zB auch wenn das
            # Interface falsch implementiert wurde
            r = res_student(eps, sig, epsv, epsvn, dt, Einf, E, eta)

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
    def res(eps, sig, epsv, epsvn, dt, Einf, E, eta):

        # Hilfswerte
        siginf = Einf*eps
        sigv = E*(eps-epsv)

        # Residuum
        r1 = sig - siginf - sigv
        r2 = (epsv - epsvn)/dt - sigv/eta

        return np.array([r1, r2])

    # Musterlösung für die Implementierung der Funktion
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

    # Tests - Materialparameter
    E = 200.
    Einf = 150.
    eta = 100.

    # Anzahl der fehlgeschlagenen Tests
    numErrs = 0

    # Schleife über n Checks
    for i in range(n):
        # Zufällige Eingaben
        eps = random.random()
        epsvn = random.random()
        dt = random.random()

        # Musterlösung
        sig_sol, epsv_sol = stress_res_solution(eps, epsvn, dt, Einf, E, eta)

        try:
            # Lösung der Studierenden - ein Fehler entsteht zB auch wenn das
            # Interface falsch implementiert wurde
            sig, epsv = stress_res_student(eps, epsvn, dt, Einf, E, eta)

            assert np.isclose(sig, sig_sol)
            assert np.isclose(epsv, epsv_sol)

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
    def res(eps, sig, epsv, epsvn, dt, Einf, E, eta):

        # Hilfswerte
        siginf = Einf*eps
        sigv = E*(eps-epsv)

        # Residuum
        r1 = sig - siginf - sigv
        r2 = (epsv - epsvn)/dt - sigv/eta

        return np.array([r1, r2])

    # Musterlösung für die Implementierung der Funktion
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

    # Tests - Materialparameter
    E = 200.
    Einf = 150.
    eta = 100.

    # Anzahl der fehlgeschlagenen Tests
    numErrs = 0

    # Schleife über n Checks
    for i in range(n):
        # Zufällige Eingaben
        sig = random.random()
        epsvn = random.random()
        dt = random.random()
        
        # Musterlösung
        eps_sol, epsv_sol = strain_res_solution(sig, epsvn, dt, Einf, E, eta)

        try:
            # Lösung der Studierenden - ein Fehler entsteht zB auch wenn das
            # Interface falsch implementiert wurde
            eps, epsv = strain_res_student(sig, epsvn, dt, Einf, E, eta)

            assert np.isclose(eps, eps_sol)
            assert np.isclose(epsv, epsv_sol)

        except BaseException:
            numErrs += 1

    if numErrs == 0:
        print('Die Funktion ist korrekt.')
    else:
        print('Die Funktion ist nicht korrekt.')
