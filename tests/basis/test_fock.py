from qsim.basis.fock import Fock


class TestFock:
    def test_bitflip_proba(self):
        nbar, N = 4, 25
        basis = Fock(nbar=nbar, d=N)
        names = ['0', '1', '+', '-']
        states = [
            basis.data.zero,
            basis.data.one,
            basis.data.evencat,
            basis.data.oddcat,
        ]
        jx_results = [1, -1, 0, 0]
        cpm_results = [0.5, -0.5, 0, 0]
        tol = 1e-4

        for i in range(len(states)):
            # print(f'state: {names[i]}')
            # assert abs(basis.bitflip_proba(states[i]) - px_results[i]) < tol
            assert abs(basis.data.jx(states[i]) - jx_results[i]) < tol
            assert abs(basis.data.cpm(states[i]) - cpm_results[i]) < tol
