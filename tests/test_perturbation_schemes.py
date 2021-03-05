from python_tsp.heuristics import perturbation_schemes


class TestPerturbationSchemes:
    x = [0, 1, 2, 3, 4]

    def test_ps1_returns_correct_num_neighbors(self):
        """PS1 has n - 1 swaps.
        But since we fix the first element as origin, it leads to n - 2 = 3 in
        this case.
        """

        all_neighbors = list(perturbation_schemes.ps1_gen(self.x))
        assert len(all_neighbors) == 3

    def test_ps2_returns_correct_num_neighbors(self):
        """PS2 has n * (n - 1) / 2 swaps.
        But since we fix the first element as origin, it leads to
        (n - 1) * (n - 2) / 2 = 6 in this case.
        """

        all_neighbors = list(perturbation_schemes.ps2_gen(self.x))
        assert len(all_neighbors) == 6

    def test_ps3_returns_correct_num_neighbors(self):
        """PS3 has n * (n - 1) elements.
        But since we fix the first element as origin, it leads to
        (n - 1) * (n - 2) = 12 in this case.
        """

        all_neighbors = list(perturbation_schemes.ps3_gen(self.x))
        assert len(all_neighbors) == 12
