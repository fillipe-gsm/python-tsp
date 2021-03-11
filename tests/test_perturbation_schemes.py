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

    def test_ps4_returns_correct_num_neighbors(self):
        """PS4 is more intricate. Doing by hand, the total perturbations here
        is equal to 20, so its neighborhood size is larger than PS3.
        """

        all_neighbors = list(perturbation_schemes.ps4_gen(self.x))
        assert len(all_neighbors) == 20

    def test_ps5_returns_correct_num_neighbors(self):
        """PS5 has sum_{i=1}^{n} (n - i) elements.
        In this case with 4 items (the first one is fixed), we get 6 neighbors
        """

        all_neighbors = list(perturbation_schemes.ps5_gen(self.x))
        assert len(all_neighbors) == 6

    def test_ps6_returns_correct_num_neighbors(self):
        """PS6 is intricate as PS4, with the same number of neighbors.
        Here, it is equal to 20 as well.
        """

        all_neighbors = list(perturbation_schemes.ps6_gen(self.x))
        assert len(all_neighbors) == 20

    def test_two_opt_returns_correct_num_neighbors(self):
        """2-top has the same number of perturbations as PS5.
        Thus, in this case with 4 items (the one fixed), we get 6 neighbors.
        """
        all_neighbors = list(perturbation_schemes.two_opt_gen(self.x))
        assert len(all_neighbors) == 6
