import unittest

import numpy as np

from physics import RollingBodySimulator


class AnalyticalSolutionTests(unittest.TestCase):
    def test_analytical_solution_matches_slip_reversal(self):
        simulator = RollingBodySimulator(
            m=0.5,
            R=0.05,
            mu=0.08,
            alpha_deg=25.0,
            body_key="solid_sphere",
        )
        numerical = simulator.simulate(x0=0.0, v0=0.0, omega0=30.0, t_max=1.2)
        analytical = simulator.analytical_solution(
            numerical["t"], x0=0.0, v0=0.0, omega0=30.0
        )

        self.assertIsNone(numerical["t_transition"])
        self.assertEqual(set(numerical["modes"]), {"slipping"})
        self.assertLess(numerical["v_c"][0], 0.0)
        self.assertGreater(numerical["v_c"][-1], 0.0)
        np.testing.assert_allclose(numerical["x"], analytical["x"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(numerical["v"], analytical["v"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            numerical["omega"], analytical["omega"], rtol=0, atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()
