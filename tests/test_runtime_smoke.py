import unittest

import agen_lab


class RuntimeSmokeTest(unittest.TestCase):
    def test_semantic_runtime_version(self):
        self.assertEqual(agen_lab.CORE_VERSION, "2.42")

    def test_integrated_agent_is_public(self):
        self.assertTrue(hasattr(agen_lab, "IntegratedCognitiveAgent"))

    def test_replacement_plan_ranking_is_public(self):
        self.assertTrue(hasattr(agen_lab, "SpatialReplanReliabilityRanker"))

    def test_capability_boundary_is_explicit(self):
        store = agen_lab.SpatialSceneStore()
        caps = store.state()
        self.assertTrue(caps["reliability_aware_replacement_plan_ranking"])
        self.assertFalse(caps["reliability_aware_spatial_replanning"])


if __name__ == "__main__":
    unittest.main()
