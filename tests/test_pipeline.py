import unittest

from src.build_pipeline import FootballPredictionPipeline


class PipelineSmokeTests(unittest.TestCase):
    def test_run_without_scrapers(self):
        pipe = FootballPredictionPipeline()
        result = pipe.run_full_pipeline(run_scrapers=False, stage_load_historical=False, stage_train_models=False, competitions=['PL'])
        self.assertIn(result.get('status'), ['completed','failed'])
        # at least stage1 should have executed
        self.assertIn('data_collection', result.get('stages_completed', []))

    def test_run_with_scrapers(self):
        pipe = FootballPredictionPipeline()
        result = pipe.run_full_pipeline(run_scrapers=True, stage_load_historical=False, stage_train_models=False)
        self.assertIn(result.get('status'), ['completed','failed'])
        self.assertIn('external_scrape', result.get('stages_completed', []))


if __name__ == "__main__":
    unittest.main()
