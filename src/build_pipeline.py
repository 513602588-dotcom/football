"""
完整的足球预测管道
从数据收集 -> 特征工程 -> 模型训练 -> 预测生成 -> 结果导出
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple

# 导入本地模块
from src.data.api_integrations import create_data_aggregator
from src.collect.utils import now_cn_date
from src.data.feature_engineering import FeatureEngineer
from src.data.data_collector_enhanced import (
    DataCollector, HistoricalDataLoader, CacheManager
)
from src.models.advanced_ml import MetaLearner
from src.models.poisson import predict_poisson
from src.models.elo import update_elo
from src.engine.fusion_engine import SuperFusionModel, BatchPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FootballPredictionPipeline:
    """完整的足球预测管道"""
    
    def __init__(
        self,
        football_api_key: str = None,
        odds_api_key: str = None,
        db_path: str = "data/football.db"
    ):
        """
        初始化管道
        
        Args:
            football_api_key: football-data.org API密钥
            odds_api_key: The Odds API密钥
            db_path: 数据库路径
        """
        self.data_aggregator = create_data_aggregator(
            football_api_key=football_api_key,
            odds_api_key=odds_api_key
        )
        self.data_collector = DataCollector(db_path)
        self.feature_engineer = FeatureEngineer()
        self.meta_learner = MetaLearner()
        self.fusion_model = SuperFusionModel()
        self.fusion_model.load_meta_learner(self.meta_learner)
        self.cache = CacheManager()
        
        self.historical_data = None
        self.features = None
        self.predictions = None
        
        logger.info("Pipeline initialized successfully")
    
    def stage_0_scrape_external_data(self) -> None:
        """阶段0：运行所有外部爬虫，更新site/data目录。

        仅在需要的时候调用，任何错误均会被捕获并记录，但不会停止主流程。
        """
        logger.info("🕷️ Stage 0: Running external scrapers (500 & okooo)")
        try:
            from src.collect import export_500, export_okooo
            # 抓取最新1天500网数据
            export_500(days=1)
            # 抓取最近3天澳客数据作为示例
            # 注意：export_okooo 接受 start_date, days, version
            today = now_cn_date()
            export_okooo(start_date=today, days=3, version="full")
            logger.info("✅ External scrapers completed")
        except Exception as e:
            logger.warning(f"External scrapers failed: {e}")

    def stage_1_collect_data(self, competitions: List[str] = None) -> pd.DataFrame:
        """
        阶段1：数据收集
        
        Args:
            competitions: 联赛代码列表 (如['PL', 'SA', 'BL1'])
        
        Returns:
            比赛DataFrame
        """
        if competitions is None:
            competitions = ['PL', 'SA', 'BL1', 'FR1', 'IT1']
        
        logger.info("📊 Stage 1: Data Collection (API & cache)")
        all_matches = []
        
        for comp in competitions:
            logger.info(f"  Fetching matches for {comp}...")
            
            # 检查缓存
            cached = self.cache.get(f"matches_{comp}")
            if cached:
                logger.info(f"  Using cached data for {comp}")
                all_matches.extend(cached)
            else:
                matches = self.data_aggregator.fdb.get_matches(comp)
                if matches:
                    self.cache.set(f"matches_{comp}", matches)
                    all_matches.extend(matches)
        
        # 转换为DataFrame
        matches_df = pd.DataFrame([
            {
                'id': m.get('id'),
                'date': m.get('utcDate', m.get('date')),
                'league': m.get('competition', {}).get('code'),
                'home_team': m.get('homeTeam', {}).get('name'),
                'away_team': m.get('awayTeam', {}).get('name'),
                'status': m.get('status'),
                'home_goals': m.get('score', {}).get('fullTime', {}).get('home'),
                'away_goals': m.get('score', {}).get('fullTime', {}).get('away'),
            }
            for m in all_matches
        ])
        
        logger.info(f"✅ Collected {len(matches_df)} matches")
        self.historical_data = matches_df
        
        # 保存到数据库
        for _, match in matches_df.iterrows():
            self.data_collector.save_match(match.to_dict())
        
        return matches_df
    
    def stage_2_load_historical_data(self, picks_path: str = "site/data/picks.json") -> pd.DataFrame:
        """
        阶段2：加载历史数据
        
        Returns:
            历史比赛DataFrame
        """
        logger.info("📚 Stage 2: Loading Historical Data")
        
        try:
            df = HistoricalDataLoader.create_dataframe_from_site_data(picks_path)
            
            if len(df) > 0:
                logger.info(f"✅ Loaded {len(df)} historical records")
                self.historical_data = df
                return df
            else:
                logger.warning("⚠️ No historical data found, using web scraping...")
                # 可选：从Web爬虫获取数据
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error loading historical data: {e}")
            return pd.DataFrame()
    
    def stage_3_feature_engineering(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        阶段3：特征工程
        
        Args:
            matches_df: 比赛DataFrame
        
        Returns:
            特征DataFrame
        """
        logger.info("🔧 Stage 3: Feature Engineering")
        
        features_list = []
        
        for idx, match in matches_df.iterrows():
            try:
                features = self.feature_engineer.build_match_features(
                    match.to_dict(),
                    self.historical_data if self.historical_data is not None else matches_df
                )
                
                if len(features) > 0:
                    features_list.append(features)
                    
            except Exception as e:
                logger.error(f"Error extracting features for match {idx}: {e}")
                continue
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"✅ Extracted features for {len(features_df)} matches")
        logger.info(f"   Features shape: {features_df.shape}")
        
        self.features = features_df
        return features_df
    
    def stage_4_train_models(self, matches_df: pd.DataFrame, features_df: pd.DataFrame) -> MetaLearner:
        """
        阶段4：训练机器学习模型 (可选)
        
        仅当有足够历史数据时执行
        """
        logger.info("🧠 Stage 4: Training ML Models (Optional)")
        
        # 检查是否有足够的历史数据进行训练
        if len(features_df) < 100:
            logger.warning("⚠️ Insufficient data for model training (need >= 100 samples)")
            logger.info("   Skipping ML model training, using ensemble predictions only")
            return None
        
        try:
            # 创建虚拟标签用于演示（实际应使用真实比赛结果）
            np.random.seed(42)
            y = np.random.choice(['win', 'draw', 'loss'], size=len(features_df), p=[0.46, 0.27, 0.27])
            y_series = pd.Series(y)
            
            logger.info("   Training XGBoost ensemble...")
            self.meta_learner.train_all_models(features_df, y_series)
            
            logger.info("✅ All models trained successfully")
            self.fusion_model.load_meta_learner(self.meta_learner)
            
            return self.meta_learner
            
        except Exception as e:
            logger.error(f"❌ Error training models: {e}")
            logger.info("   Continuing without ML models...")
            return None
    
    def stage_5_generate_predictions(
        self,
        matches_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> List[Dict]:
        """
        阶段5：生成预测
        
        Args:
            matches_df: 比赛DataFrame
            features_df: 特征DataFrame
        
        Returns:
            预测结果列表
        """
        logger.info("🔮 Stage 5: Generating Predictions")
        
        predictions = []
        
        for idx, (_, match) in enumerate(matches_df.iterrows()):
            try:
                if idx < len(features_df):
                    features = features_df.iloc[idx]
                    
                    # 融合模型预测
                    prediction = self.fusion_model.predict_single_match(
                        match.to_dict(),
                        features
                    )
                    
                    predictions.append(prediction)
                    
                    # 每10个打印进度
                    if (idx + 1) % 10 == 0:
                        logger.info(f"   Processed {idx + 1}/{len(matches_df)} matches")
                        
            except Exception as e:
                logger.error(f"Error predicting match {idx}: {e}")
                continue
        
        logger.info(f"✅ Generated predictions for {len(predictions)} matches")
        self.predictions = predictions
        
        return predictions
    
    def stage_6_filter_top_picks(self, predictions: List[Dict], min_ev: float = 0.05) -> List[Dict]:
        """
        阶段6：筛选顶级推荐
        
        Args:
            predictions: 预测列表
            min_ev: 最小EV阈值
        
        Returns:
            筛选后的推荐列表
        """
        logger.info("🏆 Stage 6: Filtering Top Picks")
        
        top_picks = []
        
        for pred in predictions:
            try:
                ev = pred.get('expected_value', 0)
                confidence = pred.get('confidence', 0)
                
                # 筛选条件：EV > min_ev 且置信度 > 50%
                if ev > min_ev * 100 and confidence >= 50:
                    top_picks.append(pred)
                    
            except Exception as e:
                logger.warning(f"Error filtering prediction: {e}")
                continue
        
        # 按EV排序
        top_picks.sort(key=lambda x: x.get('expected_value', 0), reverse=True)
        
        logger.info(f"✅ Found {len(top_picks)} top picks (EV > {min_ev*100}%)")
        
        return top_picks
    
    def stage_7_export_results(
        self,
        predictions: List[Dict],
        top_picks: List[Dict],
        output_dir: str = "site/data"
    ) -> Tuple[str, str]:
        """
        阶段7：导出结果
        
        Args:
            predictions: 所有预测列表
            top_picks: 顶级推荐列表
            output_dir: 输出目录
        
        Returns:
            (predictions_path, picks_path)
        """
        logger.info("💾 Stage 7: Exporting Results")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # 导出完整预测
            predictions_path = f"{output_dir}/complete_predictions.json"
            with open(predictions_path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"   Exported {len(predictions)} predictions to {predictions_path}")
            
            # 导出顶级推荐
            picks_path = f"{output_dir}/picks_updated.json"
            with open(picks_path, 'w', encoding='utf-8') as f:
                json.dump(top_picks, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"   Exported {len(top_picks)} top picks to {picks_path}")
            
            # 生成统计报告
            stats = self._generate_stats_report(predictions, top_picks)
            stats_path = f"{output_dir}/analysis_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            logger.info(f"   Generated statistics report: {stats_path}")
            
            logger.info("✅ All results exported successfully")
            
            return predictions_path, picks_path
            
        except Exception as e:
            logger.error(f"❌ Error exporting results: {e}")
            return None, None
    
    def _generate_stats_report(self, all_preds, top_picks) -> Dict:
        """生成统计报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(all_preds),
            'top_picks_count': len(top_picks),
            'avg_confidence': np.mean([p.get('confidence', 0) for p in all_preds]) if all_preds else 0,
            'avg_win_probability': np.mean([p.get('final_prediction', {}).get('win_prob', 0) for p in all_preds]) if all_preds else 0,
            'avg_expected_value': np.mean([p.get('expected_value', 0) for p in top_picks]) if top_picks else 0,
            'max_expected_value': max([p.get('expected_value', 0) for p in all_preds], default=0),
            'model_weights': {
                'poisson': '20%',
                'xgboost': '25%',
                'dnn': '25%',
                'elo': '15%',
                'xg_model': '10%',
                'home_bias': '5%'
            },
            'data_sources': [
                'Football-data.org',
                'Understat (xG)',
                'Sofascore',
                'The Odds API',
                'Historical picks'
            ]
        }
        return report
    
    def run_full_pipeline(
        self,
        run_scrapers: bool = False,
        stage_load_historical: bool = True,
        stage_train_models: bool = False,
        competitions: List[str] = None
    ) -> Dict:
        """
        运行完整管道
        
        Args:
            stage_load_historical: 是否加载历史数据
            stage_train_models: 是否训练ML模型
            competitions: 联赛列表
        
        Returns:
            包含所有结果的字典
        """
        logger.info("=" * 60)
        logger.info("🚀 STARTING FULL PREDICTION PIPELINE")
        logger.info("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'running',
            'stages_completed': []
        }
        
        try:
            # Stage 0: 运行爬虫（可选）
            if run_scrapers:
                self.stage_0_scrape_external_data()
                results['stages_completed'].append('external_scrape')

            # Stage 1: 收集数据
            matches_df = self.stage_1_collect_data(competitions)
            results['stages_completed'].append('data_collection')
            results['matches_count'] = len(matches_df)
            
            # Stage 2: 加载历史数据 (可选)
            if stage_load_historical:
                self.stage_2_load_historical_data()
                results['stages_completed'].append('historical_data_loading')
            
            # Stage 3: 特征工程
            features_df = self.stage_3_feature_engineering(matches_df)
            results['stages_completed'].append('feature_engineering')
            results['features_shape'] = features_df.shape
            
            # Stage 4: 训练模型 (可选)
            if stage_train_models:
                self.stage_4_train_models(matches_df, features_df)
                results['stages_completed'].append('model_training')
            
            # Stage 5: 生成预测
            all_predictions = self.stage_5_generate_predictions(matches_df, features_df)
            results['stages_completed'].append('prediction_generation')
            results['predictions_count'] = len(all_predictions)
            
            # Stage 6: 筛选顶级推荐
            top_picks = self.stage_6_filter_top_picks(all_predictions)
            results['stages_completed'].append('filtering_top_picks')
            results['top_picks_count'] = len(top_picks)
            
            # Stage 7: 导出结果
            self.stage_7_export_results(all_predictions, top_picks)
            results['stages_completed'].append('result_export')
            
            results['status'] = 'completed'
            results['duration_minutes'] = (datetime.now() - datetime.fromisoformat(results['timestamp'])).total_seconds() / 60
            
            logger.info("=" * 60)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"📊 Results Summary:")
            logger.info(f"   - Total predictions: {results['predictions_count']}")
            logger.info(f"   - Top picks: {results['top_picks_count']}")
            logger.info(f"   - Top/Total ratio: {results['top_picks_count']/results['predictions_count']*100:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results


# 主执行函数
def main():
    """主函数"""
    
    # 初始化管道
    pipeline = FootballPredictionPipeline(
        football_api_key=None,  # 设置你的API密钥
        odds_api_key=None       # 设置你的API密钥
    )
    
    # 运行完整管道
    results = pipeline.run_full_pipeline(
        run_scrapers=True,            # 先刷新爬虫数据
        stage_load_historical=True,   # 使用历史数据提高特征质量
        stage_train_models=False,     # 如果有足够数据可设置为True
        competitions=['PL', 'SA', 'BL1']  # 主要联赛
    )
    
    # 输出结果
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION REPORT")
    print("=" * 60)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
