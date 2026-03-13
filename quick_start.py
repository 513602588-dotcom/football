#!/usr/bin/env python
"""
快速开始脚本 - Football Prediction System
一键启动完整的预测管道
"""

import os
import sys
from pathlib import Path
import json
import logging
from datetime import datetime

# 添加项目目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """打印欢迎横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║  ⚽  Football Prophet Pro - Advanced AI Prediction System  ⚽     ║
    ║                                                                  ║
    ║  Powered by: XGBoost | DNN | Poisson | Elo | xG Analysis       ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """打印菜单"""
    menu = """
    📋 请选择操作:
    
    1. 🚀 运行完整预测管道
    2. 📊 加载历史数据并预测
    3. 🧠 训练机器学习模型
    4. 📈 回测策略性能
    5. 🔍 查看现有预测结果
    6. ⚙️  系统诊断和配置检查
    7. 📚 查看文档
    8. ❌ 退出
    
    请输入选项 (1-8): """
    return input(menu)

def run_full_pipeline():
    """运行完整预测管道"""
    logger.info("=" * 60)
    logger.info("启动完整预测管道...")
    logger.info("=" * 60)
    
    try:
        from src.build_pipeline import FootballPredictionPipeline
        
        # 获取API密钥
        football_key = os.getenv('FOOTBALL_API_KEY')
        odds_key = os.getenv('ODDS_API_KEY')
        
        if not football_key:
            logger.warning("⚠️  Football API密钥未配置，某些功能可能受限")
        
        # 初始化管道
        pipeline = FootballPredictionPipeline(
            football_api_key=football_key,
            odds_api_key=odds_key
        )
        
        # 获取用户选项
        print("\n获取联赛数据...")
        print("建议值: PL (英超), SA (西甲), BL1 (德甲), FR1 (法甲), IT1 (意甲)")
        user_comps = input("输入联赛代码 (逗号分隔) [默认: PL,SA]: ").strip()
        competitions = user_comps.split(',') if user_comps else ['PL', 'SA']
        
        # 运行管道
        results = pipeline.run_full_pipeline(
            run_scrapers=False,
            stage_load_historical=True,
            stage_train_models=False,
            competitions=competitions
        )
        
        # 显示结果
        print("\n" + "=" * 60)
        print("✅ 预测完成！")
        print("=" * 60)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print("\n📂 结果保存位置:")
        print("   - 完整预测: site/data/complete_predictions.json")
        print("   - 顶级推荐: site/data/picks_updated.json")
        print("   - 统计分析: site/data/analysis_stats.json")
        
    except Exception as e:
        logger.error(f"❌ 管道执行失败: {e}")
        import traceback
        traceback.print_exc()

def load_and_predict_with_history():
    """加载历史数据并预测"""
    logger.info("=" * 60)
    logger.info("使用历史数据的预测模式...")
    logger.info("=" * 60)
    
    try:
        from src.data.data_collector_enhanced import HistoricalDataLoader
        from src.data.feature_engineering import FeatureEngineer
        from src.engine.fusion_engine import SuperFusionModel
        
        # 加载历史数据
        print("📚 加载历史数据...")
        df = HistoricalDataLoader.create_dataframe_from_site_data()
        
        if len(df) == 0:
            print("❌ 历史数据为空，请先运行完整管道")
            return
        
        print(f"✅ 加载了 {len(df)} 条历史记录")
        
        # 特征工程
        print("🔧 提取特征...")
        engineer = FeatureEngineer()
        features_list = []
        
        for _, match in df.head(10).iterrows():
            features = engineer.build_match_features(match.to_dict(), df)
            if len(features) > 0:
                features_list.append(features)
        
        print(f"✅ 特征提取完成")
        
        # 预测
        print("🔮 生成预测...")
        fusion = SuperFusionModel()
        
        for idx, (_, match) in enumerate(df.head(10).iterrows()):
            if idx < len(features_list):
                pred = fusion.predict_single_match(match.to_dict(), features_list[idx])
                print(f"\n{idx+1}. {match.get('home_team')} vs {match.get('away_team')}")
                print(f"   推荐: {pred.get('recommended_bet')}")
                print(f"   概率: W{pred.get('final_prediction', {}).get('win_prob')}% "
                      f"D{pred.get('final_prediction', {}).get('draw_prob')}% "
                      f"L{pred.get('final_prediction', {}).get('loss_prob')}%")
                print(f"   EV: {pred.get('expected_value')}%")
                print(f"   置信度: {pred.get('confidence')}%")
        
    except Exception as e:
        logger.error(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()

def train_ml_models():
    """训练机器学习模型"""
    logger.info("=" * 60)
    logger.info("训练机器学习模型...")
    logger.info("=" * 60)
    
    try:
        from src.models.advanced_ml import XGBoostEnsemble, DeepNeuralNetwork
        from src.data.data_collector_enhanced import HistoricalDataLoader
        import pandas as pd
        import numpy as np
        
        # 加载数据
        print("📚 加载数据...")
        df = HistoricalDataLoader.create_dataframe_from_site_data()
        
        if len(df) < 100:
            print("❌ 数据不足，需要至少100条记录")
            return
        
        # 创建特征和标签
        X = df[['odds_win', 'odds_draw', 'odds_lose']].fillna(0)
        y = pd.Series(['win', 'draw', 'loss'] * (len(df) // 3 + 1))[:len(df)]
        
        # 训练XGBoost
        print("🧠 训练XGBoost模型...")
        xgb = XGBoostEnsemble()
        xgb.train(X, y)
        xgb.save('models/xgboost_ensemble.pkl')
        print("✅ XGBoost模型保存")
        
        # 训练DNN
        print("🧠 训练深度神经网络...")
        dnn = DeepNeuralNetwork()
        dnn.build(X.shape[1])
        dnn.train(X, y, epochs=20)
        dnn.save('models/dnn_model.h5')
        print("✅ DNN模型保存")
        
    except Exception as e:
        logger.error(f"❌ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()

def run_backtest():
    """运行回测"""
    logger.info("=" * 60)
    logger.info("运行投注策略回测...")
    logger.info("=" * 60)
    
    try:
        from src.backtest.performance_analysis import Backtester, ModelEvaluator
        from src.data.data_collector_enhanced import HistoricalDataLoader
        import numpy as np
        
        # 加载数据
        print("📚 加载预测数据...")
        predictions_path = "site/data/picks_updated.json"
        
        if not Path(predictions_path).exists():
            print(f"❌ 文件不存在: {predictions_path}")
            print("请先运行完整预测管道")
            return
        
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 生成模拟结果
        print("📊 生成模拟比赛结果...")
        np.random.seed(42)
        results = [
            {'result': np.random.choice(['win', 'draw', 'loss'], p=[0.46, 0.27, 0.27])}
            for _ in predictions
        ]
        
        # Kelly回测
        print("\n🎲 运行Kelly准则回测...")
        backtester = Backtester(initial_bankroll=1000)
        kelly_stats = backtester.backtest_kelly(predictions, results)
        
        print(f"\n📈 Kelly回测结果:")
        print(f"   初始资金: ${kelly_stats.get('initial_bankroll', 1000):.2f}")
        print(f"   最终资金: ${backtester.bankroll:.2f}")
        print(f"   总回报率: {kelly_stats.get('total_return', 0):.2f}%")
        print(f"   胜率: {kelly_stats.get('win_rate', 0)*100:.1f}%")
        print(f"   每笔平均收益: ${kelly_stats.get('avg_profit_per_trade', 0):.2f}")
        
        # 固定赌注回测
        print("\n🎲 运行固定赌注回测...")
        backtester2 = Backtester(initial_bankroll=1000)
        fixed_stats = backtester2.backtest_fixed_stake(predictions, results, stake=10)
        
        print(f"\n📈 固定赌注回测结果:")
        print(f"   初始资金: ${fixed_stats.get('initial_bankroll', 1000):.2f}")
        print(f"   最终资金: ${backtester2.bankroll:.2f}")
        print(f"   总回报率: {fixed_stats.get('total_return', 0):.2f}%")
        print(f"   胜率: {fixed_stats.get('win_rate', 0)*100:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()

def view_results():
    """查看现有预测结果"""
    print("\n📂 检查现有结果文件...\n")
    
    result_dir = Path("site/data")
    result_files = ['picks.json', 'picks_updated.json', 'complete_predictions.json', 'analysis_stats.json']
    
    for file in result_files:
        path = result_dir / file
        if path.exists():
            size = path.stat().st_size
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            print(f"✅ {file}")
            print(f"   大小: {size/1024:.1f} KB")
            print(f"   修改时间: {mtime}")
            
            if file.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"   记录数: {len(data)}")
                    print()
        else:
            print(f"❌ {file} - 不存在\n")

def system_diagnostic():
    """系统诊断"""
    print("\n" + "=" * 60)
    print("🔍 系统诊断...")
    print("=" * 60 + "\n")
    
    checks = {
        'Python版本': sys.version,
        '项目目录': str(project_root),
        '数据库': 'data/football.db' if Path('data/football.db').exists() else '不存在',
        'API密钥 (Football)': '已配置' if os.getenv('FOOTBALL_API_KEY') else '未配置',
        'API密钥 (Odds)': '已配置' if os.getenv('ODDS_API_KEY') else '未配置',
    }
    
    for check, status in checks.items():
        print(f"  {check}: {status}")
    
    # 检查依赖
    print("\n📦 依赖包检查:")
    packages = ['pandas', 'numpy', 'sklearn', 'xgboost', 'tensorflow', 'requests']
    
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"   ✅ {pkg}")
        except ImportError:
            print(f"   ❌ {pkg} (缺失)")
    
    print("\n")

def show_documentation():
    """显示文档"""
    doc = """
    📚 快速参考
    
    1️⃣  运行完整预测管道
       - 从多个API收集数据
       - 提取高级特征
       - 生成融合预测
       - 导出结果
    
    2️⃣  模型架构
       - Poisson回归 (20% 权重)
       - XGBoost集合 (25% 权重)
       - 深度神经网络 (25% 权重)
       - Elo评分 (15% 权重)
       - xG统计模型 (10% 权重)
       - 主场偏差 (5% 权重)
    
    3️⃣  关键指标
       - Expected Value (EV): 投注价值
       - Kelly Stake: 最优投注比例
       - Confidence: 预测置信度
       - ROI: 回测年化收益率
    
    4️⃣  API配置
       - Football-data.org: https://www.football-data.org
       - The Odds API: https://the-odds-api.com
       - 在 .env 文件中配置密钥
    
    5️⃣  输出文件
       - site/data/picks_updated.json: 顶级推荐
       - site/data/complete_predictions.json: 全部预测
       - site/data/analysis_stats.json: 统计分析
    
    6️⃣  查看结果
       - 打开 site/index_pro.html 在浏览器查看
       - 或运行 python -m http.server 在localhost:8000查看
    
    📖 完整文档: 请查阅 README_UPGRADE.md
    
    ⚠️  免责声明:
       本系统仅供学习和研究使用，不构成投资建议。
       投注有风险，请理性决策。
    """
    print(doc)

def main():
    """主函数"""
    print_banner()
    
    # 创建必要目录
    Path('logs').mkdir(exist_ok=True)
    Path('data/cache').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # 加载环境配置
    try:
        from dotenv import load_dotenv
        load_dotenv('.env')
    except ImportError:
        logger.warning("python-dotenv未安装，将使用默认配置")
    
    while True:
        choice = print_menu()
        
        if choice == '1':
            run_full_pipeline()
        elif choice == '2':
            load_and_predict_with_history()
        elif choice == '3':
            train_ml_models()
        elif choice == '4':
            run_backtest()
        elif choice == '5':
            view_results()
        elif choice == '6':
            system_diagnostic()
        elif choice == '7':
            show_documentation()
        elif choice == '8':
            print("\n👋 再见! 感谢使用 Football Prophet Pro")
            break
        else:
            print("❌ 无效选项，请重试")
        
        input("\n按Enter继续...")

if __name__ == "__main__":
    main()
