"""
Crypto Stock AI Prediction Agent v2.0
Main entry point - interactive menu.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

from data_collector import DataCollector
from feature_engineer import FeatureEngineer
from models import EnsemblePredictor
from backtester import Backtester
from news_scraper import NewsScraper, SentimentAnalyzer
from ai_brain import AIBrain

import warnings
warnings.filterwarnings('ignore')


def print_banner():
    os.system('clear' if os.name != 'nt' else 'cls')
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🤖  CRYPTO STOCK AI AGENT v2.0                          ║
║                                                           ║
║   📊  ML Ensemble + FinBERT Sentiment                     ║
║   🧠  HuggingFace AI Deep Reasoning                       ║
║   🗣️   Bull vs Bear Expert Debate                         ║
║   ✅  Self-Verification Loop                               ║
║                                                           ║
║   100% FREE | No paid APIs needed                         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)


def full_analysis(ticker='COIN', horizon=1):
    """COMPLETE analysis pipeline with AI reasoning."""

    print_banner()
    start_time = datetime.now()

    # PHASE 1: DATA COLLECTION
    print("━" * 60)
    print("  📥 PHASE 1: DATA COLLECTION")
    print("━" * 60)

    collector = DataCollector()
    raw_df = collector.collect_all(ticker)

    # PHASE 2: FEATURE ENGINEERING
    print("\n" + "━" * 60)
    print("  🔧 PHASE 2: FEATURE ENGINEERING")
    print("━" * 60)

    engineer = FeatureEngineer()
    df = engineer.prepare_full_dataset(raw_df, horizon=horizon)

    # PHASE 3: ML MODEL TRAINING
    print("\n" + "━" * 60)
    print("  🤖 PHASE 3: ML MODEL TRAINING")
    print("━" * 60)

    predictor = EnsemblePredictor()
    X_train, X_test, y_train, y_test = predictor.prepare_data(df, test_size=0.2)
    X_train_sel, X_test_sel = predictor.select_features(
        X_train, y_train, X_test
    )

    model_results = predictor.train(X_train_sel, y_train, X_test_sel, y_test)

    # Cross-validate
    X_all = np.vstack([X_train_sel, X_test_sel])
    y_all = np.concatenate([y_train, y_test])
    predictor.cross_validate(X_all, y_all)

    # PHASE 4: BACKTESTING
    print("\n" + "━" * 60)
    print("  📈 PHASE 4: BACKTESTING")
    print("━" * 60)

    backtester = Backtester(initial_capital=10000)

    ensemble_probs = np.zeros(len(X_test_sel))
    for name, model in predictor.models.items():
        weight = predictor.model_weights[name]
        ensemble_probs += weight * model.predict_proba(X_test_sel)[:, 1]
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    test_df = df.iloc[-len(X_test_sel):]
    bt_results, trades, portfolio = backtester.run(
        test_df, ensemble_preds,
        confidence=ensemble_probs, conf_threshold=0.52
    )

    print(f"\n  📊 Backtest Results:")
    print(f"  {'─' * 45}")
    for key, val in bt_results.items():
        label = key.replace('_', ' ').title()
        if 'pct' in key:
            color = '\033[92m' if val > 0 else '\033[91m'
            print(f"    {label:30s}: {color}{val}%\033[0m")
        else:
            print(f"    {label:30s}: {val}")

    # PHASE 5: NEWS & SENTIMENT
    print("\n" + "━" * 60)
    print("  📰 PHASE 5: NEWS & SENTIMENT ANALYSIS")
    print("━" * 60)

    scraper = NewsScraper()
    news_data = scraper.collect_all_news(ticker)

    sentiment_analyzer = SentimentAnalyzer()
    sentiment = sentiment_analyzer.analyze_headlines(
        news_data.get('all_headlines', [])
    )

    # PHASE 6: ML PREDICTION
    print("\n" + "━" * 60)
    print("  🎯 PHASE 6: TODAY'S ML PREDICTION")
    print("━" * 60)

    latest_features = X_test_sel[-1]
    prediction = predictor.predict(latest_features)

    # Get latest technical values
    latest = df.iloc[-1]
    fg = news_data.get('fear_greed', {})

    print(f"""
    ┌─────────────────────────────────────────────┐
    │  Ticker:      {ticker:29s} │
    │  Date:        {datetime.now().strftime('%Y-%m-%d %H:%M'):29s} │
    │  Horizon:     {f'{horizon}-day forecast':29s} │
    │                                             │
    │  ML Signal:   {prediction['signal']:29s} │
    │  Direction:   {prediction['direction']:29s} │
    │  Confidence:  {f"{prediction['confidence']*100:.1f}%":29s} │
    │  Agreement:   {f"{prediction['agreement']*100:.0f}% models agree":29s} │
    │                                             │
    │  Sentiment:   {sentiment.get('overall', 'N/A'):29s} │
    │  Fear/Greed:  {str(fg.get('value', 'N/A')) + ' (' + str(fg.get('label', '')) + ')':29s} │
    └─────────────────────────────────────────────┘
    """)

    print("  Individual Models:")
    for name, data in prediction['models'].items():
        direction = "UP ↑" if data['prediction'] == 1 else "DOWN ↓"
        conf = f"{data['confidence']*100:.1f}%"
        bar_len = int(data['confidence'] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {name.upper():5s}: {direction:8s} [{bar}] {conf}")

    # PHASE 7: AI DEEP REASONING
    print("\n" + "━" * 60)
    print("  🧠 PHASE 7: AI DEEP REASONING (Chain-of-Thought)")
    print("━" * 60)

    brain = AIBrain()

    if brain.available:
        # Prepare data package for AI
        data_package = {
            'ticker': ticker,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'close_price': f"${latest['close']:.2f}",
            'rsi': f"{latest.get('rsi_14', 0):.1f}",
            'macd': f"{latest.get('MACD_12_26_9', latest.get('macd', 0)):.4f}",
            'price_vs_sma50': (
                f"{'ABOVE' if latest['close'] > latest.get('sma_50', 0) else 'BELOW'} "
                f"(${latest.get('sma_50', 0):.2f})"
            ) if 'sma_50' in df.columns else 'N/A',
            'price_vs_sma200': 'N/A',
            'golden_cross': str(bool(latest.get('golden_cross', 0))),
            'volume_ratio': f"{latest.get('volume_ratio', 0):.2f}x",
            'atr': f"{latest.get('atr_14', 0):.2f}",
            'volatility': f"{latest.get('volatility_20', 0)*100:.2f}%",
            'z_score': f"{latest.get('z_score_20', 0):.2f}",
            'btc_rsi': (
                f"{latest.get('btc_rsi', 0):.1f}"
                if 'btc_rsi' in df.columns else 'N/A'
            ),
            'btc_return_30d': (
                f"{latest.get('btc_return_30d', 0)*100:.1f}%"
                if 'btc_return_30d' in df.columns else 'N/A'
            ),
            'btc_trend': (
                'BULLISH' if latest.get('btc_return_30d', 0) > 0 else 'BEARISH'
            ) if 'btc_return_30d' in df.columns else 'N/A',
            'fear_greed': f"{fg.get('value', 'N/A')} ({fg.get('label', '')})",
            'news_sentiment': sentiment.get('overall', 'N/A'),
            'sentiment_score': str(sentiment.get('avg_score', 0)),
            'headlines': '\n'.join(
                news_data.get('all_headlines', [])[:10]
            ),
            'reddit_sentiment': 'See headlines',
            'ml_signal': prediction['signal'],
            'ml_confidence': f"{prediction['confidence']*100:.1f}%",
            'model_accuracy': (
                f"{model_results.get('ensemble', {}).get('accuracy', 0)*100:.1f}%"
            ),
            'model_agreement': f"{prediction['agreement']*100:.0f}%",
            'max_drawdown': f"{bt_results.get('max_drawdown_pct', 0)}%",
            'sharpe': str(bt_results.get('sharpe_ratio', 0)),
            'win_rate': f"{bt_results.get('win_rate_pct', 0)}%",
        }

        # Chain-of-Thought Analysis
        cot_result = brain.think_step_by_step(data_package)

        print(f"\n  {'─' * 55}")
        print("  📋 TECHNICAL REASONING:")
        print(f"  {'─' * 55}")
        print(f"  {cot_result['technical_reasoning']}")

        print(f"\n  {'─' * 55}")
        print("  📋 SENTIMENT REASONING:")
        print(f"  {'─' * 55}")
        print(f"  {cot_result['sentiment_reasoning']}")

        print(f"\n  {'─' * 55}")
        print("  📋 RISK ASSESSMENT:")
        print(f"  {'─' * 55}")
        print(f"  {cot_result['risk_assessment']}")

        print(f"\n  {'─' * 55}")
        print("  📋 SYNTHESIS:")
        print(f"  {'─' * 55}")
        print(f"  {cot_result['synthesis']}")

        print(f"\n  {'─' * 55}")
        print("  ✅ SELF-VERIFICATION:")
        print(f"  {'─' * 55}")
        print(f"  {cot_result['final_analysis']}")

        # Expert Debate
        print("\n" + "━" * 60)
        print("  🗣️  PHASE 8: EXPERT DEBATE (Bull vs Bear)")
        print("━" * 60)

        debate = brain.expert_debate(data_package)

        print(f"\n  🐂 BULL CASE:")
        print(f"  {'─' * 45}")
        print(f"  {debate['bull_case']}")

        print(f"\n  🐻 BEAR CASE:")
        print(f"  {'─' * 45}")
        print(f"  {debate['bear_case']}")

        print(f"\n  ⚖️  MODERATOR VERDICT:")
        print(f"  {'─' * 45}")
        print(f"  {debate['moderator_verdict']}")

    else:
        print("  ⚠️  AI Brain unavailable (no HF_TOKEN)")
        print("  Add HF_TOKEN to .env for AI reasoning")
        print("  Get free token at: https://huggingface.co/settings/tokens")

    # FINAL SUMMARY
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "═" * 60)
    print("  📋 FINAL SUMMARY")
    print("═" * 60)

    # Combine all signals
    signals = {
        'ML Ensemble': prediction['signal'],
        'News Sentiment': sentiment.get('overall', 'NEUTRAL'),
        'Fear & Greed': (
            'BULLISH' if fg.get('value', 50) > 60
            else 'BEARISH' if fg.get('value', 50) < 40
            else 'NEUTRAL'
        ),
    }

    if 'btc_return_30d' in df.columns:
        signals['BTC Trend'] = (
            'BULLISH' if latest.get('btc_return_30d', 0) > 0 else 'BEARISH'
        )
    if 'rsi_14' in df.columns:
        rsi_val = latest.get('rsi_14', 50)
        signals['RSI Signal'] = (
            'OVERBOUGHT' if rsi_val > 70
            else 'OVERSOLD' if rsi_val < 30
            else 'NEUTRAL'
        )

    print(f"\n  {'Signal Source':<25} {'Reading'}")
    print(f"  {'─' * 45}")
    for source, signal in signals.items():
        emoji = '🟢' if 'BULL' in signal or 'BUY' in signal else (
            '🔴' if 'BEAR' in signal or 'SELL' in signal else '🟡'
        )
        print(f"  {emoji} {source:<23} {signal}")

    # Count bullish vs bearish
    bull_count = sum(
        1 for s in signals.values()
        if any(w in s for w in ['BULL', 'BUY', 'OVERSOLD'])
    )
    bear_count = sum(
        1 for s in signals.values()
        if any(w in s for w in ['BEAR', 'SELL', 'OVERBOUGHT'])
    )
    total_signals = len(signals)

    print(f"\n  Bullish signals: {bull_count}/{total_signals}")
    print(f"  Bearish signals: {bear_count}/{total_signals}")

    if bull_count > bear_count + 1:
        composite = "BULLISH BIAS 🟢"
    elif bear_count > bull_count + 1:
        composite = "BEARISH BIAS 🔴"
    else:
        composite = "MIXED / NEUTRAL 🟡"

    print(f"\n  ╔═══════════════════════════════════════╗")
    print(f"  ║  COMPOSITE SIGNAL: {composite:20s}║")
    print(f"  ╚═══════════════════════════════════════╝")

    # Save models
    predictor.save(f'saved_models/{ticker}')

    print(f"\n  ⏱️  Total analysis time: {elapsed:.1f} seconds")
    print(f"  📁  Models saved to: saved_models/{ticker}/")
    print(f"\n  ⚠️  DISCLAIMER: This is NOT financial advice.")
    print(f"  ⚠️  Past performance does not guarantee future results.")
    print(f"  ⚠️  Always do your own research.")
    print()

    return model_results, bt_results, prediction


def scan_all():
    """Scan all crypto stocks and rank."""
    print_banner()
    print("  🔍 SCANNING ALL CRYPTO STOCKS...")
    print("━" * 60)

    tickers = ['COIN', 'MARA', 'RIOT', 'CLSK', 'MSTR']
    results = []

    for ticker in tickers:
        try:
            print(f"\n{'━' * 60}")
            print(f"  Analyzing {ticker}...")

            collector = DataCollector()
            raw_df = collector.collect_all(ticker)

            engineer = FeatureEngineer()
            df = engineer.prepare_full_dataset(raw_df)

            predictor = EnsemblePredictor()
            X_train, X_test, y_train, y_test = predictor.prepare_data(df)
            X_train_s, X_test_s = predictor.select_features(
                X_train, y_train, X_test
            )
            model_results = predictor.train(
                X_train_s, y_train, X_test_s, y_test
            )

            prediction = predictor.predict(X_test_s[-1])

            results.append({
                'ticker': ticker,
                'signal': prediction['signal'],
                'confidence': prediction['confidence'],
                'direction': prediction['direction'],
                'agreement': prediction['agreement'],
                'accuracy': model_results.get(
                    'ensemble', {}
                ).get('accuracy', 0)
            })

            predictor.save(f'saved_models/{ticker}')

        except Exception as e:
            print(f"  ❌ {ticker} failed: {e}")

    # Print rankings
    results.sort(key=lambda x: x['confidence'], reverse=True)

    print("\n" + "═" * 65)
    print("  📋 SCAN RESULTS - RANKED BY CONFIDENCE")
    print("═" * 65)
    print(f"  {'Ticker':<8} {'Signal':<18} {'Conf':>6} {'Agree':>7} {'Acc':>6}")
    print(f"  {'─' * 55}")

    for r in results:
        emoji = '🟢' if 'BUY' in r['signal'] else (
            '🔴' if 'SELL' in r['signal'] else '🟡'
        )
        print(
            f"  {emoji} {r['ticker']:<6} {r['signal']:<18} "
            f"{r['confidence']*100:5.1f}% {r['agreement']*100:5.0f}% "
            f"{r['accuracy']*100:5.1f}%"
        )

    print()


def show_menu():
    """Interactive menu."""
    print_banner()
    print("  What would you like to do?\n")
    print("    1  │  Full Analysis (single stock + AI reasoning)")
    print("    2  │  Scan All Crypto Stocks (quick ranking)")
    print("    3  │  Quick Re-predict (uses saved models)")
    print("    4  │  Exit")
    print()

    choice = input("  Enter choice (1/2/3/4): ").strip()
    return choice


def main():
    while True:
        choice = show_menu()

        if choice == '1':
            print("\n  Available tickers: COIN, MARA, RIOT, CLSK, MSTR, HUT, BITF")
            ticker = input("  Enter ticker [COIN]: ").strip().upper()
            if not ticker:
                ticker = 'COIN'

            try:
                full_analysis(ticker)
            except Exception as e:
                print(f"\n  ❌ Error: {e}")
                import traceback
                traceback.print_exc()

            input("\n  Press Enter to continue...")

        elif choice == '2':
            try:
                scan_all()
            except Exception as e:
                print(f"\n  ❌ Error: {e}")

            input("\n  Press Enter to continue...")

        elif choice == '3':
            ticker = input("  Enter ticker [COIN]: ").strip().upper()
            if not ticker:
                ticker = 'COIN'

            model_path = f'saved_models/{ticker}'
            if os.path.exists(model_path):
                print(f"\n  ⚡ Loading saved models for {ticker}...")
                predictor = EnsemblePredictor()
                predictor.load(model_path)

                collector = DataCollector()
                raw_df = collector.collect_all(ticker)
                engineer = FeatureEngineer()
                df = engineer.prepare_full_dataset(raw_df)

                feat_cols = [
                    c for c in predictor.feature_columns if c in df.columns
                ]
                latest = df[feat_cols].iloc[-1].values

                prediction = predictor.predict(latest)

                print(
                    f"\n    {ticker}: {prediction['signal']} "
                    f"(confidence: {prediction['confidence']*100:.1f}%)"
                )
                print(f"    Agreement: {prediction['agreement']*100:.0f}%")
            else:
                print(
                    f"\n  No saved models for {ticker}. "
                    f"Run full analysis first (option 1)."
                )

            input("\n  Press Enter to continue...")

        elif choice == '4' or choice.lower() == 'q':
            print("\n  👋 Goodbye!\n")
            sys.exit(0)

        else:
            print("\n  Invalid choice. Try again.")
            input("  Press Enter...")


if __name__ == '__main__':
    main()
