import matplotlib.pyplot as plt
import os
from .tools import MarketDataManager
from .multimodal import VisionAnalyst
from langchain_core.messages import HumanMessage

class ChartAgent:
    def __init__(self, config):
        self.data_manager = MarketDataManager()
        self.vision_analyst = VisionAnalyst(model_name=config['models']['vision'])
        self.chart_dir = config['paths']['chart_save_dir']
        os.makedirs(self.chart_dir, exist_ok=True)

    def _generate_and_save_chart(self, symbol: str) -> str:
        """데이터를 받아 차트를 그리고 이미지 파일로 저장"""
        df = self.data_manager.get_price_history(symbol)
        df = self.data_manager.add_technical_indicators(df)
        
        # Matplotlib로 차트 그리기 (스타일링 적용)
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Close'], label='Close Price')
        plt.plot(df.index, df['SMA_20'], label='SMA 20', linestyle='--')
        plt.title(f"{symbol} Price Chart Analysis")
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.chart_dir, f"{symbol}_chart.png")
        plt.savefig(save_path)
        plt.close() # 메모리 해제
        return save_path

    def analyze(self, state):
        """LangGraph 노드에서 호출될 메인 함수"""
        symbol = state['stock_symbol']
        print(f"📈 [ChartAgent] Generating and analyzing chart for {symbol}...")
        
        # 1. 차트 생성 및 저장
        image_path = self._generate_and_save_chart(symbol)
        
        # 2. VLM을 통한 이미지 분석
        analysis_result = self.vision_analyst.analyze_chart(image_path)
        
        print(f"✅ [ChartAgent] Analysis Complete.")
        
        # 3. 결과 반환 (State 업데이트)
        return {
            "chart_analysis": analysis_result,
            "messages": [HumanMessage(content=f"Chart Analysis for {symbol}:\n{analysis_result}")]
        }
