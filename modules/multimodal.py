import base64
import json
import os
from typing import List, Dict, Union, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

class VisionAnalyst:
    """
    [Advanced VLM Engine]
    멀티모달 모델(GPT-4o 등)을 활용하여 단일/다중 금융 차트를 분석하고
    구조화된 데이터(JSON)를 반환하는 분석 엔진.
    """
    def __init__(self, model_name="gpt-4o", temperature=0.0):
        # Temperature를 0으로 설정하여 분석의 일관성 유지
        self.llm = ChatOpenAI(model=model_name, max_tokens=2048, temperature=temperature)

    def _encode_image(self, image_path: str) -> str:
        """로컬 이미지를 Base64 문자열로 인코딩 (예외 처리 추가)"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_chart(self, image_paths: Union[str, List[str]], context: str = "", strategy: str = "General") -> Dict:
        """
        [Upgrade] 단일 또는 다중 차트 이미지를 받아 JSON 형태의 정형화된 리포트 반환
        
        Args:
            image_paths: 이미지 경로 문자열 또는 경로 리스트 (예: [일봉, 주봉])
            context: 추가 텍스트 정보 (예: "현재 금리 인상기임")
            strategy: 분석 관점 ("Momentum", "Reversal", "General")
        """
        # 1. 입력 정규화 (항상 리스트로 처리)
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # 2. 이미지 메시지 블록 생성
        content_blocks = []
        
        # 시스템 프롬프트 (JSON 강제화)
        system_prompt = f"""
        You are a Wall Street Senior Technical Analyst specializing in {strategy} strategies.
        Analyze the provided chart images. If multiple images are provided, treat them as Multi-Timeframe Analysis (e.g., Daily & Weekly).
        
        You MUST output the result in valid JSON format with the following keys:
        - "trend": "Uptrend" | "Downtrend" | "Sideways"
        - "support_resistance": List of key price levels.
        - "patterns": Detected chart patterns (e.g., Head & Shoulders, Bull Flag).
        - "signals": Key technical signals (e.g., Golden Cross, Divergence).
        - "risk_score": Integer (1-10, where 10 is High Risk).
        - "summary": A concise summary of the visual analysis.
        
        Do not include markdown formatting like ```json ... ```. Just return the raw JSON string.
        """
        content_blocks.append({"type": "text", "text": system_prompt})
        
        # 사용자 컨텍스트 추가
        if context:
            content_blocks.append({"type": "text", "text": f"Additional Context: {context}"})

        # 다중 이미지 로드 및 추가
        for idx, path in enumerate(image_paths):
            base64_img = self._encode_image(path)
            content_blocks.append({
                "type": "text", 
                "text": f"[Image {idx+1}] Chart View"
            })
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })

        # 3. LLM 호출
        message = HumanMessage(content=content_blocks)
        
        try:
            response = self.llm.invoke([message])
            # JSON 파싱 시도 (LLM이 가끔 마크다운을 섞을 때를 대비)
            raw_content = response.content.strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content.split("```")[1].strip()
                
            return json.loads(raw_content)
            
        except json.JSONDecodeError:
            # 파싱 실패 시 원본 텍스트를 포함한 에러 딕셔너리 반환
            return {
                "error": "Failed to parse JSON", 
                "raw_text": response.content,
                "trend": "Unknown",
                "risk_score": 5
            }
        except Exception as e:
            return {"error": str(e)}

# ==========================================
# 🧪 Test Code (이 파일을 직접 실행했을 때 동작)
# ==========================================
if __name__ == "__main__":
    # 더미 테스트 (실제 이미지가 있어야 동작함)
    print("Testing VisionAnalyst...")
    # analyst = VisionAnalyst()
    # result = analyst.analyze_chart(["./data/NVDA_daily.png", "./data/NVDA_weekly.png"], context="Earnings reported yesterday.")
    # print(result)
