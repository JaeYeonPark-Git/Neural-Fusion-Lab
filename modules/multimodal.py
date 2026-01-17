import base64
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import os

class VisionAnalyst:
    """
    이미지 데이터를 LLM이 이해할 수 있는 포맷으로 변환하고 분석을 요청하는 클래스
    """
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, max_tokens=1024)

    def _encode_image(self, image_path: str) -> str:
        """로컬 이미지를 Base64 문자열로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_chart(self, image_path: str, context: str = "") -> str:
        """
        차트 이미지를 분석하여 텍스트 리포트 반환
        """
        base64_image = self._encode_image(image_path)
        
        prompt_text = f"""
        You are a professional Technical Analyst. 
        Analyze the attached stock chart image.
        
        Additional Context: {context}
        
        Please provide the analysis in the following format:
        1. **Trend Identification**: Is it Uptrend, Downtrend, or Sideways?
        2. **Key Levels**: Identify Support and Resistance levels.
        3. **Chart Patterns**: Are there any patterns (e.g., Head & Shoulders, Double Bottom)?
        4. **Volume Analysis**: Any significant volume spikes?
        5. **Conclusion**: Bullish / Bearish / Neutral
        """

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ]
        )

        response = self.llm.invoke([message])
        return response.content
