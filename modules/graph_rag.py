from neo4j import GraphDatabase
import os

class GraphRAGEngine:
    """
    Neo4j 지식 그래프와 상호작용하여 기업 관계 정보를 추출하는 엔진
    """
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def query_supply_chain(self, symbol: str):
        """
        특정 기업의 공급망(Supply Chain) 및 경쟁사 관계 조회
        """
        query = """
        MATCH (c:Company {symbol: $symbol})-[r:SUPPLIES_TO|COMPETES_WITH]-(other)
        RETURN c.name, type(r) as relation, other.name as related_company
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, symbol=symbol)
            return [record.data() for record in result]
            
    def get_entity_context(self, symbol: str) -> str:
        """
        LLM에게 전달할 텍스트 형태의 컨텍스트 생성
        """
        data = self.query_supply_chain(symbol)
        if not data:
            return "No graph data found."
        
        context_str = f"Knowledge Graph Context for {symbol}:\n"
        for item in data:
            context_str += f"- {item['related_company']} is related via {item['relation']}\n"
        return context_str
