"""GraphRAG retrieval pipeline using Neo4j knowledge graph."""

from neo4j import GraphDatabase

from core.config import Config


class GraphRAGPipeline:
    """Retrieves medical knowledge from the Neo4j knowledge graph."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD),
        )

    def close(self):
        self.driver.close()

    def retrieve_by_concept(self, concept: str, top_k: int = 3) -> list[dict]:
        """
        Find QAPairs tagged with the given concept.
        Returns a list of {question, answer} dicts.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (q:QAPair)-[:TAGGED_WITH]->(c:MedicalConcept)
                WHERE toLower(c.name) CONTAINS toLower($concept)
                RETURN q.question AS question, q.answer AS answer
                LIMIT $top_k
                """,
                concept=concept,
                top_k=top_k,
            )
            return [{"question": r["question"], "answer": r["answer"]} for r in result]

    def retrieve_related_concepts(self, concept: str, top_k: int = 5) -> list[dict]:
        """
        Find MedicalConcepts that co-occur with the given concept,
        ranked by co-occurrence weight.
        Returns a list of {concept, weight} dicts.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a:MedicalConcept)-[r:CO_OCCURS_WITH]-(b:MedicalConcept)
                WHERE toLower(a.name) CONTAINS toLower($concept)
                RETURN b.name AS concept, r.weight AS weight
                ORDER BY r.weight DESC
                LIMIT $top_k
                """,
                concept=concept,
                top_k=top_k,
            )
            return [{"concept": r["concept"], "weight": r["weight"]} for r in result]

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        Main entry point called by the LangChain tool.
        Runs both concept lookup and relationship query, formats into a
        single context string for the LLM.
        """
        # Extract the most likely medical concept from the query.
        # We search for each word (longest first) against concept names.
        concept = self._extract_concept(query)

        if not concept:
            return "No relevant medical concepts found in the knowledge graph."

        qa_results = self.retrieve_by_concept(concept, top_k=top_k)
        related = self.retrieve_related_concepts(concept, top_k=5)

        parts = [f"[Knowledge Graph results for concept: '{concept}']"]

        if qa_results:
            parts.append("\n--- Relevant Q&A pairs ---")
            for i, qa in enumerate(qa_results, 1):
                parts.append(f"[{i}] Q: {qa['question']}\n    A: {qa['answer']}")
        else:
            parts.append("No Q&A pairs found for this concept.")

        if related:
            related_names = ", ".join(r["concept"] for r in related)
            parts.append(f"\n--- Related medical concepts ---\n{related_names}")

        return "\n".join(parts)

    def _extract_concept(self, query: str) -> str:
        """
        Find the best matching concept in the graph for a given query.
        Tries multi-word matches first (longer = more specific), then single words.
        """
        # Tokenize query into candidate phrases (1-3 words)
        words = query.lower().split()
        candidates = []
        for length in range(3, 0, -1):  # try 3-word, 2-word, 1-word phrases
            for i in range(len(words) - length + 1):
                candidates.append(" ".join(words[i: i + length]))

        with self.driver.session() as session:
            for candidate in candidates:
                result = session.run(
                    """
                    MATCH (c:MedicalConcept)
                    WHERE toLower(c.name) CONTAINS $candidate
                    RETURN c.name AS name
                    LIMIT 1
                    """,
                    candidate=candidate,
                )
                record = result.single()
                if record:
                    return record["name"]

        return ""
