"""Build Neo4j knowledge graph from medical Q&A CSV data."""

import csv
import re
from collections import defaultdict
from neo4j import GraphDatabase

from config import Config


def parse_tags(raw: str) -> list[str]:
    """Parse numpy-style string array e.g. \"['rash' 'antibiotic']\" into a list."""
    tags = re.findall(r"'([^']+)'", raw)
    return [t.strip().lower() for t in tags if t.strip()]


def load_rows(csv_path: str) -> list[dict]:
    """Load only label=1.0 rows from the CSV."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row["label"].strip() == "1.0":
                tags = parse_tags(row["tags"])
                if not tags:
                    continue
                rows.append({
                    "id": i,
                    "question": row["short_question"].strip(),
                    "answer": row["short_answer"].strip(),
                    "tags": tags,
                })
    return rows


def build_graph(uri: str, username: str, password: str, csv_path: str):
    driver = GraphDatabase.driver(uri, auth=(username, password))

    rows = load_rows(csv_path)
    print(f"Loaded {len(rows)} valid rows (label=1.0 with tags)")

    # Pre-compute co-occurrence weights across the full dataset
    co_occurrence: dict[tuple, int] = defaultdict(int)
    for row in rows:
        tags = row["tags"]
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                pair = tuple(sorted([tags[i], tags[j]]))
                co_occurrence[pair] += 1

    print(f"Found {len(co_occurrence)} co-occurrence pairs across concepts")

    with driver.session() as session:
        # --- Clear existing graph ---
        print("Clearing existing graph...")
        session.run("MATCH (n) DETACH DELETE n")

        # --- Constraints for uniqueness + fast lookup ---
        session.run(
            "CREATE CONSTRAINT qa_id IF NOT EXISTS "
            "FOR (q:QAPair) REQUIRE q.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT concept_name IF NOT EXISTS "
            "FOR (c:MedicalConcept) REQUIRE c.name IS UNIQUE"
        )

        # --- Batch insert QAPair nodes ---
        print("Creating QAPair nodes...")
        batch_size = 200
        for start in range(0, len(rows), batch_size):
            batch = rows[start: start + batch_size]
            session.run(
                """
                UNWIND $rows AS row
                MERGE (q:QAPair {id: row.id})
                SET q.question = row.question,
                    q.answer   = row.answer
                """,
                rows=[{"id": r["id"], "question": r["question"], "answer": r["answer"]}
                      for r in batch],
            )
        print(f"  Created {len(rows)} QAPair nodes")

        # --- Collect all unique concept names ---
        all_concepts = {tag for row in rows for tag in row["tags"]}
        concept_list = [{"name": name} for name in all_concepts]

        # --- Batch insert MedicalConcept nodes ---
        print("Creating MedicalConcept nodes...")
        for start in range(0, len(concept_list), batch_size):
            batch = concept_list[start: start + batch_size]
            session.run(
                "UNWIND $concepts AS c MERGE (:MedicalConcept {name: c.name})",
                concepts=batch,
            )
        print(f"  Created {len(concept_list)} MedicalConcept nodes")

        # --- Create TAGGED_WITH relationships ---
        print("Creating TAGGED_WITH relationships...")
        tag_rels = [
            {"qa_id": row["id"], "concept": tag}
            for row in rows
            for tag in row["tags"]
        ]
        for start in range(0, len(tag_rels), batch_size):
            batch = tag_rels[start: start + batch_size]
            session.run(
                """
                UNWIND $rels AS rel
                MATCH (q:QAPair {id: rel.qa_id})
                MATCH (c:MedicalConcept {name: rel.concept})
                MERGE (q)-[:TAGGED_WITH]->(c)
                """,
                rels=batch,
            )
        print(f"  Created {len(tag_rels)} TAGGED_WITH relationships")

        # --- Create CO_OCCURS_WITH relationships ---
        print("Creating CO_OCCURS_WITH relationships...")
        co_list = [
            {"a": pair[0], "b": pair[1], "weight": weight}
            for pair, weight in co_occurrence.items()
        ]
        for start in range(0, len(co_list), batch_size):
            batch = co_list[start: start + batch_size]
            session.run(
                """
                UNWIND $rels AS rel
                MATCH (a:MedicalConcept {name: rel.a})
                MATCH (b:MedicalConcept {name: rel.b})
                MERGE (a)-[r:CO_OCCURS_WITH]-(b)
                SET r.weight = rel.weight
                """,
                rels=batch,
            )
        print(f"  Created {len(co_list)} CO_OCCURS_WITH relationships")

    driver.close()
    print("\nKnowledge graph build complete.")
    print(f"  Nodes : {len(rows)} QAPairs + {len(all_concepts)} MedicalConcepts")
    print(f"  Edges : {len(tag_rels)} TAGGED_WITH + {len(co_list)} CO_OCCURS_WITH")


if __name__ == "__main__":
    build_graph(
        uri=Config.NEO4J_URI,
        username=Config.NEO4J_USERNAME,
        password=Config.NEO4J_PASSWORD,
        csv_path=Config.CSV_PATH,
    )
