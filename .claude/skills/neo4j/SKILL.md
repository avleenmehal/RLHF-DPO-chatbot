---
name: neo4j
description: Neo4j graph database expert — use this skill whenever the user wants to create nodes, relationships, or schemas in Neo4j, write or run Cypher queries, model data as a graph, explore an existing Neo4j database, import data into Neo4j, or do anything involving the Neo4j MCP server. Trigger on keywords like Neo4j, Cypher, graph database, nodes, relationships, labels, properties (in a graph context), knowledge graph, or graph traversal. Also trigger if the user says things like "add this to my graph", "query the database", "create a relationship between X and Y", or "model this as a graph".
---

# Neo4j Skill

You have access to the `neo4j-cypher` MCP server. Use it to interact with a live Neo4j database.

## Available MCP Tools

The MCP server exposes these tools (prefixed with `mcp__neo4j_cypher__` or similar):
- **query** — Run any Cypher query (read or write)
- **get_schema** — Retrieve the current database schema (node labels, relationship types, properties)

Always check which tools are actually available in your context by looking at the tool list.

## Workflow: Always Start with Schema

Before creating anything, call `get_schema` (or run `CALL apoc.meta.schema()` / `CALL db.schema.visualization()`) to understand what already exists. This prevents duplicate labels, mismatched property names, and broken relationships.

```cypher
// Quick schema overview
CALL db.labels()
CALL db.relationshipTypes()
CALL db.propertyKeys()
```

## Creating Nodes

```cypher
// Single node
CREATE (n:Person {name: "Alice", age: 30})

// Merge (create only if not exists — prefer this over CREATE for idempotency)
MERGE (n:Person {name: "Alice"})
ON CREATE SET n.age = 30, n.created = timestamp()
ON MATCH SET n.lastSeen = timestamp()
```

Use `MERGE` over `CREATE` when there's a natural unique identifier — it avoids duplicates on reruns.

## Creating Relationships

```cypher
// Always MATCH nodes first, then create the relationship
MATCH (a:Person {name: "Alice"})
MATCH (b:Person {name: "Bob"})
MERGE (a)-[:KNOWS {since: 2024}]->(b)
```

Relationship types are UPPERCASE_WITH_UNDERSCORES by convention.

## Querying

```cypher
// Find all nodes of a label
MATCH (n:Person) RETURN n LIMIT 25

// Traverse relationships
MATCH (a:Person)-[:KNOWS]->(b:Person)
RETURN a.name, b.name

// Filter and sort
MATCH (n:Person)
WHERE n.age > 25
RETURN n.name, n.age
ORDER BY n.age DESC
```

## Deleting

```cypher
// Delete a node and all its relationships (safe)
MATCH (n:Person {name: "Alice"})
DETACH DELETE n

// Delete just a relationship
MATCH ()-[r:KNOWS {since: 2020}]->()
DELETE r
```

## Modeling Tips

| Scenario | Pattern |
|---|---|
| Entity with attributes | Node with label + properties |
| Connection between entities | Relationship with type |
| Attribute shared by many nodes | Separate node, linked by relationship |
| Time-varying connection | Property on the relationship |
| Hierarchical data | `:PARENT_OF` / `:CHILD_OF` relationships |

**Good graph modeling rule**: If you'd query "find all X connected to Y", it belongs as a relationship. If it's just a fact about one thing, it belongs as a property.

## Step-by-step for Common Tasks

### "Create a graph from this data"
1. Call `get_schema` — understand what exists
2. Identify entities (→ nodes) and connections (→ relationships)
3. Choose labels and relationship types
4. Use `MERGE` to create nodes with unique identifiers
5. Use `MERGE` to create relationships between matched nodes
6. Verify with a `MATCH ... RETURN` query

### "Add X to the database"
1. Check if it already exists: `MATCH (n:Label {id: value}) RETURN n`
2. Use `MERGE` to create or update
3. Confirm the write with a retrieval query

### "Query the database for X"
1. Call `get_schema` if you don't know the structure
2. Write a `MATCH ... WHERE ... RETURN` query
3. Always include `LIMIT` on exploratory queries to avoid flooding the context

## Safety Rules

- Always use `LIMIT` when exploring (e.g., `LIMIT 25`)
- Prefer `MERGE` over `CREATE` to keep operations idempotent
- Before a bulk `DELETE`, run the same `MATCH` with `RETURN count(n)` first to see what would be affected
- Never run `MATCH (n) DETACH DELETE n` without explicit user confirmation — this wipes the entire database

## Connection Details

The MCP server connects using the environment variables set during installation:
- `NEO4J_URI` — e.g., `bolt://localhost:7687`
- `NEO4J_USERNAME` / `NEO4J_PASSWORD`
- `NEO4J_DATABASE` — defaults to `neo4j`

If the MCP server isn't responding, remind the user to ensure Neo4j is running:
```bash
docker start neo4j
# or check: docker ps | grep neo4j
```
