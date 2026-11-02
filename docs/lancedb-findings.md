# LanceDB Cloud Probe Findings

## Sample Data
- **Samples processed**: 2
- **Samples by title**: Introduction to dbt (Drew Banin), YugabyteDB: A Brief Introduction (Hari Krishna Sunder)

## Connectivity
- **LanceDB Cloud URI**: db://cmu-database-group-eqdwgi
- **Region**: us-east-1
- **Table**: lectures_sample_probe
- **Status**: Successfully connected and created table

## Query Support
- **Vector Search**: ✓ Supported
- **FTS Search**: ✓ Supported
- **Hybrid Search**: ✓ Supported
- **Read-Only Keys**: ✗ Not supported

## Observations
- All vector queries use Nomic v1.5 768d embeddings
- Search document prefix: `search_document: `
- Search query prefix: `search_query: `
- Hybrid search uses equal weights (0.5/0.5) for vector and text components

## Recommendations
1. Vector search is the primary capability for semantic queries
2. FTS provides keyword-based search fallback
3. Hybrid search combines both approaches for optimal results
4. Read-only keys should be used for production inference endpoints
