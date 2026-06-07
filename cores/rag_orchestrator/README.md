# RAG Orchestrator Module

## Overview

The RAG Orchestrator is the central module for managing Retrieval-Augmented Generation (RAG) capabilities in the UBP Enterprise Hybrid platform. It provides a complete RAG solution with knowledge base management, access control, configuration tuning, and an interactive chat simulator.

## Features

### 1. Knowledge Base Management
- **Create Knowledge Bases**: Create new Qdrant collections to store document embeddings
- **Document Ingestion**: Upload and process text documents (.txt, .md) into searchable chunks
- **Automatic Chunking**: Intelligent text splitting with configurable chunk sizes and overlap
- **Metadata Support**: Attach custom metadata to documents for enhanced filtering and retrieval

### 2. Access Control (ACL)
- **Fine-grained Permissions**: Control read/write access at the user and client level
- **Per-Collection ACL**: Set different permissions for different knowledge bases
- **Redis-backed Storage**: Fast and scalable permission management
- **Default Access Policies**: Configurable default access behavior

### 3. RAG Configuration & Tuning
- **Entity-specific Configs**: Different RAG settings for users, clients, or system default
- **Model Selection**: Choose which LLM model to use for generation
- **Temperature Control**: Adjust creativity vs consistency with temperature slider (0-1)
- **Top-K Retrieval**: Configure how many document chunks to retrieve
- **Custom System Prompts**: Define specialized prompts for different use cases
- **Configuration Hierarchy**: Entity-specific configs with fallback to defaults

### 4. RAG Chat Simulator
- **Interactive Testing**: Test your RAG pipeline in real-time
- **Source Display**: See which documents were used to generate answers
- **Debug Panel**: View retrieval scores, configuration used, and permissions checked
- **Real-time Chat**: Chat interface with user/bot message history

## Architecture

The RAG Orchestrator follows the UBP 3-file pattern:

```
rag_orchestrator/
├── __init__.py          # Module initialization
├── adapter.py           # UBP bridge layer (BaseHybridModule)
├── providers.py         # Business logic (RAGPipeline, DocumentChunker)
├── manifest.json        # Module metadata and operations
├── config.json          # Configuration settings
└── README.md            # This file
```

### Dependencies

The RAG Orchestrator requires the following modules:
- **rag_qdrant**: Vector database operations (Qdrant integration)
- **inference_ollama_grok**: LLM inference for answer generation
- **Redis**: For ACL and configuration storage

## Configuration

### Default RAG Configuration (config.json)

```json
{
  "default_rag_config": {
    "model": "tinyllama",
    "temperature": 0.7,
    "top_k": 5,
    "system_prompt": "You are a helpful AI assistant. Answer based on the provided context."
  },
  "chunking": {
    "chunk_size": 500,
    "overlap": 50
  },
  "acl": {
    "redis_key_prefix": "rag:acl",
    "default_access": "none"
  },
  "rag_config_storage": {
    "redis_key_prefix": "rag:config"
  },
  "security": {
    "max_query_length": 500,
    "max_collections_per_query": 5
  }
}
```

## API Operations

### Knowledge Base Management

#### `create_knowledge_base`
Create a new knowledge base (Qdrant collection).

**Parameters:**
- `name` (string, required): Unique collection name
- `description` (string, optional): Description of the knowledge base

**Returns:**
```json
{
  "collection_id": "string",
  "status": "created|error",
  "message": "string"
}
```

#### `ingest_document`
Ingest a text document into a knowledge base.

**Parameters:**
- `collection_id` (string, required): Target knowledge base
- `text` (string, required): Document text content
- `metadata` (object, optional): Custom metadata

**Returns:**
```json
{
  "document_id": "string",
  "chunks_count": 10,
  "status": "success|error",
  "message": "string"
}
```

#### `list_knowledge_bases`
List all knowledge bases accessible by the user.

**Parameters:**
- `context_user_id` (string, optional): Filter by user access
- `context_client_id` (string, optional): Filter by client access

**Returns:**
```json
{
  "collections": [{"name": "kb_name"}],
  "count": 5
}
```

### Access Control

#### `set_permission`
Set access permission for a user or client on a collection.

**Parameters:**
- `entity_type` (string, required): "user" or "client"
- `entity_id` (string, required): User ID or Client ID
- `collection_id` (string, required): Target collection
- `access_level` (string, required): "read", "write", or "none"

**Returns:**
```json
{
  "status": "success|error",
  "message": "string"
}
```

#### `get_permissions`
Get permissions for a collection or entity.

**Parameters:**
- `collection_id` (string, optional): Filter by collection
- `entity_type` (string, optional): Filter by "user" or "client"
- `entity_id` (string, optional): Filter by specific entity

**Returns:**
```json
{
  "permissions": [
    {
      "entity_type": "user",
      "entity_id": "user123",
      "collection_id": "kb_docs",
      "access_level": "read"
    }
  ],
  "count": 10
}
```

### RAG Configuration

#### `set_rag_config`
Set RAG configuration for an entity.

**Parameters:**
- `entity_type` (string, required): "user", "client", or "default"
- `entity_id` (string, optional): ID (null for default)
- `config_json` (object, required): Configuration object

**Returns:**
```json
{
  "status": "success|error",
  "message": "string"
}
```

#### `get_rag_config`
Get RAG configuration with fallback to defaults.

**Parameters:**
- `entity_type` (string, required): "user", "client", or "default"
- `entity_id` (string, optional): ID

**Returns:**
```json
{
  "config": {
    "model": "tinyllama",
    "temperature": 0.7,
    "top_k": 5,
    "system_prompt": "..."
  },
  "source": "user_specific|default|config_file_default"
}
```

### RAG Chat

#### `rag_chat` (The Magic Function)
Execute the complete RAG pipeline with ACL checks.

**Parameters:**
- `query` (string, required): User question
- `context_user_id` (string, required): User ID for ACL and config
- `context_client_id` (string, optional): Client ID for ACL and config
- `collections` (array, optional): Specific collections to query
- `override_config` (object, optional): Temporary config override

**Returns:**
```json
{
  "answer": "Generated answer based on retrieved context",
  "mode": "rag | pure_llm | error",
  "mode_reason": "no_kb_access | acl_filtered_all | explicit_pure_llm | null",
  "sources": [
    {
      "text": "chunk text",
      "score": 0.95,
      "metadata": {
        "document_id": "uuid",
        "chunk_index": 0
      }
    }
  ],
  "config_used": {...},
  "permissions_checked": ["kb1", "kb2"],
  "conversation_id": "uuid-v4",
  "debug": {...}
}
```

## Inference Modes

The RAG Orchestrator supports multiple inference modes to ensure users can always interact with the system, even without Knowledge Base access.

### Mode Types

| Mode | Value | Description |
|------|-------|-------------|
| **RAG** | `"rag"` | Full Retrieval-Augmented Generation with KB context. Sources are retrieved and included in the response. |
| **Pure LLM** | `"pure_llm"` | Direct LLM inference without retrieval. Used as fallback when user has no KB access. |
| **Error** | `"error"` | Chat pipeline failed. Check the error message for details. |

### Mode Reasons

When `mode` is `"pure_llm"`, the `mode_reason` field explains why:

| Reason | Description |
|--------|-------------|
| `no_kb_access` | User has no permissions on any Knowledge Base |
| `acl_filtered_all` | User requested specific KBs but has no access to any of them |
| `explicit_pure_llm` | User explicitly requested Pure LLM mode (future feature) |
| `null` | Mode is RAG (no fallback needed) |

### Automatic Fallback Behavior

```
User sends query
    │
    ▼
Check KB permissions
    │
    ├─── Has KB access ──► RAG Mode (retrieve + generate)
    │                           │
    │                           ▼
    │                      Return answer with sources
    │
    └─── No KB access ──► Pure LLM Mode (generate only)
                               │
                               ▼
                          Return answer without sources
                          + mode: "pure_llm"
                          + mode_reason: "no_kb_access"
```

### Frontend Integration

The frontend displays visual indicators based on the inference mode:
- **RAG Mode**: Green badge showing "RAG: N fonti" (N sources)
- **Pure LLM Mode**: Amber banner explaining the response is from LLM only

## Usage Examples

### 1. Create a Knowledge Base

```javascript
const response = await fetch('/api/modules/rag_orchestrator/create_knowledge_base', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    name: 'company_docs',
    description: 'Company documentation and policies'
  })
});
```

### 2. Ingest a Document

```javascript
const response = await fetch('/api/modules/rag_orchestrator/ingest_document', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    collection_id: 'company_docs',
    text: 'Your document text here...',
    metadata: {
      source: 'manual',
      category: 'hr_policies'
    }
  })
});
```

### 3. Set User Permission

```javascript
const response = await fetch('/api/modules/rag_orchestrator/set_permission', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    entity_type: 'user',
    entity_id: 'user123',
    collection_id: 'company_docs',
    access_level: 'read'
  })
});
```

### 4. Configure RAG for a User

```javascript
const response = await fetch('/api/modules/rag_orchestrator/set_rag_config', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    entity_type: 'user',
    entity_id: 'user123',
    config_json: {
      model: 'mistral',
      temperature: 0.5,
      top_k: 3,
      system_prompt: 'You are a technical support assistant.'
    }
  })
});
```

### 5. Ask a Question (RAG Chat)

```javascript
const response = await fetch('/api/modules/rag_orchestrator/rag_chat', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    query: 'What is the company vacation policy?',
    context_user_id: 'user123',
    context_client_id: 'web_app'
  })
});

const result = await response.json();
console.log('Answer:', result.result.answer);
console.log('Sources:', result.result.sources);
```

## Admin UI Features

The RAG Management UI in the admin dashboard provides:

### 📚 Knowledge Bases Tab
- View all knowledge bases
- Create new knowledge bases with descriptions
- Upload documents (.txt, .md files)
- Paste text directly for ingestion
- Add custom metadata to documents

### 🔐 Access Control Tab
- Select a knowledge base
- View all users and clients
- Set read/write permissions with checkboxes
- Real-time permission updates

### ⚙️ Tuning Tab
- Select entity type (Default, User, Client)
- Configure system prompts
- Adjust temperature with slider (0-1)
- Set Top-K retrieval count
- Choose LLM model
- Save and load configurations

### 💬 Simulator Tab
- Interactive chat interface
- Real-time RAG question answering
- Debug panel showing:
  - Retrieved source documents
  - Similarity scores
  - Configuration used
  - Collections queried

## Security Features

1. **ACL Enforcement**: All queries check user/client permissions before accessing collections
2. **Query Length Limits**: Prevent excessive query sizes (configurable)
3. **Collection Limits**: Maximum collections per query (configurable)
4. **Default Deny**: No access by default unless explicitly granted
5. **Audit Trail**: All operations logged for security monitoring

## Performance Considerations

- **Redis Caching**: Fast permission lookups and config retrieval
- **Batch Operations**: Efficient document chunking and ingestion
- **Vector Search**: Optimized Qdrant queries with configurable Top-K
- **Async Operations**: Non-blocking I/O for all database operations

## Troubleshooting

### Common Issues

1. **"No knowledge bases found"**
   - Create a knowledge base first using the "New Knowledge Base" button
   - Ensure Qdrant service is running

2. **"Access denied"**
   - Check permissions in the Access Control tab
   - Verify user/client has read access to the collection

3. **"No answer generated"**
   - Ensure documents have been ingested
   - Check that the LLM model is available (AI Models tab)
   - Verify Ollama service is running

4. **"Configuration not loading"**
   - Check Redis connection
   - Verify entity ID is correct
   - Default config always available as fallback

## Integration with Other Modules

- **rag_qdrant**: Provides vector database operations
- **inference_ollama_grok**: Generates answers using LLM
- **admin_users**: User management for ACL
- **admin_clients**: Client management for ACL

## Future Enhancements

- [ ] Bulk document upload
- [ ] Advanced filtering (by metadata)
- [x] Conversation history *(completed v1.5.0)*
- [x] Multi-turn dialogue support *(completed v1.5.0)*
- [x] Pure LLM fallback mode *(completed v1.5.1)*
- [ ] Analytics and usage metrics
- [ ] Document versioning
- [ ] Collection cloning/backup
- [ ] Advanced chunking strategies (semantic, sentence-based)

## License

Part of the UBP Enterprise Hybrid platform.

## Support

For issues and questions, please refer to the main UBP documentation or contact the development team.
