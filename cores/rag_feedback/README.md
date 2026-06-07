# RAG Feedback Module v1.0

Feedback collection and analytics for RAG responses in UBP Enterprise Hybrid.

## Architecture Overview

This module implements the **3-File Pattern Architecture**:

```
rag_feedback/
├── __init__.py          # Entry point with create_module() factory
├── adapter.py           # FeedbackAdapter (UBP integration layer)
├── providers.py         # RedisFeedbackProvider (business logic)
├── config.json          # Configuration
├── manifest.json        # Module metadata and operations
└── README.md            # This file
```

## Features

### Feedback Types
- **Thumbs Up/Down**: Quick binary feedback
- **Rating**: 1-5 star ratings
- **Contextual Data**: Store query, answer preview, collection name

### Analytics
- **Aggregated Stats**: Total feedback, satisfaction rates
- **Time-based Analysis**: Filter by date range
- **Collection-level Stats**: Per-knowledge-base analytics

### Production Features
- **Redis Persistence**: Durable storage with configurable TTL
- **User Isolation**: Users see own feedback, admins see all
- **Event Publishing**: Publishes events on feedback submission
- **Request Tracking**: All operations return `request_id` for tracing

## Operations

| Operation | Description | Auth Required |
|-----------|-------------|---------------|
| `submit_feedback` | Submit feedback for a response | Yes |
| `get_response_feedback` | Get feedback for a specific response | Yes |
| `get_feedback_stats` | Get aggregated statistics | Yes |
| `list_feedback` | List feedback entries | Yes |
| `delete_feedback` | Delete feedback | Yes |
| `health_check` | Check module health | No |

## Configuration

### config.json

```json
{
  "enabled": true,
  "stats_ttl_seconds": 604800,
  "feedback_ttl_seconds": 7776000,
  "max_feedback_per_user": 1000,
  "allow_anonymous_feedback": false
}
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | true | Enable/disable module |
| `stats_ttl_seconds` | integer | 604800 | Stats cache TTL (7 days) |
| `feedback_ttl_seconds` | integer | 7776000 | Feedback retention (90 days) |
| `max_feedback_per_user` | integer | 1000 | Max feedback per user |
| `allow_anonymous_feedback` | boolean | false | Allow unauthenticated feedback |

## Redis Key Schema

Following NAMING_POLICY.md Section 7:

```
ubp:{env}:feedback:response:{response_id}      # Feedback data
ubp:{env}:feedback:stats:daily:{date}          # Daily aggregates
ubp:{env}:feedback:stats:collection:{name}     # Collection aggregates
ubp:{env}:feedback:user:{user_id}:list         # User's feedback index
ubp:{env}:feedback:all:list                    # Global feedback index
```

## Events

### Published Events
- `feedback.submitted` - New feedback submitted
- `feedback.deleted` - Feedback deleted

## API Usage Examples

### Submit Thumbs Up
```bash
curl -X POST "http://localhost:8000/api/modules/rag_feedback/submit_feedback" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "response_id": "uuid-here",
    "feedback_type": "thumbs_up",
    "value": true,
    "query": "What is Python?",
    "answer": "Python is a programming language..."
  }'
```

### Submit Rating
```bash
curl -X POST "http://localhost:8000/api/modules/rag_feedback/submit_feedback" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "response_id": "uuid-here",
    "feedback_type": "rating",
    "value": 5,
    "collection": "python_docs"
  }'
```

### Get Statistics
```bash
curl -X POST "http://localhost:8000/api/modules/rag_feedback/get_feedback_stats" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2025-01-01",
    "end_date": "2025-01-31",
    "collection": "python_docs"
  }'
```

## UI Integration

The feedback buttons in `rag_admin.html` integrate with this module:

```javascript
// After RAG response is displayed
async function submitFeedback(responseId, isPositive) {
    await fetch('/api/modules/rag_feedback/submit_feedback', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            response_id: responseId,
            feedback_type: 'thumbs_up',
            value: isPositive
        })
    });
}
```

## Statistics Response Example

```json
{
  "total_feedback": 150,
  "thumbs_up": 120,
  "thumbs_down": 20,
  "ratings_count": 10,
  "average_rating": 4.2,
  "satisfaction_rate": 85.7,
  "period": {
    "start": "2025-01-01",
    "end": "2025-01-31"
  },
  "request_id": "req-uuid"
}
```

## Dependencies

- **Required**: Redis (for feedback storage)
- **Required**: Event Bus (for event publishing)

## Testing

```bash
# Run module tests
cd ubp_enterprise_hybrid
pytest tests/integration/test_rag_v150_modules.py -k "feedback" -v
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-31 | Initial release with ROADMAP v1.5.0 |

---

**Module Type:** analytics  
**Architecture:** 3-file-pattern  
**Production Ready:** Yes
