# Admin Clients Module

Enterprise-grade OAuth/API client management module for UBP Enterprise Hybrid Edition.

## Overview

The `admin_clients` module provides comprehensive client management operations for OAuth2, API keys, and service accounts with enterprise-level security, including secure secret generation, PBKDF2-SHA256 secret hashing, multi-tenancy support, and event-driven architecture.

## Features

### Security
- **Secure Secret Generation**: Cryptographically secure random secrets using Python's `secrets` module
- **Secret Hashing**: PBKDF2-SHA256 (configurable)
- **O(1) Client Name Lookup**: Redis-based index for fast client_name uniqueness checks
- **Secret Rotation**: Generate new secrets and invalidate old ones
- **Soft Revocation**: Disable clients without deletion
- **Audit Logging**: All operations logged with request tracking

### Client Types
- **OAuth2**: Full OAuth2 flow support with redirect URIs and scopes
- **API Key**: Simple API key authentication
- **Service Account**: Machine-to-machine authentication

### Operations
- **create_client**: Create new client with secure secret generation
- **list_clients**: List all clients with filtering and pagination
- **get_client**: Retrieve client by ID
- **update_client**: Update client information (including client_name change with index update)
- **delete_client**: Permanently delete client
- **rotate_secret**: Generate new secret and invalidate old one
- **revoke_client**: Soft delete (set is_active to false)
- **get_client_stats**: Client management statistics (total, active, by type, by tenant)
- **health_check**: Module health status with Redis connectivity
- **assign_kb_to_user** (v1.2.3): Assign KB to specific user with custom access level
- **revoke_kb_from_user** (v1.2.3): Revoke user's custom KB access, revert to inheritance
- **get_user_kb_assignments** (v1.2.3): Query inherited, custom, and effective KB access
- **ingest_to_client_kb** (v1.2.3): Ingest document to client KB with ACL verification

### Integration
- **Event Bus**: Publishes lifecycle events (`admin.client.created`, `admin.client.updated`, `admin.client.deleted`, `admin.client.secret_rotated`, `admin.client.revoked`)
- **Auto-Discovery**: Automatically discovered by UBP module loader
- **Hot Reload**: Configuration changes applied without restart
- **Statistics Tracking**: All operations tracked automatically by framework

## Architecture

### Pattern A - UBP Module
```
admin_clients/
├── __init__.py      # Factory entry point
├── adapter.py       # AdminClientsAdapter(BaseHybridModule)
├── providers.py     # Pure technical logic (ZERO UBP dependencies)
├── manifest.json    # UBP standard manifest
├── config.json      # Redis keys, security, validation
└── README.md        # This file
```

### Dependencies
- **Redis**: Required for client storage and indexing
- **passlib**: Required for secret hashing (PBKDF2-SHA256 via passlib CryptContext)
- **Event Bus**: Required for lifecycle events

### Storage Schema

#### Redis Keys
```
ubp:admin:clients                          # Hash: client_id -> client JSON (index)
ubp:admin:client:{client_id}              # String: Full client object
ubp:admin:client_name_index:{client_name} # String: client_id (O(1) lookup)
```

#### Client Object
```json
{
  "client_id": "uuid",
  "client_name": "string",
  "client_secret_hash": "pbkdf2_sha256 hash",
  "client_type": "oauth2|api_key|service_account",
  "description": "string|null",
  "redirect_uris": ["uri1", "uri2"],
  "scopes": ["scope1", "scope2"],
  "tenant_id": "string|null",
  "is_active": true,
  "created_at": "ISO datetime",
  "updated_at": "ISO datetime",
  "expires_at": "ISO datetime|null",
  "last_used_at": "ISO datetime|null",
  "secret_rotated_at": "ISO datetime|null",
  "revoked_at": "ISO datetime|null",
  "revocation_reason": "string|null"
}
```

## Installation

### 1. Install Dependencies
```bash
pip install passlib[bcrypt]
```

### 2. Configure Redis
Ensure Redis is available and configured in `backend/app/core/config.py`.

### 3. Module Auto-Discovery
The module is automatically discovered by UBP framework on startup. No manual registration needed.

## Usage

### API Endpoints

All operations available via dynamic routing:
```
POST   /api/modules/admin_clients/initialize
POST   /api/modules/admin_clients/create_client
GET    /api/modules/admin_clients/list_clients
GET    /api/modules/admin_clients/get_client
PUT    /api/modules/admin_clients/update_client
DELETE /api/modules/admin_clients/delete_client
POST   /api/modules/admin_clients/rotate_secret
POST   /api/modules/admin_clients/revoke_client
GET    /api/modules/admin_clients/get_client_stats
GET    /api/modules/admin_clients/health_check

# NEW in v1.2.3 - KB Permission Management
POST   /api/modules/admin_clients/assign_kb_to_user
POST   /api/modules/admin_clients/revoke_kb_from_user
POST   /api/modules/admin_clients/get_user_kb_assignments
POST   /api/modules/admin_clients/ingest_to_client_kb
```

### Examples

#### Initialize Module
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/initialize
```

**Response**:
```json
{
  "success": true,
  "module": "admin_clients",
  "operation": "initialize",
  "result": {
    "status": "initialized",
    "module": "admin_clients",
    "storage": "redis",
    "security": {
      "secret_generation": "alphanumeric",
      "secret_length": 32,
      "secret_hashing": "bcrypt",
      "bcrypt_rounds": 12
    },
    "features": {
      "client_name_index": "O(1) lookup",
      "event_bus": "enabled",
      "multi_tenancy": "supported",
      "secret_rotation": "enabled",
      "soft_revocation": "enabled"
    }
  }
}
```

#### Create OAuth2 Client
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/create_client \
  -H "Content-Type: application/json" \
  -d '{
    "client_name": "my_oauth_app",
    "client_type": "oauth2",
    "description": "My OAuth2 Application",
    "redirect_uris": ["https://example.com/callback"],
    "scopes": ["read", "write"],
    "is_active": true
  }'
```

**Response** (note: client_secret only returned once):
```json
{
  "success": true,
  "module": "admin_clients",
  "operation": "create_client",
  "result": {
    "client_id": "550e8400-e29b-41d4-a716-446655440000",
    "client_secret": "a8B3dF9kL2mN4pQ6rS8tU0vW2xY4zA6",
    "client_name": "my_oauth_app",
    "client_type": "oauth2",
    "description": "My OAuth2 Application",
    "redirect_uris": ["https://example.com/callback"],
    "scopes": ["read", "write"],
    "tenant_id": null,
    "is_active": true,
    "created_at": "2025-12-25T22:30:00.000000",
    "updated_at": "2025-12-25T22:30:00.000000",
    "expires_at": null,
    "last_used_at": null,
    "secret_rotated_at": null,
    "request_id": "123e4567-e89b-12d3-a456-426614174000"
  }
}
```

#### Create API Key Client
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/create_client \
  -H "Content-Type: application/json" \
  -d '{
    "client_name": "mobile_app_v1",
    "client_type": "api_key",
    "description": "Mobile App API Key",
    "scopes": ["mobile_api"],
    "expires_at": "2026-12-31T23:59:59.000000"
  }'
```

#### List Clients (with filtering)
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/list_clients \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {
      "is_active": true,
      "client_type": "oauth2"
    },
    "limit": 10,
    "offset": 0
  }'
```

#### Update Client
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/update_client \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "550e8400-e29b-41d4-a716-446655440000",
    "description": "Updated OAuth2 Application",
    "scopes": ["read", "write", "delete"],
    "is_active": true
  }'
```

#### Rotate Secret
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/rotate_secret \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

**Response** (note: new client_secret only returned once):
```json
{
  "success": true,
  "module": "admin_clients",
  "operation": "rotate_secret",
  "result": {
    "message": "Client secret rotated successfully",
    "client_id": "550e8400-e29b-41d4-a716-446655440000",
    "client_secret": "x9Y2wV4uT8sR6qP4nM2lK0jH9gF7eD5",
    "rotated_at": "2025-12-25T23:00:00.000000",
    "request_id": "234e5678-e89b-12d3-a456-426614174001"
  }
}
```

#### Revoke Client (Soft Delete)
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/revoke_client \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "550e8400-e29b-41d4-a716-446655440000",
    "reason": "Security breach suspected"
  }'
```

#### Get Statistics
```bash
curl http://localhost:8000/api/modules/admin_clients/get_client_stats
```

**Response**:
```json
{
  "success": true,
  "module": "admin_clients",
  "operation": "get_client_stats",
  "result": {
    "module": "admin_clients",
    "total_clients": 45,
    "active_clients": 40,
    "revoked_clients": 5,
    "clients_by_type": {
      "oauth2": 15,
      "api_key": 25,
      "service_account": 5
    },
    "clients_by_tenant": {
      "default": 30,
      "tenant_1": 10,
      "tenant_2": 5
    },
    "storage": "redis",
    "security": {
      "secret_hashing": "bcrypt",
      "client_name_index": "O(1)"
    }
  }
}
```

#### Health Check
```bash
curl http://localhost:8000/api/modules/admin_clients/health_check
```

**Response**:
```json
{
  "success": true,
  "module": "admin_clients",
  "operation": "health_check",
  "result": {
    "module": "admin_clients",
    "status": "healthy",
    "initialized": true,
    "redis": {
      "connected": true,
      "status": "healthy"
    }
  }
}
```

### KB Permission Management (v1.2.3)

#### Assign KB to User
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/assign_kb_to_user \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "35f96c24-5d9c-4c5c-95b7-e3f5330dae41",
    "user_id": "5be073e2-a17e-4601-8d6b-5dfdbccc6a98",
    "kb_name": "special_project",
    "access_level": "write"
  }'
```

**Response**:
```json
{
  "success": true,
  "module": "admin_clients",
  "operation": "assign_kb_to_user",
  "result": {
    "status": "success",
    "message": "KB 'special_project' assigned to user with 'write' access",
    "assignment": {
      "user_id": "5be073e2-a17e-4601-8d6b-5dfdbccc6a98",
      "kb_name": "special_project",
      "access_level": "write",
      "assigned_at": "2026-01-08T10:30:00.000000"
    }
  }
}
```

#### Revoke KB from User
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/revoke_kb_from_user \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "35f96c24-5d9c-4c5c-95b7-e3f5330dae41",
    "user_id": "5be073e2-a17e-4601-8d6b-5dfdbccc6a98",
    "kb_name": "special_project"
  }'
```

**Response**:
```json
{
  "success": true,
  "module": "admin_clients",
  "operation": "revoke_kb_from_user",
  "result": {
    "status": "success",
    "message": "Custom KB assignment revoked. User now inherits client-level access.",
    "inherited_access": "read"
  }
}
```

#### Get User KB Assignments
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/get_user_kb_assignments \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "35f96c24-5d9c-4c5c-95b7-e3f5330dae41",
    "user_id": "5be073e2-a17e-4601-8d6b-5dfdbccc6a98"
  }'
```

**Response**:
```json
{
  "success": true,
  "module": "admin_clients",
  "operation": "get_user_kb_assignments",
  "result": {
    "status": "success",
    "user_id": "5be073e2-a17e-4601-8d6b-5dfdbccc6a98",
    "inherited_kbs": {
      "company_docs": "read",
      "faq_general": "read"
    },
    "custom_kbs": {
      "special_project": "write"
    },
    "effective_access": {
      "company_docs": {"level": "read", "source": "client"},
      "faq_general": {"level": "read", "source": "client"},
      "special_project": {"level": "write", "source": "custom"}
    }
  }
}
```

## Configuration

### config.json

```json
{
  "redis": {
    "keys": {
      "clients_index": "ubp:admin:clients",
      "client_prefix": "ubp:admin:client:",
      "client_name_index_prefix": "ubp:admin:client_name_index:"
    }
  },
  "security": {
    "secret_generation": {
      "length": 32,
      "charset": "alphanumeric"
    },
    "secret_hashing": {
      "algorithm": "bcrypt",
      "rounds": 12
    }
  },
  "defaults": {
    "scopes": [],
    "is_active": true,
    "client_type": "api_key"
  },
  "validation": {
    "client_name": {
      "min_length": 3,
      "max_length": 100,
      "pattern": "^[a-zA-Z0-9_-]+( [a-zA-Z0-9_-]+)*$"
    },
    "description": {
      "max_length": 500
    },
    "client_type": {
      "allowed_values": ["oauth2", "api_key", "service_account"]
    }
  }
}
```

### Customization

#### Increase Secret Length
```json
{
  "security": {
    "secret_generation": {
      "length": 64,
      "charset": "alphanumeric"
    }
  }
}
```

#### Use Hexadecimal Secrets
```json
{
  "security": {
    "secret_generation": {
      "length": 32,
      "charset": "hex"
    }
  }
}
```

#### Increase Secret Hash Strength
```json
{
  "security": {
    "secret_hashing": {
      "rounds": 14
    }
  }
}
```

## Events

### Published Events

The module publishes these events to the event bus:

#### admin.client.created
```json
{
  "client_id": "uuid",
  "client_name": "string",
  "client_type": "string",
  "scopes": ["scope1", "scope2"],
  "tenant_id": "string|null",
  "timestamp": "ISO datetime",
  "request_id": "uuid"
}
```

#### admin.client.updated
```json
{
  "client_id": "uuid",
  "client_name": "string",
  "changes": {
    "client_name": false,
    "scopes": true,
    "is_active": false,
    "expires_at": true
  },
  "timestamp": "ISO datetime",
  "request_id": "uuid"
}
```

#### admin.client.deleted
```json
{
  "client_id": "uuid",
  "client_name": "string",
  "timestamp": "ISO datetime",
  "request_id": "uuid"
}
```

#### admin.client.secret_rotated
```json
{
  "client_id": "uuid",
  "rotated_at": "ISO datetime",
  "request_id": "uuid"
}
```

#### admin.client.revoked
```json
{
  "client_id": "uuid",
  "client_name": "string",
  "reason": "string|null",
  "revoked_at": "ISO datetime",
  "request_id": "uuid"
}
```

### Event Subscribers

Other modules can subscribe to these events:

```python
async def on_client_created(event: Event):
    """React to new client creation."""
    client_id = event.payload["client_id"]
    client_name = event.payload["client_name"]
    # ... handle event
```

## Security Best Practices

### 1. Never Return client_secret_hash
All operations automatically exclude `client_secret_hash` from responses.

### 2. client_secret Only Returned Once
The plaintext `client_secret` is only returned during:
- `create_client` - Initial creation
- `rotate_secret` - Secret rotation

Store it securely immediately!

### 3. Client Name Index for O(1) Lookup
Client name uniqueness checked in O(1) time using Redis index.

### 4. Bcrypt Rounds
Default 12 rounds provides strong security. Increase to 14+ for higher security environments.

### 5. Secret Rotation
Regularly rotate secrets for active clients to minimize compromise risk.

### 6. Soft Revocation
Use `revoke_client` instead of `delete_client` to maintain audit trail.

### 7. Request Tracking
All operations include `request_id` for audit trails and debugging.

## Performance

### Metrics
- **create_client**: ~50ms (bcrypt hashing + secret generation + 3 Redis operations)
- **list_clients**: ~10ms (single Redis HGETALL)
- **get_client**: ~2ms (single Redis GET)
- **update_client**: ~15ms (2-3 Redis operations)
- **delete_client**: ~10ms (3 Redis DEL operations)
- **rotate_secret**: ~55ms (secret generation + bcrypt hash + 2 Redis operations)
- **revoke_client**: ~10ms (2 Redis operations)

### Optimization
- Client name index provides O(1) uniqueness checking
- Redis operations pipelined where possible
- Secret generation uses cryptographically secure random
- Bcrypt hashing async-friendly (non-blocking)

## Testing

### Full CRUD Test
```bash
# 1. Initialize module
curl -X POST http://localhost:8000/api/modules/admin_clients/initialize

# 2. Create client
CLIENT_ID=$(curl -X POST http://localhost:8000/api/modules/admin_clients/create_client \
  -H "Content-Type: application/json" \
  -d '{"client_name":"test_client","client_type":"api_key"}' | jq -r '.result.client_id')

# 3. Get client
curl -X POST http://localhost:8000/api/modules/admin_clients/get_client \
  -H "Content-Type: application/json" \
  -d "{\"client_id\":\"$CLIENT_ID\"}"

# 4. Update client
curl -X POST http://localhost:8000/api/modules/admin_clients/update_client \
  -H "Content-Type: application/json" \
  -d "{\"client_id\":\"$CLIENT_ID\",\"description\":\"Updated test client\"}"

# 5. Rotate secret
curl -X POST http://localhost:8000/api/modules/admin_clients/rotate_secret \
  -H "Content-Type: application/json" \
  -d "{\"client_id\":\"$CLIENT_ID\"}"

# 6. Revoke client
curl -X POST http://localhost:8000/api/modules/admin_clients/revoke_client \
  -H "Content-Type: application/json" \
  -d "{\"client_id\":\"$CLIENT_ID\",\"reason\":\"Test completed\"}"

# 7. Delete client
curl -X POST http://localhost:8000/api/modules/admin_clients/delete_client \
  -H "Content-Type: application/json" \
  -d "{\"client_id\":\"$CLIENT_ID\"}"

# 8. Get statistics
curl http://localhost:8000/api/modules/admin_clients/get_client_stats

# 9. Health check
curl http://localhost:8000/api/modules/admin_clients/health_check
```

## Troubleshooting

### Error: "passlib[bcrypt] not installed"
**Solution**: Install dependency
```bash
pip install passlib[bcrypt]
```

### Error: "Redis client not available"
**Solution**: Check Redis connection and ensure Redis is running
```bash
redis-cli ping  # Should return PONG
```

### Error: "Client name already exists"
**Solution**: Choose different client_name or delete existing client

### Error: "Client type must be one of: oauth2, api_key, service_account"
**Solution**: Use valid client_type value

## Use Cases

### OAuth2 Application
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/create_client \
  -H "Content-Type: application/json" \
  -d '{
    "client_name": "web_app",
    "client_type": "oauth2",
    "redirect_uris": ["https://app.example.com/callback"],
    "scopes": ["openid", "profile", "email"]
  }'
```

### Mobile App API Key
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/create_client \
  -H "Content-Type: application/json" \
  -d '{
    "client_name": "mobile_app_ios",
    "client_type": "api_key",
    "scopes": ["mobile_api"],
    "expires_at": "2026-12-31T23:59:59.000000"
  }'
```

### Service Account (M2M)
```bash
curl -X POST http://localhost:8000/api/modules/admin_clients/create_client \
  -H "Content-Type: application/json" \
  -d '{
    "client_name": "backend_service",
    "client_type": "service_account",
    "scopes": ["internal_api", "admin"]
  }'
```

## Version History

### 1.2.3 (2026-01-08)
- **KB Permission Management**:
  - `assign_kb_to_user`: Assign KB to specific user with custom access level
  - `revoke_kb_from_user`: Revoke user's custom KB access, revert to inheritance
  - `get_user_kb_assignments`: Query inherited, custom, and effective KB access
- **KB Inheritance Fix (GAP-001)**:
  - Auto-sync ACLs when client created with `universal_kbs_assigned`
- **Cross-module Integration**:
  - Uses `admin_users.get_user_internal()` for user verification

### 1.0.0 (2025-12-25)
- Initial release
- Secure secret generation (cryptographically secure)
- bcrypt secret hashing
- O(1) client_name lookup
- Event bus integration
- Multi-tenancy support
- CRUD operations
- Secret rotation
- Soft revocation
- Statistics and health monitoring
- Support for OAuth2, API Key, and Service Account types

## License

UBP Enterprise Hybrid Edition - Proprietary License

## Author

UBP Team - Universal Backend Platform

## Support

For issues or questions, please contact the UBP development team.
