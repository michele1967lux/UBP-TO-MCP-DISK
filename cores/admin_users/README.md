# Admin Users Module

Enterprise-grade user management module for UBP Enterprise Hybrid Edition.

## Overview

The `admin_users` module provides comprehensive user management operations with enterprise-level security, including secure password hashing, role-based access control, multi-tenancy support, and event-driven architecture.

## Features

### Security
- **Password Hashing**: PBKDF2-SHA256 with 12000 iterations (primary, configurable via rounds * 1000)
  - Bcrypt available as fallback for legacy compatibility
  - Cross-platform portability without C++ dependencies
  - NIST-approved algorithm
- **O(1) Username Lookup**: Redis-based index for fast username uniqueness checks
- **Password Validation**: Configurable password policies
- **Audit Logging**: All operations logged with request tracking
- **MFA Support**: Framework ready for multi-factor authentication

### Operations
- **create_user**: Create new user with secure password hashing
- **list_users**: List all users with filtering and pagination
- **get_user**: Retrieve user by ID
- **update_user**: Update user information (including username change with index update)
- **delete_user**: Permanently delete user account
- **change_password**: Change password with old password verification
- **get_stats**: User management statistics (total, active, by role, by tenant)
- **health_check**: Module health status with Redis connectivity
- **ingest_to_personal_kb** (v1.2.3): Ingest document to user's Personal KB

### Internal Methods (Not API Exposed)
- **get_user_internal** (v1.2.3): Cross-module user lookup (used by `admin_clients` for GAP-003/004)

### Integration
- **Event Bus**: Publishes lifecycle events (`admin.user.created`, `admin.user.updated`, `admin.user.deleted`, `admin.user.password_changed`)
- **Auto-Discovery**: Automatically discovered by UBP module loader
- **Hot Reload**: Configuration changes applied without restart
- **Statistics Tracking**: All operations tracked automatically by framework

## Architecture

### Pattern A - UBP Module
```
admin_users/
├── adapter.py      # AdminUsersAdapter(BaseHybridModule)
├── manifest.json   # UBP standard manifest
├── config.json     # Redis keys, security, validation
└── README.md       # This file
```

### Dependencies
- **Redis**: Required for user storage and indexing
- **passlib[bcrypt]**: Required for password hashing
- **Event Bus**: Required for lifecycle events

### Storage Schema

#### Redis Keys
```
ubp:admin:users                        # Hash: user_id -> user JSON (index)
ubp:admin:user:{user_id}              # String: Full user object
ubp:admin:username_index:{username}    # String: user_id (O(1) lookup)
```

#### User Object
```json
{
  "user_id": "uuid",
  "username": "string",
  "password_hash": "$pbkdf2-sha256$12000$... (PBKDF2-SHA256 hash, 12000 iterations)",
  "email": "string|null",
  "full_name": "string|null",
  "roles": ["role1", "role2"],
  "tenant_id": "string|null",
  "is_active": true,
  "mfa_enabled": false,
  "created_at": "ISO datetime",
  "updated_at": "ISO datetime"
}
```

**Example from Redis DB**:
```json
{
  "user_id": "d31f9df7-c57f-4a0f-85fa-264ef9e2a706",
  "username": "admin",
  "password_hash": "$pbkdf2-sha256$12000$dI5xLmXM2bv3fg.h1DpHyA$F28/fNO9fNaF1nbNQ6dRXwkCGz.24D0jSY0dlA2laOk",
  "email": "admin@ubp-enterprise.local",
  "roles": ["admin"],
  "is_active": true,
  "created_at": "2025-12-28T06:12:36.099535"
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
POST   /api/modules/admin_users/create_user
GET    /api/modules/admin_users/list_users
GET    /api/modules/admin_users/get_user
PUT    /api/modules/admin_users/update_user
DELETE /api/modules/admin_users/delete_user
POST   /api/modules/admin_users/change_password
GET    /api/modules/admin_users/get_stats
GET    /api/modules/admin_users/health_check
```

### Examples

#### Create User
```bash
curl -X POST http://localhost:8000/api/modules/admin_users/create_user \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "password": "SecurePass123",
    "email": "john@example.com",
    "full_name": "John Doe",
    "roles": ["user", "developer"],
    "is_active": true
  }'
```

**Response**:
```json
{
  "success": true,
  "module": "admin_users",
  "operation": "create_user",
  "result": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "username": "john_doe",
    "email": "john@example.com",
    "full_name": "John Doe",
    "roles": ["user", "developer"],
    "tenant_id": null,
    "is_active": true,
    "mfa_enabled": false,
    "created_at": "2025-12-25T22:10:00.000000",
    "updated_at": "2025-12-25T22:10:00.000000",
    "request_id": "123e4567-e89b-12d3-a456-426614174000"
  }
}
```

#### List Users (with filtering)
```bash
curl -X POST http://localhost:8000/api/modules/admin_users/list_users \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {
      "is_active": true,
      "roles": ["admin"]
    },
    "limit": 10,
    "offset": 0
  }'
```

#### Update User
```bash
curl -X POST http://localhost:8000/api/modules/admin_users/update_user \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "john.doe@example.com",
    "roles": ["user", "developer", "admin"],
    "is_active": true
  }'
```

#### Change Password
```bash
curl -X POST http://localhost:8000/api/modules/admin_users/change_password \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "old_password": "SecurePass123",
    "new_password": "NewSecurePass456"
  }'
```

#### Get Statistics
```bash
curl http://localhost:8000/api/modules/admin_users/get_stats
```

**Response**:
```json
{
  "success": true,
  "module": "admin_users",
  "operation": "get_stats",
  "result": {
    "module": "admin_users",
    "total_users": 150,
    "active_users": 142,
    "inactive_users": 8,
    "users_by_role": {
      "user": 120,
      "admin": 25,
      "developer": 45
    },
    "users_by_tenant": {
      "default": 100,
      "tenant_1": 30,
      "tenant_2": 20
    },
    "storage": "redis",
    "security": {
      "password_hashing": "bcrypt",
      "username_index": "O(1)"
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
      "users_index": "ubp:admin:users",
      "user_prefix": "ubp:admin:user:",
      "username_index_prefix": "ubp:admin:username_index:"
    }
  },
  "security": {
    "password_hashing": {
      "algorithm": "bcrypt",
      "rounds": 12
    },
    "password_policy": {
      "min_length": 6,
      "require_uppercase": false,
      "require_lowercase": false,
      "require_numbers": false,
      "require_special": false
    },
    "mfa_enabled": false
  },
  "defaults": {
    "roles": [],
    "is_active": true,
    "mfa_enabled": false
  },
  "validation": {
    "username": {
      "min_length": 3,
      "max_length": 50,
      "pattern": "^[a-zA-Z0-9_-]+$"
    },
    "email": {
      "max_length": 100
    },
    "full_name": {
      "max_length": 100
    }
  }
}
```

### Customization

#### Increase Password Strength
```json
{
  "security": {
    "password_hashing": {
      "rounds": 14
    },
    "password_policy": {
      "min_length": 12,
      "require_uppercase": true,
      "require_numbers": true,
      "require_special": true
    }
  }
}
```

#### Enable MFA
```json
{
  "security": {
    "mfa_enabled": true
  }
}
```

## Events

### Published Events

The module publishes these events to the event bus:

#### admin.user.created
```json
{
  "user_id": "uuid",
  "username": "string",
  "roles": ["role1", "role2"],
  "tenant_id": "string|null",
  "timestamp": "ISO datetime",
  "request_id": "uuid"
}
```

#### admin.user.updated
```json
{
  "user_id": "uuid",
  "username": "string",
  "changes": {
    "username": false,
    "roles": true,
    "is_active": false
  },
  "timestamp": "ISO datetime",
  "request_id": "uuid"
}
```

#### admin.user.deleted
```json
{
  "user_id": "uuid",
  "username": "string",
  "timestamp": "ISO datetime",
  "request_id": "uuid"
}
```

#### admin.user.password_changed
```json
{
  "user_id": "uuid",
  "username": "string",
  "timestamp": "ISO datetime",
  "request_id": "uuid"
}
```

### Event Subscribers

Other modules can subscribe to these events:

```python
async def on_user_created(event: Event):
    """React to new user creation."""
    user_id = event.payload["user_id"]
    username = event.payload["username"]
    # ... handle event
```

## Security Best Practices

### 1. Never Return password_hash
All operations automatically exclude `password_hash` from responses.

### 2. Verify Old Password on Change
`change_password` requires old password verification to prevent unauthorized changes.

### 3. Username Index for O(1) Lookup
Username uniqueness checked in O(1) time using Redis index, preventing enumeration attacks.

### 4. Bcrypt Rounds
Default 12 rounds provides strong security. Increase to 14+ for higher security environments.

### 5. Request Tracking
All operations include `request_id` for audit trails and debugging.

## Performance

### Metrics
- **create_user**: ~50ms (bcrypt hashing + 3 Redis operations)
- **list_users**: ~10ms (single Redis HGETALL)
- **get_user**: ~2ms (single Redis GET)
- **update_user**: ~15ms (2-3 Redis operations)
- **delete_user**: ~10ms (3 Redis DEL operations)
- **change_password**: ~55ms (bcrypt verify + hash + 2 Redis operations)

### Optimization
- Username index provides O(1) uniqueness checking
- Redis operations pipelined where possible
- Password hashing async-friendly (non-blocking)

## Testing

### Health Check
```bash
curl http://localhost:8000/api/modules/admin_users/health_check
```

Expected response:
```json
{
  "success": true,
  "module": "admin_users",
  "operation": "health_check",
  "result": {
    "module": "admin_users",
    "status": "healthy",
    "initialized": true,
    "redis": {
      "connected": true,
      "status": "healthy"
    }
  }
}
```

### Full CRUD Test
```bash
# 1. Create user
USER_ID=$(curl -X POST http://localhost:8000/api/modules/admin_users/create_user \
  -d '{"username":"test_user","password":"test123"}' | jq -r '.result.user_id')

# 2. Get user
curl http://localhost:8000/api/modules/admin_users/get_user \
  -d "{\"user_id\":\"$USER_ID\"}"

# 3. Update user
curl -X POST http://localhost:8000/api/modules/admin_users/update_user \
  -d "{\"user_id\":\"$USER_ID\",\"email\":\"test@example.com\"}"

# 4. Change password
curl -X POST http://localhost:8000/api/modules/admin_users/change_password \
  -d "{\"user_id\":\"$USER_ID\",\"old_password\":\"test123\",\"new_password\":\"test456\"}"

# 5. Delete user
curl -X POST http://localhost:8000/api/modules/admin_users/delete_user \
  -d "{\"user_id\":\"$USER_ID\"}"
```

## Troubleshooting

### Error: "passlib[bcrypt] not installed"
**Solution**: Install dependency
```bash
pip install passlib[bcrypt]
```

### Error: "Redis client not available"
**Solution**: Check Redis connection in `config.py` and ensure Redis is running
```bash
redis-cli ping  # Should return PONG
```

### Error: "Username already exists"
**Solution**: Choose different username or delete existing user

### Error: "Invalid current password"
**Solution**: Verify old password is correct when changing password

## Version History

### 1.2.3 (2026-01-08)
- **Personal KB Ingestion (GAP-INGEST-002)**:
  - `ingest_to_personal_kb`: Ingest document to user's Personal KB with ACL verification
- **Cross-module Integration (GAP-003/004 support)**:
  - `get_user_internal`: Internal method for cross-module user lookup
  - Bypasses security context - for module-to-module calls only
  - Used by `admin_clients` to verify user belongs to client

### 1.0.0 (2025-12-25)
- Initial release
- bcrypt password hashing
- O(1) username lookup
- Event bus integration
- Multi-tenancy support
- CRUD operations
- Statistics and health monitoring

## License

UBP Enterprise Hybrid Edition - Proprietary License

## Author

UBP Team - Universal Backend Platform

## Support

For issues or questions, please contact the UBP development team.
