"""
Unit tests for Authentication API
Tests signup, login, token validation, and protected endpoints.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from arango import ArangoClient
from api.auth import AuthManager, SignupRequest, LoginRequest
from fastapi import HTTPException


@pytest.fixture
def arango_db():
    """Create test database connection."""
    config = {
        'url': 'http://localhost:8529',
        'database': 'dappy_test_auth',
        'username': 'root',
        'password': 'dappy_dev_password'
    }
    
    client = ArangoClient(hosts=config['url'])
    sys_db = client.db('_system', username=config['username'], password=config['password'])
    
    if sys_db.has_database(config['database']):
        sys_db.delete_database(config['database'])
    
    sys_db.create_database(config['database'])
    
    db = client.db(
        config['database'],
        username=config['username'],
        password=config['password']
    )
    
    yield db
    
    sys_db.delete_database(config['database'])


@pytest.fixture
def auth_manager(arango_db):
    """Create AuthManager instance with test database."""
    return AuthManager(
        arango_db=arango_db,
        secret_key="test-secret-key-12345",
        algorithm="HS256",
        token_expire_minutes=30
    )


def test_collection_creation(auth_manager, arango_db):
    """Test that users collection is created with proper indexes."""
    assert arango_db.has_collection('users')
    
    col = arango_db.collection('users')
    indexes = col.indexes()
    
    index_fields = [idx['fields'] for idx in indexes]
    assert ['username'] in index_fields
    assert ['email'] in index_fields
    
    print("✅ Users collection created with unique indexes")


def test_signup_success(auth_manager):
    """Test successful user signup."""
    req = SignupRequest(
        username="testuser",
        email="test@example.com",
        password="password123",
        display_name="Test User"
    )
    
    result = auth_manager.signup(req)
    
    assert result['access_token']
    assert result['token_type'] == 'bearer'
    assert result['user_id']
    assert result['username'] == 'testuser'
    assert result['display_name'] == 'Test User'
    assert result['expires_in'] == 30 * 60
    
    print(f"✅ Signup successful: user_id={result['user_id']}")


def test_signup_duplicate_username(auth_manager):
    """Test that duplicate username is rejected."""
    req1 = SignupRequest(
        username="duplicate",
        email="user1@example.com",
        password="password123"
    )
    auth_manager.signup(req1)
    
    req2 = SignupRequest(
        username="duplicate",
        email="user2@example.com",
        password="password456"
    )
    
    with pytest.raises(HTTPException) as exc_info:
        auth_manager.signup(req2)
    
    assert exc_info.value.status_code == 409
    assert "Username already taken" in str(exc_info.value.detail)
    
    print("✅ Duplicate username rejected")


def test_signup_duplicate_email(auth_manager):
    """Test that duplicate email is rejected."""
    req1 = SignupRequest(
        username="user1",
        email="duplicate@example.com",
        password="password123"
    )
    auth_manager.signup(req1)
    
    req2 = SignupRequest(
        username="user2",
        email="duplicate@example.com",
        password="password456"
    )
    
    with pytest.raises(HTTPException) as exc_info:
        auth_manager.signup(req2)
    
    assert exc_info.value.status_code == 409
    assert "Email already registered" in str(exc_info.value.detail)
    
    print("✅ Duplicate email rejected")


def test_login_success(auth_manager):
    """Test successful login."""
    req_signup = SignupRequest(
        username="logintest",
        email="login@example.com",
        password="mypassword"
    )
    signup_result = auth_manager.signup(req_signup)
    
    req_login = LoginRequest(
        username="logintest",
        password="mypassword"
    )
    login_result = auth_manager.login(req_login)
    
    assert login_result['access_token']
    assert login_result['user_id'] == signup_result['user_id']
    assert login_result['username'] == 'logintest'
    
    print(f"✅ Login successful: token={login_result['access_token'][:20]}...")


def test_login_invalid_username(auth_manager):
    """Test login with non-existent username."""
    req = LoginRequest(
        username="nonexistent",
        password="password123"
    )
    
    with pytest.raises(HTTPException) as exc_info:
        auth_manager.login(req)
    
    assert exc_info.value.status_code == 401
    assert "Invalid username or password" in str(exc_info.value.detail)
    
    print("✅ Invalid username rejected")


def test_login_wrong_password(auth_manager):
    """Test login with incorrect password."""
    req_signup = SignupRequest(
        username="pwdtest",
        email="pwd@example.com",
        password="correct_password"
    )
    auth_manager.signup(req_signup)
    
    req_login = LoginRequest(
        username="pwdtest",
        password="wrong_password"
    )
    
    with pytest.raises(HTTPException) as exc_info:
        auth_manager.login(req_login)
    
    assert exc_info.value.status_code == 401
    assert "Invalid username or password" in str(exc_info.value.detail)
    
    print("✅ Wrong password rejected")


def test_token_decode_valid(auth_manager):
    """Test decoding a valid JWT token."""
    req = SignupRequest(
        username="tokentest",
        email="token@example.com",
        password="password123"
    )
    result = auth_manager.signup(req)
    token = result['access_token']
    
    payload = auth_manager.decode_token(token)
    
    assert payload['sub'] == result['user_id']
    assert payload['username'] == 'tokentest'
    assert 'exp' in payload
    assert 'iat' in payload
    
    print(f"✅ Token decoded: user_id={payload['sub']}")


def test_token_decode_invalid(auth_manager):
    """Test decoding an invalid token."""
    with pytest.raises(HTTPException) as exc_info:
        auth_manager.decode_token("invalid.token.here")
    
    assert exc_info.value.status_code == 401
    assert "Invalid or expired token" in str(exc_info.value.detail)
    
    print("✅ Invalid token rejected")


def test_token_decode_expired(auth_manager):
    """Test decoding an expired token."""
    from jose import jwt
    
    expired_payload = {
        'sub': 'test_user_id',
        'username': 'testuser',
        'exp': datetime.utcnow() - timedelta(hours=1),
        'iat': datetime.utcnow() - timedelta(hours=2),
    }
    expired_token = jwt.encode(expired_payload, auth_manager.secret_key, algorithm=auth_manager.algorithm)
    
    with pytest.raises(HTTPException) as exc_info:
        auth_manager.decode_token(expired_token)
    
    assert exc_info.value.status_code == 401
    
    print("✅ Expired token rejected")


def test_get_user_exists(auth_manager):
    """Test retrieving an existing user."""
    req = SignupRequest(
        username="gettest",
        email="get@example.com",
        password="password123",
        display_name="Get Test User"
    )
    result = auth_manager.signup(req)
    user_id = result['user_id']
    
    user = auth_manager.get_user(user_id)
    
    assert user is not None
    assert user['user_id'] == user_id
    assert user['username'] == 'gettest'
    assert user['email'] == 'get@example.com'
    assert user['display_name'] == 'Get Test User'
    assert 'created_at' in user
    
    print(f"✅ User retrieved: {user['username']}")


def test_get_user_not_found(auth_manager):
    """Test retrieving a non-existent user."""
    user = auth_manager.get_user('nonexistent_id')
    assert user is None
    
    print("✅ Non-existent user returns None")


def test_password_hashing(auth_manager):
    """Test that passwords are properly hashed."""
    req = SignupRequest(
        username="hashtest",
        email="hash@example.com",
        password="my_secret_password"
    )
    result = auth_manager.signup(req)
    
    col = auth_manager.db.collection('users')
    user_doc = col.get(result['user_id'])
    
    assert 'password_hash' in user_doc
    assert user_doc['password_hash'] != "my_secret_password"
    assert user_doc['password_hash'].startswith('$2b$')
    
    print(f"✅ Password hashed: {user_doc['password_hash'][:30]}...")


def test_display_name_default(auth_manager):
    """Test that display_name defaults to username if not provided."""
    req = SignupRequest(
        username="defaultname",
        email="default@example.com",
        password="password123"
    )
    result = auth_manager.signup(req)
    
    assert result['display_name'] == 'defaultname'
    
    print("✅ Display name defaults to username")


def test_multiple_users_independent(auth_manager):
    """Test that multiple users can be created independently."""
    users = []
    for i in range(3):
        req = SignupRequest(
            username=f"user{i}",
            email=f"user{i}@example.com",
            password=f"password{i}"
        )
        result = auth_manager.signup(req)
        users.append(result)
    
    assert len(users) == 3
    assert len(set(u['user_id'] for u in users)) == 3
    assert len(set(u['access_token'] for u in users)) == 3
    
    print(f"✅ Created {len(users)} independent users")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧪 DAPPY Authentication API Tests")
    print("="*60 + "\n")
    
    pytest.main([__file__, '-v', '-s'])
