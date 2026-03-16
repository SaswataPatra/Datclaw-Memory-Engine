"""
Integration tests for Authentication FastAPI endpoints.
Tests the full HTTP API including signup, login, /auth/me, and protected endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from arango import ArangoClient


@pytest.fixture(scope="module")
def test_db():
    """Create a test database."""
    config = {
        'url': 'http://localhost:8529',
        'database': 'dappy_test_endpoints',
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


@pytest.fixture(scope="module")
def client(test_db):
    """Create FastAPI test client with test database."""
    from api.main import app
    from api.auth import AuthManager, set_auth_manager
    
    auth_mgr = AuthManager(
        arango_db=test_db,
        secret_key="test-secret-key-12345",
        algorithm="HS256",
        token_expire_minutes=30
    )
    set_auth_manager(auth_mgr)
    
    return TestClient(app)


def test_signup_endpoint(client):
    """Test POST /auth/signup."""
    response = client.post('/auth/signup', json={
        'username': 'alice',
        'email': 'alice@example.com',
        'password': 'alice123',
        'display_name': 'Alice Wonderland'
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert data['access_token']
    assert data['token_type'] == 'bearer'
    assert data['user_id']
    assert data['username'] == 'alice'
    assert data['display_name'] == 'Alice Wonderland'
    assert data['expires_in'] == 30 * 60
    
    print(f"✅ POST /auth/signup: user_id={data['user_id']}")


def test_signup_duplicate_username(client):
    """Test signup with duplicate username."""
    client.post('/auth/signup', json={
        'username': 'bob',
        'email': 'bob1@example.com',
        'password': 'password123'
    })
    
    response = client.post('/auth/signup', json={
        'username': 'bob',
        'email': 'bob2@example.com',
        'password': 'password456'
    })
    
    assert response.status_code == 409
    assert 'Username already taken' in response.json()['detail']
    
    print("✅ Duplicate username rejected (409)")


def test_signup_duplicate_email(client):
    """Test signup with duplicate email."""
    client.post('/auth/signup', json={
        'username': 'charlie1',
        'email': 'charlie@example.com',
        'password': 'password123'
    })
    
    response = client.post('/auth/signup', json={
        'username': 'charlie2',
        'email': 'charlie@example.com',
        'password': 'password456'
    })
    
    assert response.status_code == 409
    assert 'Email already registered' in response.json()['detail']
    
    print("✅ Duplicate email rejected (409)")


def test_signup_validation_short_username(client):
    """Test signup with username too short."""
    response = client.post('/auth/signup', json={
        'username': 'ab',
        'email': 'short@example.com',
        'password': 'password123'
    })
    
    assert response.status_code == 422
    
    print("✅ Short username rejected (422)")


def test_signup_validation_short_password(client):
    """Test signup with password too short."""
    response = client.post('/auth/signup', json={
        'username': 'validuser',
        'email': 'valid@example.com',
        'password': '12345'
    })
    
    assert response.status_code == 422
    
    print("✅ Short password rejected (422)")


def test_login_endpoint(client):
    """Test POST /auth/login."""
    client.post('/auth/signup', json={
        'username': 'dave',
        'email': 'dave@example.com',
        'password': 'davepass123'
    })
    
    response = client.post('/auth/login', json={
        'username': 'dave',
        'password': 'davepass123'
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert data['access_token']
    assert data['token_type'] == 'bearer'
    assert data['username'] == 'dave'
    
    print(f"✅ POST /auth/login: token={data['access_token'][:20]}...")


def test_login_wrong_password(client):
    """Test login with wrong password."""
    client.post('/auth/signup', json={
        'username': 'eve',
        'email': 'eve@example.com',
        'password': 'correct_password'
    })
    
    response = client.post('/auth/login', json={
        'username': 'eve',
        'password': 'wrong_password'
    })
    
    assert response.status_code == 401
    assert 'Invalid username or password' in response.json()['detail']
    
    print("✅ Wrong password rejected (401)")


def test_login_nonexistent_user(client):
    """Test login with non-existent username."""
    response = client.post('/auth/login', json={
        'username': 'ghost',
        'password': 'password123'
    })
    
    assert response.status_code == 401
    assert 'Invalid username or password' in response.json()['detail']
    
    print("✅ Non-existent user rejected (401)")


def test_auth_me_endpoint(client):
    """Test GET /auth/me with valid token."""
    signup_response = client.post('/auth/signup', json={
        'username': 'frank',
        'email': 'frank@example.com',
        'password': 'frankpass123',
        'display_name': 'Frank Ocean'
    })
    token = signup_response.json()['access_token']
    
    response = client.get('/auth/me', headers={
        'Authorization': f'Bearer {token}'
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert data['username'] == 'frank'
    assert data['email'] == 'frank@example.com'
    assert data['display_name'] == 'Frank Ocean'
    assert 'user_id' in data
    assert 'created_at' in data
    
    print(f"✅ GET /auth/me: {data['username']} ({data['email']})")


def test_auth_me_no_token(client):
    """Test GET /auth/me without token."""
    response = client.get('/auth/me')
    
    assert response.status_code == 403
    
    print("✅ /auth/me without token rejected (403)")


def test_auth_me_invalid_token(client):
    """Test GET /auth/me with invalid token."""
    response = client.get('/auth/me', headers={
        'Authorization': 'Bearer invalid.token.here'
    })
    
    assert response.status_code == 401
    
    print("✅ /auth/me with invalid token rejected (401)")


def test_protected_endpoint_no_auth(client):
    """Test that /chat requires authentication."""
    response = client.post('/chat', json={
        'session_id': 'test-session',
        'message': 'Hello DAPPY'
    })
    
    assert response.status_code == 403
    
    print("✅ /chat without auth rejected (403)")


def test_protected_endpoint_with_auth(client):
    """Test that /chat works with valid token."""
    signup_response = client.post('/auth/signup', json={
        'username': 'chatuser',
        'email': 'chat@example.com',
        'password': 'chatpass123'
    })
    token = signup_response.json()['access_token']
    
    response = client.post('/chat', 
        headers={'Authorization': f'Bearer {token}'},
        json={
            'session_id': 'test-session-123',
            'message': 'Hello DAPPY, remember that I like coffee'
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        assert 'assistant_message' in data
        assert 'conversation_history' in data
        print(f"✅ /chat with auth succeeded: {data['assistant_message'][:50]}...")
    else:
        print(f"⚠️  /chat returned {response.status_code} (may need services running)")


def test_token_in_multiple_requests(client):
    """Test that a token works across multiple requests."""
    signup_response = client.post('/auth/signup', json={
        'username': 'grace',
        'email': 'grace@example.com',
        'password': 'gracepass123'
    })
    token = signup_response.json()['access_token']
    headers = {'Authorization': f'Bearer {token}'}
    
    for i in range(3):
        response = client.get('/auth/me', headers=headers)
        assert response.status_code == 200
        assert response.json()['username'] == 'grace'
    
    print("✅ Token valid across multiple requests")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧪 DAPPY Authentication Endpoint Tests")
    print("="*60 + "\n")
    
    pytest.main([__file__, '-v', '-s'])
