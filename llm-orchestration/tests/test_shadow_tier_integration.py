#!/usr/bin/env python3
"""
Quick test script to verify shadow tier integration
"""

import asyncio
import httpx
from rich.console import Console

console = Console()


async def test_shadow_tier():
    """Test shadow tier integration"""
    
    base_url = "http://localhost:8000"
    client = httpx.AsyncClient(timeout=60.0)
    
    try:
        # 1. Health check
        console.print("[bold cyan]1. Health Check[/bold cyan]")
        response = await client.get(f"{base_url}/health")
        response.raise_for_status()
        console.print(f"   ✅ Service is healthy\n")
        
        # 2. Send a high-ego message (should trigger shadow tier)
        console.print("[bold cyan]2. Testing High-Ego Message (Shadow Tier)[/bold cyan]")
        console.print("   Sending: 'My name is Test User'\n")
        
        chat_response = await client.post(
            f"{base_url}/chat",
            json={
                "user_id": "test_user_shadow",
                "session_id": "test_session_shadow_001",
                "message": "My name is Test User and I love Python programming",
                "conversation_history": [],
                "temperature": 0.7,
                "debug": True
            }
        )
        chat_response.raise_for_status()
        chat_data = chat_response.json()
        
        console.print(f"   Assistant: {chat_data['assistant_message'][:100]}...\n")
        
        # 3. Check for pending shadow tier confirmation
        console.print("[bold cyan]3. Checking for Shadow Tier Confirmation[/bold cyan]")
        shadow_response = await client.get(
            f"{base_url}/shadow/pending/session/test_session_shadow_001"
        )
        shadow_response.raise_for_status()
        shadow_data = shadow_response.json()
        
        if shadow_data.get("has_pending"):
            console.print(f"   ✅ Shadow tier confirmation pending!")
            console.print(f"   Question: {shadow_data['question']}")
            console.print(f"   Content: {shadow_data['content'][:80]}...")
            console.print(f"   Ego Score: {shadow_data['ego_score']:.2f}\n")
            
            # 4. Approve the shadow tier memory
            console.print("[bold cyan]4. Approving Shadow Tier Memory[/bold cyan]")
            confirm_response = await client.post(
                f"{base_url}/shadow/confirm",
                json={
                    "session_id": "test_session_shadow_001",
                    "user_id": "test_user_shadow",
                    "confirmed": True
                }
            )
            confirm_response.raise_for_status()
            confirm_data = confirm_response.json()
            
            console.print(f"   Status: {confirm_data['status']}")
            console.print(f"   Message: {confirm_data['message']}")
            console.print(f"   Node ID: {confirm_data.get('node_id', 'N/A')}\n")
            
            if confirm_data['status'] == 'approved':
                console.print("[bold green]✅ SHADOW TIER INTEGRATION WORKING![/bold green]\n")
            else:
                console.print("[bold red]❌ Unexpected status[/bold red]\n")
        else:
            console.print(f"   ⚠️  No pending confirmation (might be auto-promoted or stored directly)")
            console.print(f"   Message: {shadow_data.get('message', 'N/A')}\n")
            console.print("[bold yellow]⚠️  Shadow tier might not be triggered (check ego score and confidence)[/bold yellow]\n")
        
        # 5. Test low-ego message (should NOT trigger shadow tier)
        console.print("[bold cyan]5. Testing Low-Ego Message (No Shadow Tier)[/bold cyan]")
        console.print("   Sending: 'What is 2+2?'\n")
        
        chat_response2 = await client.post(
            f"{base_url}/chat",
            json={
                "user_id": "test_user_shadow",
                "session_id": "test_session_shadow_002",
                "message": "What is 2+2?",
                "conversation_history": [],
                "temperature": 0.7
            }
        )
        chat_response2.raise_for_status()
        
        shadow_response2 = await client.get(
            f"{base_url}/shadow/pending/session/test_session_shadow_002"
        )
        shadow_response2.raise_for_status()
        shadow_data2 = shadow_response2.json()
        
        if not shadow_data2.get("has_pending"):
            console.print(f"   ✅ No shadow tier confirmation (as expected for low-ego message)\n")
        else:
            console.print(f"   ⚠️  Unexpected shadow tier confirmation for low-ego message\n")
        
        console.print("[bold green]✅ All tests completed![/bold green]")
        
    except httpx.HTTPStatusError as e:
        console.print(f"[bold red]❌ HTTP Error {e.response.status_code}:[/bold red]")
        console.print(f"   {e.response.text}")
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
    finally:
        await client.aclose()


if __name__ == "__main__":
    console.print("\n[bold]Testing Shadow Tier Integration[/bold]\n")
    console.print("Make sure the service is running: ./start_service.sh\n")
    
    try:
        asyncio.run(test_shadow_tier())
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted[/yellow]")

