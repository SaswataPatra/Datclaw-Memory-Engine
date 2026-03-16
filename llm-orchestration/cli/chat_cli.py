"""
DAPPY Chatbot CLI
Interactive CLI for testing DAPPY features with debug mode
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional

try:
    import httpx
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.markdown import Markdown
except ImportError:
    print("Missing dependencies. Install with: pip install httpx rich")
    sys.exit(1)

console = Console()


class DAPPYChatCLI:
    """Interactive CLI for DAPPY chatbot"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.conversation_history: List[Dict] = []
        self.message_counter = 0
        self.debug_mode = False
        
    async def start(self):
        """Start interactive CLI session"""
        console.print(Panel.fit(
            "[bold cyan]🤖 DAPPY Chatbot CLI[/bold cyan]\n"
            "[dim]Type '/help' for commands or start chatting![/dim]",
            border_style="cyan"
        ))
        
        # Initialize session
        self.user_id = "saswata"  # Fixed user ID for persistent memories
        self.session_id = f"cli_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        console.print(f"[dim]Session: {self.session_id}[/dim]\n")
        
        # Health check
        await self.health_check()
        
        # Main loop
        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ").strip()
                
                if not user_input:
                    continue
                    
                # Handle commands
                if user_input.startswith('/'):
                    await self.handle_command(user_input)
                    continue
                
                # Handle chat message
                await self.chat(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except EOFError:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
    
    async def chat(self, user_message: str):
        """Send message and get response"""
        self.message_counter += 1
        
        # Build request
        request = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "message": user_message,
            "conversation_history": self.conversation_history,
            "temperature": 0.7,
            "debug": self.debug_mode
        }
        
        # Show request in debug mode
        if self.debug_mode:
            console.print(f"\n[dim]📤 Request to {self.base_url}/chat[/dim]")
            console.print(Syntax(
                json.dumps({
                    "user_id": request["user_id"],
                    "message": request["message"],
                    "history_length": len(request["conversation_history"]),
                    "debug": request["debug"]
                }, indent=2),
                "json",
                theme="monokai",
                line_numbers=False
            ))
        
        # Make API call
        with console.status("[bold yellow]🤔 Thinking..."):
            try:
                response = await self.client.post(
                    f"{self.base_url}/chat",
                    json=request
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                console.print(f"[red]HTTP Error {e.response.status_code}:[/red] {e.response.text}")
                return
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                return
        
        # Update conversation history
        self.conversation_history = data["conversation_history"]
        
        # Display response
        self.display_response(data)
        
        # Check for pending shadow tier confirmation
        await self.check_shadow_confirmation()
    
    def display_response(self, data: Dict):
        """Display response with debug information"""
        # Show assistant response
        console.print(Panel(
            f"[bold blue]Assistant:[/bold blue]\n\n{data['assistant_message']}",
            border_style="blue",
            padding=(1, 2)
        ))
        
        # Show debug panel if enabled
        if self.debug_mode and data.get("debug"):
            metadata = data.get("metadata", {})
            debug_info = data.get("debug", {})
            
            debug_table = Table(
                title="🔍 Debug Info",
                show_header=True,
                header_style="bold magenta",
                border_style="magenta"
            )
            debug_table.add_column("Metric", style="cyan", width=25)
            debug_table.add_column("Value", style="yellow")
            
            # LLM info
            llm_meta = metadata.get("llm", {})
            debug_table.add_row("LLM Provider", llm_meta.get("provider", "N/A"))
            debug_table.add_row("Model", llm_meta.get("model", "N/A"))
            
            # Token usage
            usage = llm_meta.get("usage", {})
            debug_table.add_row("Prompt Tokens", str(usage.get("prompt_tokens", 0)))
            debug_table.add_row("Completion Tokens", str(usage.get("completion_tokens", 0)))
            debug_table.add_row("Total Tokens", str(usage.get("total_tokens", 0)))
            
            # Context info
            context_meta = metadata.get("context", {})
            debug_table.add_row("Context Tokens", str(context_meta.get("current_tokens", 0)))
            debug_table.add_row("Max Tokens", str(context_meta.get("max_tokens", 0)))
            debug_table.add_row("Usage %", f"{context_meta.get('usage_percent', 0) * 100:.2f}%")
            debug_table.add_row("Flushed", "✅" if context_meta.get("flushed") else "❌")
            debug_table.add_row("Emergency Flush", "⚠️ YES" if context_meta.get("emergency_flush") else "✅ No")
            
            # Memory extraction
            mem_extract = debug_info.get("memory_extraction", {})
            triggers = mem_extract.get("triggers_detected", [])
            debug_table.add_row("Memory Triggers", ", ".join(triggers) if triggers else "None")
            
            console.print(debug_table)
    
    async def handle_command(self, command: str):
        """Handle CLI commands"""
        cmd_parts = command.split()
        cmd = cmd_parts[0]
        
        if cmd == "/help":
            self.show_help()
        elif cmd == "/debug":
            self.debug_mode = not self.debug_mode
            status = "ON" if self.debug_mode else "OFF"
            console.print(f"[yellow]Debug mode: {status}[/yellow]")
        elif cmd == "/health":
            await self.health_check()
        elif cmd == "/memories":
            await self.show_memories()
        elif cmd == "/stats":
            await self.show_stats()
        elif cmd == "/shadow":
            await self.show_shadow_memories()
        elif cmd == "/clear":
            self.conversation_history = []
            console.print("[green]✅ Conversation history cleared[/green]")
        elif cmd == "/quit" or cmd == "/exit":
            raise KeyboardInterrupt
        else:
            console.print(f"[red]Unknown command: {cmd}. Type /help for commands[/red]")
    
    async def health_check(self):
        """Check service health and wait for initialization"""
        max_retries = 30  # 30 seconds max wait
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = await self.client.get(f"{self.base_url}/health")
                data = response.json()
                
                status = data.get("status", "unknown")
                
                # If still initializing, wait and retry
                if status == "initializing":
                    if attempt == 0:
                        console.print("[yellow]⏳ Service is initializing (loading ML models)...[/yellow]")
                    await asyncio.sleep(retry_delay)
                    continue
                
                # Service is ready, show health table
                table = Table(title="🏥 Service Health", show_header=True, border_style="green")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="yellow")
                
                table.add_row("Service", status)
                for component, comp_status in data.get("components", {}).items():
                    icon = "✅" if comp_status in ["healthy", "running", "ready"] else "❌"
                    table.add_row(component, f"{icon} {comp_status}")
                
                console.print(table)
                return
                
            except httpx.ConnectError:
                if attempt == 0:
                    console.print("[yellow]⏳ Waiting for service to start...[/yellow]")
                await asyncio.sleep(retry_delay)
            except Exception as e:
                console.print(f"[red]Health check failed: {e}[/red]")
                return
        
        # Timeout
        console.print("[red]❌ Service failed to initialize within 30 seconds[/red]")
    
    async def show_memories(self):
        """Show memories from Go service"""
        try:
            go_url = "http://localhost:8080"
            response = await self.client.get(
                f"{go_url}/api/v1/memories/user/{self.user_id}?limit=10"
            )
            
            if response.status_code == 200:
                memories = response.json()
                
                if memories:
                    table = Table(title="💾 Your Memories", show_header=True, border_style="blue")
                    table.add_column("Node ID", style="cyan", width=20)
                    table.add_column("Content", style="yellow", overflow="fold", width=40)
                    table.add_column("Ego Score", style="green", width=10)
                    
                    for mem in memories[:10]:
                        table.add_row(
                            mem.get("node_id", "")[:18] + "...",
                            mem.get("content", "")[:50] + ("..." if len(mem.get("content", "")) > 50 else ""),
                            f"{mem.get('ego_score', 0):.2f}"
                        )
                    console.print(table)
                else:
                    console.print("[yellow]No memories found yet. Share some personal info![/yellow]")
            else:
                console.print(f"[yellow]Memories not available (Go service may be down)[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to fetch memories: {e}[/red]")
    
    async def show_stats(self):
        """Show consolidation stats"""
        try:
            response = await self.client.get(f"{self.base_url}/consolidation/stats")
            data = response.json()
            
            console.print(Syntax(
                json.dumps(data, indent=2),
                "json",
                theme="monokai",
                line_numbers=False
            ))
        except Exception as e:
            console.print(f"[red]Failed to fetch stats: {e}[/red]")
    
    async def show_shadow_memories(self):
        """Show pending shadow memories"""
        try:
            response = await self.client.get(f"{self.base_url}/shadow/pending/{self.user_id}")
            data = response.json()
            
            if data.get("pending_count", 0) > 0:
                console.print(f"[yellow]Found {data['pending_count']} pending shadow memories:[/yellow]")
                for mem in data.get("pending_memories", []):
                    console.print(f"  • {mem.get('content', 'N/A')[:60]}...")
            else:
                console.print("[green]No pending shadow memories[/green]")
        except Exception as e:
            console.print(f"[red]Failed to fetch shadow memories: {e}[/red]")
    
    def show_help(self):
        """Show help menu"""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[bold]/help[/bold]              Show this help menu
[bold]/debug[/bold]             Toggle debug mode (show API calls & metadata)
[bold]/health[/bold]            Check service health
[bold]/memories[/bold]          Show your memories from Go service
[bold]/stats[/bold]             Show consolidation statistics
[bold]/shadow[/bold]            Show pending shadow memories
[bold]/clear[/bold]             Clear conversation history
[bold]/quit[/bold] or /exit     Exit CLI

[dim]Just type normally to chat with DAPPY![/dim]

[bold cyan]Examples:[/bold cyan]
  Hello, my name is Alice
  I love Python programming
  What do you remember about me?
        """
        console.print(Panel(help_text, title="Help", border_style="cyan"))
    
    async def check_shadow_confirmation(self):
        """Check for pending shadow tier confirmation and prompt user"""
        try:
            response = await self.client.get(
                f"{self.base_url}/shadow/pending/{self.session_id}"
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("has_pending"):
                return
            
            # Show clarification question
            console.print("\n" + "─" * 60)
            console.print(Panel(
                f"[bold yellow]🤔 Clarification Needed[/bold yellow]\n\n"
                f"{data['question']}\n\n"
                f"[dim]Content: {data['content'][:100]}...[/dim]\n"
                f"[dim]Ego Score: {data['ego_score']:.2f}[/dim]",
                border_style="yellow",
                padding=(1, 2)
            ))
            
            # Prompt user
            console.print("\n[bold]Should I remember this as core memory (Tier 1)?[/bold]")
            console.print("  [green]y[/green] = Yes, promote to Tier 1 (permanent)")
            console.print("  [red]n[/red] = No, store as Tier 2 (long-term)")
            console.print("  [dim]Enter[/dim] = Skip (auto-promote after 7 days)")
            
            choice = console.input("\n[bold cyan]Your choice (y/n/Enter):[/bold cyan] ").strip().lower()
            
            if choice == "":
                console.print("[dim]⏭️  Skipped - will auto-promote after 7 days[/dim]")
                return
            
            confirmed = choice == "y"
            
            # Send confirmation
            confirm_response = await self.client.post(
                f"{self.base_url}/shadow/confirm",
                json={
                    "session_id": self.session_id,
                    "user_id": self.user_id,
                    "confirmed": confirmed
                }
            )
            confirm_response.raise_for_status()
            result = confirm_response.json()
            
            if result["status"] == "approved":
                console.print(f"[green]✅ {result['message']}[/green]")
            elif result["status"] == "rejected":
                console.print(f"[yellow]📝 {result['message']}[/yellow]")
            else:
                console.print(f"[red]❌ {result['message']}[/red]")
            
            console.print("─" * 60 + "\n")
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                console.print(f"[dim red]Shadow check error: {e}[/dim red]")
        except Exception as e:
            console.print(f"[dim red]Shadow check error: {e}[/dim red]")
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


async def main():
    """Main entry point"""
    import sys
    
    # Parse arguments
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    cli = DAPPYChatCLI(base_url=base_url)
    
    try:
        await cli.start()
    finally:
        await cli.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye! 👋[/yellow]")

