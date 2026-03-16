# Contributing to Datclaw Memory Engine

Thank you for your interest in contributing to Datclaw Memory Engine! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Datclaw-Memory-Engine.git
cd Datclaw-Memory-Engine

# Add upstream remote
git remote add upstream https://github.com/SaswataPatra/Datclaw-Memory-Engine.git
```

### 2. Set Up Development Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your development credentials
nano .env

# Start infrastructure
docker-compose up -d

# Set up Python environment
cd llm-orchestration
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests to verify setup
pytest tests/ -v
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b fix/bug-description
```

## 📝 Development Guidelines

### Code Style

#### Python

- Follow [PEP 8](https://pep8.org/)
- Use type hints
- Write docstrings for functions and classes
- Maximum line length: 100 characters

```python
def process_memory(
    memory: Memory,
    user_id: str,
    ego_score: float = 0.5
) -> ProcessedMemory:
    """
    Process a memory and calculate importance score.
    
    Args:
        memory: The memory object to process
        user_id: User identifier
        ego_score: Importance score (0-1)
    
    Returns:
        ProcessedMemory with calculated scores
    """
    pass
```

#### TypeScript/React

- Use TypeScript for type safety
- Follow React best practices
- Use functional components and hooks
- Use meaningful variable names

```typescript
interface ChatMessage {
  id: string;
  content: string;
  timestamp: Date;
  userId: string;
}

const ChatComponent: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  // ...
}
```

### Testing

- Write tests for new features
- Maintain test coverage above 80%
- Run tests before submitting PR

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/graph/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new ego scoring component
fix: resolve entity extraction bug
docs: update deployment guide
test: add tests for relation classifier
refactor: simplify consolidation logic
chore: update dependencies
```

## 🔒 Security Guidelines

### Never Commit Secrets

- API keys, passwords, tokens
- Production configuration files
- Private keys or certificates

### Files That Should NEVER Be Committed

- `.env` (use `.env.example` instead)
- `docker-compose.prod.yml` (use `.example` template)
- Any file containing real credentials

### Before Committing

```bash
# Check what you're committing
git diff --staged

# Search for potential secrets
git grep -i "password\|api_key\|secret\|token" --staged

# If you accidentally staged secrets
git reset HEAD <file>
```

## 📋 Pull Request Process

### 1. Update Your Branch

```bash
# Fetch latest changes from upstream
git fetch upstream
git rebase upstream/main
```

### 2. Run Tests and Linting

```bash
# Python tests
cd llm-orchestration
pytest tests/ -v

# Python linting
flake8 .
black . --check
mypy .

# Frontend tests (if applicable)
cd frontend
npm test
npm run lint
```

### 3. Create Pull Request

1. Push your branch to your fork
2. Go to GitHub and create a Pull Request
3. Fill out the PR template
4. Link related issues

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new features
- [ ] Updated documentation

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No secrets committed
```

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Verify it's reproducible
3. Test on latest version

### Bug Report Template

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What should happen

**Actual behavior**
What actually happens

**Environment**
- OS: [e.g., macOS 13.0]
- Python version: [e.g., 3.9.7]
- Docker version: [e.g., 20.10.8]

**Logs**
```
Paste relevant logs here
```

**Additional context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
Clear description of the problem

**Describe the solution you'd like**
What you want to happen

**Describe alternatives you've considered**
Other solutions you've thought about

**Additional context**
Any other relevant information
```

## 📚 Documentation

### When to Update Docs

- Adding new features
- Changing APIs
- Updating configuration
- Adding new dependencies

### Documentation Files

- `README.md` - Project overview and quick start
- `QUICK_START.md` - Fast setup guide
- `STARTUP_GUIDE.md` - Detailed setup instructions
- `DEPLOYMENT.md` - Production deployment
- `docs/` - Technical documentation

## 🎯 Areas for Contribution

### High Priority

- [ ] Performance optimizations
- [ ] Additional test coverage
- [ ] Documentation improvements
- [ ] Bug fixes

### Feature Ideas

- [ ] Additional LLM provider support
- [ ] Enhanced entity resolution
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Export/import functionality

### Good First Issues

Look for issues labeled `good first issue` on GitHub.

## 🤝 Code Review Process

### What We Look For

1. **Correctness**: Does it work as intended?
2. **Tests**: Are there adequate tests?
3. **Style**: Does it follow project conventions?
4. **Documentation**: Is it well-documented?
5. **Security**: Are there security concerns?

### Review Timeline

- Initial review: Within 3-5 days
- Follow-up reviews: Within 2 days
- Merge: After approval and CI passes

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code review and feedback

### Questions?

- Check existing documentation
- Search closed issues
- Ask in GitHub Discussions
- Tag maintainers in PR/issue

## 🏆 Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Credited in commits

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Datclaw Memory Engine! 🎉**
