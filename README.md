# Technology-Agnostic Development Template

A flexible, enterprise-grade development template that adapts to any modern technology stack while maintaining security-first development patterns, comprehensive testing, and excellent developer experience. Built with AI assistant integration for Cursor, GitHub Copilot, and other AI coding tools.

## Overview

This template provides a proven agent-based development structure that works with any technology stack. It includes comprehensive rules, patterns, and guidelines that help both developers and AI assistants maintain consistent, secure, and high-quality code across different projects.

## Key Features

- **Technology-Agnostic**: Works with React, Vue, Angular, Node.js, Python, Go, Rust, or any modern stack
- **AI Assistant Ready**: Pre-configured for Cursor, GitHub Copilot, and other AI coding tools
- **Security-First**: Built-in OWASP Top 10 protection patterns and security guidelines
- **Comprehensive Testing**: Testing strategy with unit, integration, and E2E coverage targets
- **Agent-Based Architecture**: Specialized agents for frontend, backend, database, testing, and security
- **Type-Safe Patterns**: Strong typing and validation patterns across the stack
- **Accessibility Compliant**: WCAG 2.1 AA standards built into development patterns

## Quick Start

### 1. Clone the Template

```bash
git clone <this-repo> my-project
cd my-project
```

### 2. Choose Your Technology Stack

This template adapts to your preferred technologies:

**Frontend Options:**
- React/Next.js, Vue/Nuxt, Angular, Svelte/SvelteKit, or traditional SSR

**Backend Options:**
- Node.js (Express/Fastify), Python (Django/FastAPI), Go, Rust, Java/Spring

**Database Options:**
- PostgreSQL/MySQL with ORM, MongoDB/NoSQL, or cloud solutions like Supabase

### 3. Configure AI Assistants

The template includes configuration files for multiple AI assistants:

- **Cursor**: Uses `.cursorrules` and `agents/rules/*.mdc` files (auto-loaded)
- **GitHub Copilot**: Uses `.github/copilot-instructions.md` (auto-loaded)
- **Other AI Tools**: Reference `AI_INSTRUCTIONS.md` in your prompts

### 4. Initialize Your Project

Set up your chosen technologies:

```bash
# Example for Node.js + React
npm init -y
npm install react next typescript

# Example for Python + Django
pip install django djangorestframework
django-admin startproject myproject .

# Example for Go + Gin
go mod init myproject
go get github.com/gin-gonic/gin
```

## Directory Structure

```
├── agents/                          # Agent rules and skills
│   ├── rules/                      # Development rules and patterns
│   │   ├── core.mdc               # Core architecture guidelines
│   │   ├── security.mdc           # Security patterns and requirements
│   │   ├── testing.mdc            # Testing strategy and patterns
│   │   ├── frontend.mdc           # Frontend development patterns
│   │   ├── database.mdc           # Database design and patterns
│   │   └── style.mdc              # Code style and formatting rules
│   └── skills/                     # Reusable agent skills
│       ├── web-design-guidelines/  # Web interface guidelines skill
│       └── find-skills/           # Skill discovery utilities
├── .github/
│   └── copilot-instructions.md    # GitHub Copilot configuration
├── AGENTS.md                       # Agent responsibilities and usage guide
├── CLAUDE.md                       # Project guidelines and architecture
├── AI_INSTRUCTIONS.md              # Universal AI assistant instructions
├── .cursorrules                    # Cursor AI assistant configuration
├── .aiderignore                    # Aider AI assistant ignore patterns
├── .editorconfig                   # Editor configuration
├── .gitignore                      # Git ignore patterns
└── README.md                       # This file
```

## Core Configuration Files

### Documentation Files

- **`CLAUDE.md`** - Overall project guidelines, architecture principles, and technology stack selection
- **`AGENTS.md`** - Agent delegation patterns, responsibilities, and when to use each agent
- **`AI_INSTRUCTIONS.md`** - Universal instructions for any AI assistant (reference explicitly)

### AI Assistant Configuration

- **`.cursorrules`** - Cursor-specific rules (auto-loaded by Cursor)
- **`.github/copilot-instructions.md`** - GitHub Copilot instructions (auto-loaded)
- **`agents/rules/*.mdc`** - Detailed rules with `alwaysApply: true` for auto-loading in Cursor

### Rule Files

- **`agents/rules/core.mdc`** - Core architecture principles and best practices
- **`agents/rules/security.mdc`** - Security patterns, OWASP Top 10 protection
- **`agents/rules/testing.mdc`** - Testing strategy, coverage targets, patterns
- **`agents/rules/frontend.mdc`** - Frontend development patterns and accessibility
- **`agents/rules/database.mdc`** - Database schema design and query optimization
- **`agents/rules/style.mdc`** - Code style guidelines and formatting rules

## Agent-Based Development

This template uses specialized agents for different aspects of development:

### FrontendAgent
Handles UI/UX development, component creation, design system implementation, and accessibility compliance.

### BackendAgent
Manages API development, business logic, authentication, authorization, rate limiting, and security middleware.

### DatabaseAgent
Owns database schema design, migrations, query optimization, and data access patterns.

### TestAgent
Implements comprehensive testing strategy across unit, integration, E2E, accessibility, and security testing.

### SecurityAgent
Ensures security-first development with input validation, authentication patterns, authorization checks, and OWASP compliance.

### ReviewerAgent
Performs code reviews focusing on correctness, security, performance, test coverage, and accessibility.

## Core Principles

### Security-First Development
- Validate all inputs at application boundaries with schema validation
- Authenticate and authorize every protected endpoint
- Rate limit public endpoints to prevent abuse
- Sanitize outputs to prevent injection attacks
- Never expose sensitive data in error messages or logs

### Testing Strategy
- **Unit Tests**: 80% coverage for business logic
- **Integration Tests**: 15% coverage for API endpoints and database operations
- **E2E Tests**: 5% coverage for critical user journeys
- **Accessibility Tests**: WCAG 2.1 AA compliance for all UI

### Type Safety
- Use strong typing systems available in your chosen language
- Implement runtime validation for all external/user-provided data
- Validate at boundaries: API endpoints, form submissions, configuration
- Generate types from schemas when possible (OpenAPI, GraphQL, database schemas)

### Code Quality
- Maintain consistent patterns throughout the codebase
- Follow established code style guidelines
- Ensure proper error handling and logging
- Keep documentation updated with code changes

## Security Patterns

### Input Validation Example

```typescript
// TypeScript with Zod
import { z } from 'zod'

const userSchema = z.object({
  email: z.string().email().toLowerCase(),
  password: z.string().min(8)
})
```

```python
# Python with Pydantic
from pydantic import BaseModel, EmailStr

class User(BaseModel):
    email: EmailStr
    password: str
```

### Rate Limiting Example

```javascript
// Node.js with express-rate-limit
const rateLimit = require('express-rate-limit')

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 attempts per window
  message: 'Too many authentication attempts'
})
```

## Testing Strategy

### Coverage Targets
- **Unit Tests**: 80% coverage for business logic
- **Integration Tests**: 15% coverage for API endpoints and database operations
- **E2E Tests**: 5% coverage for critical user journeys

### Testing Tools by Technology

**JavaScript/TypeScript:**
- Unit/Integration: Jest, Vitest, or framework-specific test runners
- E2E: Playwright, Cypress, or Selenium
- Component: Testing Library, Enzyme

**Python:**
- Unit/Integration: pytest, unittest
- E2E: Selenium, Playwright
- API: requests, httpx

**Go:**
- Unit/Integration: Built-in testing package, testify
- E2E: Selenium, Playwright
- HTTP: httptest package

## AI Assistant Integration

### Cursor
Rules are automatically loaded from:
- `.cursorrules` file
- `agents/rules/*.mdc` files with `alwaysApply: true`
- `AGENTS.md` and `CLAUDE.md` (via workspace rules)

### GitHub Copilot
Instructions are automatically loaded from:
- `.github/copilot-instructions.md`

### Other AI Assistants
Reference `AI_INSTRUCTIONS.md` explicitly in your prompts:
- "Follow the patterns in `AI_INSTRUCTIONS.md`"
- "Check `AGENTS.md` for agent delegation"
- "Apply security patterns from `agents/rules/security.mdc`"

## Usage Examples

### Frontend Development
```
"Create a responsive navigation component with accessibility support"
"Redesign the user profile page following our design system"
"Implement the login form with proper validation"
```

### Backend Development
```
"Create a secure user registration endpoint with rate limiting"
"Implement JWT authentication middleware"
"Add password reset functionality with email verification"
```

### Database Operations
```
"Design a user roles and permissions schema"
"Create a migration to add audit logging"
"Optimize the user query to prevent N+1 problems"
```

## Documentation

### Getting Started
1. Review `CLAUDE.md` for architecture and technology selection guidance
2. Check `AGENTS.md` for agent responsibilities and delegation patterns
3. Reference `agents/rules/*.mdc` files for detailed implementation patterns
4. Configure your AI assistant using the appropriate configuration file

### Additional Resources
- **Architecture**: See `CLAUDE.md` for detailed architecture guidelines
- **Agent Usage**: See `AGENTS.md` for detailed agent responsibilities
- **Security**: See `agents/rules/security.mdc` for security patterns
- **Testing**: See `agents/rules/testing.mdc` for testing strategy
- **Code Style**: See `agents/rules/style.mdc` for formatting rules

## Quality Gates

All code must meet these standards:
- Pass linting and formatting checks
- Meet minimum test coverage thresholds (80% unit, 15% integration, 5% E2E)
- Pass security scans without high-severity issues
- Pass accessibility tests for user-facing features
- Follow established patterns and conventions

## Contributing

When contributing to this template:
1. Maintain technology-agnostic patterns
2. Update relevant rule files in `agents/rules/`
3. Keep documentation synchronized with code changes
4. Follow security and testing patterns
5. Ensure AI assistant configurations remain compatible

---

**Ready to start building?** Choose your technology stack, adapt the configuration files, and begin development with enterprise-grade patterns from day one.
