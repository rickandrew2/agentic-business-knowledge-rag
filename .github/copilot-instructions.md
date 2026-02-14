# GitHub Copilot Instructions

This project follows enterprise-grade, technology-agnostic development patterns.

## Quick Start

- **Project Guidelines**: See `CLAUDE.md`
- **Agent Patterns**: See `AGENTS.md`
- **Detailed Rules**: See `agents/rules/*.mdc` files

## Always Apply

1. **Security-first**: Validate inputs, authenticate endpoints, rate limit public APIs
   - Reference: `agents/rules/security.mdc`

2. **Testing**: Unit (80%), Integration (15%), E2E (5%) coverage
   - Reference: `agents/rules/testing.mdc`

3. **Type Safety**: Strong typing with runtime validation at boundaries
   - Reference: `agents/rules/core.mdc`

## Agent Delegation

When implementing features, follow agent patterns from `AGENTS.md`:
- **UI/Design** → FrontendAgent patterns (`agents/rules/frontend.mdc`)
- **API/Logic** → BackendAgent patterns (`agents/rules/security.mdc`)
- **Database** → DatabaseAgent patterns (`agents/rules/database.mdc`)
- **Testing** → TestAgent patterns (`agents/rules/testing.mdc`)
- **Security** → SecurityAgent patterns (`agents/rules/security.mdc`)

## Critical Rules

- Validate ALL user inputs with schema validation
- Authenticate and authorize protected endpoints
- Rate limit public endpoints
- Write tests for all business logic
- Ensure WCAG 2.1 AA accessibility compliance
- Use ORM/ODM patterns, avoid raw queries
- Never expose sensitive data in errors/logs

## Reference Files

- `CLAUDE.md` - Architecture and technology stack guidance
- `AGENTS.md` - Agent responsibilities and delegation
- `agents/rules/core.mdc` - Core principles
- `agents/rules/security.mdc` - Security patterns (CRITICAL)
- `agents/rules/testing.mdc` - Testing strategy (CRITICAL)
- `agents/rules/frontend.mdc` - Frontend patterns
- `agents/rules/database.mdc` - Database patterns
- `agents/rules/style.mdc` - Code style guidelines

---

**Note**: This is technology-agnostic. Adapt patterns to your chosen stack while maintaining security and quality standards.
