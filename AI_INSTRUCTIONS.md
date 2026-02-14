# AI Assistant Instructions

This project follows enterprise-grade, technology-agnostic development patterns. Always reference these files when providing assistance.

## Primary Documentation (Always Reference)

- **`CLAUDE.md`** - Overall project guidelines, architecture principles, and technology stack selection
- **`AGENTS.md`** - Agent delegation patterns, responsibilities, and when to use each agent

## Detailed Rules (Reference When Relevant)

When implementing features, always check the relevant rule file:

- **`agents/rules/core.mdc`** - Core architecture principles, security-first development, performance patterns
- **`agents/rules/security.mdc`** - Security patterns, OWASP Top 10 protection, authentication/authorization (ALWAYS APPLY)
- **`agents/rules/testing.mdc`** - Testing strategy, coverage targets, testing patterns (ALWAYS APPLY)
- **`agents/rules/frontend.mdc`** - Frontend development patterns, component architecture, accessibility
- **`agents/rules/database.mdc`** - Database schema design, migrations, query optimization
- **`agents/rules/style.mdc`** - Code style guidelines, formatting rules, naming conventions

## Usage Workflow

When implementing any feature:

1. **Check `AGENTS.md`** to determine which agent pattern applies (FrontendAgent, BackendAgent, DatabaseAgent, etc.)
2. **Reference `CLAUDE.md`** for overall architecture and technology stack guidance
3. **Apply relevant rules** from `agents/rules/*.mdc` files based on the task
4. **Always apply security patterns** from `agents/rules/security.mdc`
5. **Always apply testing patterns** from `agents/rules/testing.mdc`

## Critical Rules (Always Apply)

### Security-First Development
- **Validate all inputs** at application boundaries with schema validation
- **Authenticate and authorize** every protected endpoint
- **Rate limit** public endpoints to prevent abuse
- **Sanitize outputs** to prevent injection attacks
- **Never expose sensitive data** in error messages or logs

### Testing Requirements
- **Unit tests**: 80% coverage for business logic
- **Integration tests**: 15% coverage for API endpoints and database operations
- **E2E tests**: 5% coverage for critical user journeys
- **Accessibility tests**: WCAG 2.1 AA compliance for all UI

### Code Quality
- **Type safety**: Use strong typing, validate at boundaries
- **Performance**: Monitor bundle/binary size, implement lazy loading
- **Accessibility**: WCAG 2.1 AA compliance for user-facing components
- **Documentation**: Keep README, CLAUDE.md, and AGENTS.md updated

## Architecture Principles

- **Technology-agnostic**: Adapt patterns to chosen technology stack
- **Security-first**: Always consider security implications first
- **Feature-oriented**: Group by domain/feature, not just technical layer
- **Type-safe**: Strong typing with runtime validation at boundaries
- **Test-driven**: Comprehensive testing at all levels

## Quick Reference

### For UI/Design Work
→ Reference `AGENTS.md` → FrontendAgent section
→ Apply `agents/rules/frontend.mdc` patterns
→ Ensure accessibility compliance

### For API/Business Logic
→ Reference `AGENTS.md` → BackendAgent section
→ Apply `agents/rules/security.mdc` for authentication/validation
→ Apply `agents/rules/testing.mdc` for test coverage

### For Database Work
→ Reference `AGENTS.md` → DatabaseAgent section
→ Apply `agents/rules/database.mdc` patterns
→ Use ORM/ODM patterns, avoid raw queries

### For Security Reviews
→ Reference `AGENTS.md` → SecurityAgent section
→ Apply `agents/rules/security.mdc` comprehensively
→ Check OWASP Top 10 compliance

## When in Doubt

1. Check `AGENTS.md` for agent delegation guidance
2. Review `CLAUDE.md` for architecture decisions
3. Reference the specific `agents/rules/*.mdc` file for detailed patterns
4. Always prioritize security and testing requirements

---

**Remember**: This is a technology-agnostic template. Adapt all patterns to the chosen technology stack while maintaining the core principles of security, testing, and code quality.
