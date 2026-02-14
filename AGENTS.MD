## Agents

This file defines the different agents (MCP helpers) available in this **technology-agnostic development template** and when to use each one.

- See `CLAUDE.md` for overall project guidelines.
- Use this file as a quick reference for **which agent to call for which task**.
- All agents support **security-first development** with comprehensive validation patterns.
- Adapt the specific technologies and tools mentioned to match your chosen tech stack.

---

### FrontendAgent (UI/UX Developer)

**FrontendAgent is your frontend developer.** For all UI/design work, use this MCP. Adapt tool descriptions to your chosen frontend framework.

#### When to use FrontendAgent

- New visual component (popup, card, section, etc.) → `snippet_frontend` or `create_frontend`
- Redesign/restyle of existing UI → `modify_frontend`
- Design system implementation → follow design integration patterns
- Component library and design system work
- Accessibility improvements and WCAG compliance
- Pure logic or trivial tweaks → handle directly without delegating

#### Enhanced Capabilities

- **Security-aware components**: Automatically includes proper input sanitization
- **Accessibility-first**: Components include ARIA attributes and keyboard navigation
- **Performance-optimized**: Uses framework-specific optimization patterns
- **Type-safe**: All components include proper type definitions (TypeScript/PropTypes/etc.)

#### Framework Adaptations

**React/Next.js/Vue/Svelte:**
- Component-based architecture with proper state management
- Framework-specific image optimization and lazy loading
- CSS-in-JS, Tailwind, or styled-components patterns

**Angular:**
- Component + service architecture
- RxJS reactive patterns
- Angular Material or custom design system

#### Critical Rules

1. If UI already exists and you need to redesign/restyle it → use `modify_frontend`, **not** `snippet_frontend`.
2. Tasks can be mixed (logic + UI). Handle logic yourself, delegate UI to FrontendAgent.
3. **Always include accessibility testing** patterns for new components.
4. **Follow security patterns** from `agents/rules/security.mdc` for user inputs.

---

### BackendAgent (API & Business Logic)

#### Scope

- Owns **API routes**, **business logic layers**, and **service functions**
- **Authentication & Authorization** implementation
- **Rate limiting** and **security middleware**
- **Input validation** with schema validation libraries

#### Enhanced Responsibilities

- Implement secure API endpoints with comprehensive validation
- Add authentication middleware and role-based access control
- Integrate rate limiting for DoS protection
- Create type-safe API responses and data fetching
- Handle errors gracefully with proper logging
- Move business logic into dedicated service functions (not in controllers/handlers)
- Keep responses typed and consistent with shared interfaces

#### Security Requirements

- **Always validate input** at API boundaries using schema validation
- **Implement rate limiting** for all public endpoints
- **Apply OWASP Top 10** protection patterns
- **Log security events** appropriately without exposing sensitive data
- **Handle errors** gracefully with proper error responses

#### Technology Adaptations

**Node.js (Express/Fastify/Next.js API):**
- RESTful or GraphQL API patterns
- Middleware-based authentication and validation
- Schema validation with Zod, Joi, or similar

**Python (Django/FastAPI/Flask):**
- Django REST Framework or FastAPI patterns
- Pydantic models for validation
- SQLAlchemy or Django ORM patterns

**Other Stacks:**
- Adapt patterns to your chosen backend framework
- Maintain consistent validation, authentication, and error handling

---

### DatabaseAgent (Data Layer & Schema Management)

#### Scope

- Owns **database schema, migrations, and data access patterns**
- **Technology-agnostic**: Supports various database and ORM approaches

#### Enhanced Responsibilities

- **Schema Design**: Create optimized, scalable database schemas
- **Migration Management**: Handle schema evolution safely
- **Query Optimization**: Prevent N+1 queries and performance issues
- **Data Access Patterns**: Implement consistent, type-safe data access
- **Security**: Access controls, input sanitization, and audit logging

#### Database Strategy Support

**SQL Databases (PostgreSQL/MySQL/SQLite):**
- Schema design with proper relationships and constraints
- Migration strategies with version control
- ORM integration (Prisma, TypeORM, Sequelize, SQLAlchemy, etc.)
- Connection pooling and optimization

**NoSQL Databases (MongoDB/DynamoDB/Firestore):**
- Document schema design and validation
- Index optimization and query patterns
- ODM integration (Mongoose, etc.)

**Cloud-Native Solutions:**
- Supabase, Firebase, AWS RDS, Azure SQL, etc.
- Built-in authentication and real-time features
- Row-level security policies

#### Security Requirements

- **Input validation**: All database inputs must be validated
- **Access control**: Implement proper authorization checks
- **Injection prevention**: Use ORM/ODM-only access, avoid raw queries
- **Audit logging**: Track important data changes
- **Data encryption**: Handle sensitive data appropriately

#### Must

- **Choose one approach** and maintain consistency throughout the project
- Never run manual destructive queries in production code
- Always keep schema + migrations in sync
- **Test database operations** with appropriate test patterns
- **Document schema decisions** and migration strategies

---

### TestAgent (Comprehensive Testing)

#### Scope

- Owns **complete testing strategy**: unit, integration, E2E, and accessibility testing
- **Framework-agnostic**: Adapts to chosen testing tools and frameworks

#### Enhanced Responsibilities

- **Unit Testing**: Components, functions, utilities with >80% coverage
- **Integration Testing**: API endpoints, database operations, authentication flows
- **E2E Testing**: Critical user journeys across browsers/devices
- **Accessibility Testing**: Automated WCAG compliance checking
- **Performance Testing**: Load times, Core Web Vitals, and performance budgets
- **Security Testing**: Input validation and authentication testing

#### Testing Patterns

- **Component/Unit Testing**: Framework-appropriate testing utilities
- **API Testing**: Mock dependencies, test authentication and validation
- **E2E Testing**: Page Object Model with stable selectors and user scenarios
- **Accessibility Testing**: Automated scanning and keyboard navigation
- **Visual Regression**: Screenshot comparison for critical UI components

#### Quality Gates

- **Code Coverage**: Maintain >80% coverage for business logic
- **E2E Coverage**: All critical user flows must have E2E tests
- **Accessibility**: All public interfaces must pass WCAG 2.1 AA standards
- **Performance**: Meet performance thresholds for your platform
- **Security**: Authentication and authorization flows fully tested

#### Technology Adaptations

**JavaScript/TypeScript:**
- Jest, Vitest, or framework-specific test runners
- Testing Library, Enzyme, or component testing utilities
- Playwright, Cypress, or Selenium for E2E testing

**Python:**
- pytest, unittest with appropriate fixtures and mocking
- Selenium or Playwright for browser testing

**Other Languages:**
- Adapt to language-specific testing frameworks
- Maintain consistent testing patterns and quality gates

---

### SecurityAgent (Security & Compliance)

#### Scope

- **Security-first development** across all application layers
- **OWASP Top 10** protection implementation
- **Authentication & authorization** security patterns
- **Input validation** and **output sanitization**
- **Security monitoring** and **audit logging**

#### Responsibilities

- **Input Validation**: Implement schema validation for all user inputs
- **Authentication Security**: Secure session management and multi-factor authentication
- **Authorization**: Role-based access control (RBAC) implementation
- **Rate Limiting**: DoS protection with appropriate algorithms
- **Security Headers**: CSP, HSTS, and comprehensive security header configuration
- **Data Protection**: Encryption, sanitization, and secure data handling
- **Audit Logging**: Security event logging without exposing sensitive data
- **Vulnerability Assessment**: Regular security pattern reviews

#### Security Patterns

- **Authentication**: Multi-provider authentication with secure session management
- **Authorization**: Middleware-based route protection with role validation
- **Input Validation**: Schema validation at every boundary
- **Rate Limiting**: Technology-appropriate rate limiting implementation
- **Error Handling**: Secure error responses that don't leak sensitive information
- **Database Security**: ORM/ODM-only access with parameterized queries
- **File Upload Security**: Content type validation and secure storage

#### Technology Adaptations

**Web Applications:**
- JWT tokens with proper expiration and refresh patterns
- HTTPS enforcement and security headers
- CORS configuration and CSRF protection

**API Security:**
- API key management and OAuth2/OpenID Connect
- Request signing and payload validation
- Rate limiting per API consumer

#### Must

- **Validate all inputs** with appropriate schema validation
- **Implement rate limiting** on authentication and public endpoints
- **Apply security headers** to all responses
- **Log security events** appropriately
- **Never expose sensitive data** in error messages or logs

---

### ReviewerAgent (Code Review & Quality)

#### Enhanced Scope

- Reviews changes for **correctness, security, performance, and readability**
- **Security vulnerability assessment** using OWASP guidelines
- **Performance impact analysis** and optimization recommendations
- **Test coverage verification** and quality assessment
- **Accessibility compliance** checking
- **Code quality standards** enforcement

#### Security Review Focus

- **Input validation**: Ensure all user inputs are properly validated
- **Authentication flows**: Verify secure authentication implementation
- **Authorization checks**: Validate role-based access controls
- **Data exposure**: Check for sensitive data leaks in responses
- **Error handling**: Ensure secure error message patterns
- **Dependencies**: Review third-party package security

#### Quality Standards

- **Type Safety**: Strict typing without loose type usage
- **Testing**: Appropriate test coverage for new functionality
- **Performance**: Bundle/binary size impact and performance considerations
- **Accessibility**: WCAG 2.1 AA compliance for user-facing features
- **Security**: OWASP Top 10 protection patterns
- **Documentation**: Clear code documentation and README updates

#### Must

- **Flag missing tests** for important logic and security flows
- **Ensure security patterns** from `agents/rules/security.mdc` are followed
- **Validate accessibility** for all user-facing components
- **Check performance impact** of new dependencies and code changes
- **Verify documentation** is updated for significant changes
- **Ensure conventions** in `CLAUDE.md` are followed throughout

---

## Design Integration Patterns

These rules define how to translate design inputs into code and must be followed for design-driven development.

#### Required Design-to-Code Flow

1. **Analyze Design**: Extract design tokens, components, and interaction patterns
2. **Create Design System**: Establish consistent color, typography, and spacing tokens
3. **Component Architecture**: Break down designs into reusable components
4. **Responsive Patterns**: Ensure designs work across device sizes
5. **Accessibility Integration**: Include ARIA attributes and keyboard navigation
6. **Performance Optimization**: Implement lazy loading and efficient rendering

#### Implementation Rules

- **Design System First**: Create consistent tokens and reusable components
- **Accessibility by Default**: Include ARIA labels, focus management, and keyboard navigation
- **Performance Conscious**: Optimize images, fonts, and rendering patterns
- **Responsive Design**: Mobile-first approach with appropriate breakpoints
- **Type Safety**: Define proper interfaces for component props and data
- **Testing Integration**: Include visual regression and accessibility testing

#### Framework-Specific Patterns

**Component-Based (React/Vue/Svelte):**
- Atomic design principles with proper component hierarchy
- Consistent prop interfaces and state management
- Style composition with CSS modules, styled-components, or utility classes

**Template-Based (Django/Rails/PHP):**
- Consistent partial/template structure
- Server-side rendering with progressive enhancement
- Asset optimization and caching strategies

---

## Usage Guidelines

1. **Start with Security**: Always consider security implications first
2. **Choose Your Stack**: Adapt the specific technologies to your project needs
3. **Maintain Consistency**: Once you choose patterns, apply them consistently
4. **Test Everything**: Implement comprehensive testing at all levels
5. **Document Decisions**: Keep architectural decisions and patterns documented
6. **Review Regularly**: Use ReviewerAgent to maintain code quality

This template provides the agent structure while remaining flexible for any technology stack you choose to implement.