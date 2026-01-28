# Next.js Full-Stack Boilerplate

A modern, production-ready boilerplate for Next.js projects with TypeScript, Tailwind CSS, and Supabase/Prisma integration. This boilerplate includes comprehensive Cursor AI rules to help maintain consistency and best practices across projects.

## Features

- **Next.js 14+** with App Router
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **Supabase/Prisma** for database management
- **Cursor AI Rules** (.mdc files) for consistent development
- **Pre-configured** ESLint and Prettier
- **Modern folder structure** following best practices

## Quick Start

1. **Clone or copy this boilerplate** to your new project directory
2. **Install dependencies**:
   ```bash
   npm install
   # or
   yarn install
   # or
   pnpm install
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your configuration
   ```

4. **Run the development server**:
   ```bash
   npm run dev
   # or
   yarn dev
   # or
   pnpm dev
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser

## Project Structure

```
├── .cursor/
│   └── rules/          # Cursor AI rule files (.mdc)
│       ├── core.mdc    # Core guidelines and tech stack
│       ├── frontend.mdc # React/Next.js patterns
│       ├── database.mdc # Database and backend patterns
│       └── style.mdc   # Code style and formatting
├── app/                # Next.js App Router pages
├── components/         # React components
├── hooks/              # Custom React hooks
├── lib/                # Utility functions and clients
├── public/             # Static assets
├── docs/               # Project documentation
└── [config files]      # TypeScript, Tailwind, etc.
```

## Cursor AI Rules

This boilerplate includes comprehensive `.mdc` rule files in `.cursor/rules/` that guide Cursor AI to:

- Follow consistent coding patterns
- Use the correct tech stack conventions
- Maintain code style and formatting
- Implement best practices for Next.js, React, and databases

### Rule Files

- **core.mdc**: General architecture, tech stack, and developer preferences
- **frontend.mdc**: React/Next.js patterns, Tailwind CSS, and UI guidelines
- **database.mdc**: Supabase and Prisma patterns, data fetching, and API routes
- **style.mdc**: Naming conventions, formatting, and code quality standards

### Customizing Rules

You can edit any `.mdc` file in `.cursor/rules/` to match your project's specific needs. The rules use YAML frontmatter to define when they apply:

- `alwaysApply: true` - Always active
- `globs: ["**/*.tsx"]` - Apply to specific file patterns
- `tags: ["frontend"]` - Categorize rules for manual selection

See [Cursor Rules Documentation](https://docs.cursor.com/context/rules) for more details.

## Tech Stack

- **Framework**: Next.js 14+ (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Database**: Supabase (PostgreSQL) or Prisma ORM
- **Deployment**: Vercel (recommended)

## Development

### Code Formatting

This project uses ESLint and Prettier for code quality:

```bash
# Check for linting errors
npm run lint

# Format code
npm run format
```

### Type Checking

```bash
npm run type-check
```

## Environment Variables

Create a `.env.local` file with the following variables:

```env
# Supabase (if using)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key

# Database (if using Prisma)
DATABASE_URL=your_database_url

# Add other environment variables as needed
```

## Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Import your repository in [Vercel](https://vercel.com)
3. Add your environment variables
4. Deploy!

### Other Platforms

This boilerplate can be deployed to any platform that supports Next.js:
- Netlify
- AWS Amplify
- Railway
- DigitalOcean App Platform

## Documentation

- [Architecture Overview](docs/architecture.md) - System design and data flow
- [Cursor Rules](.cursor/rules/) - AI assistant guidelines

## Contributing

This is a personal boilerplate template. Customize it to fit your needs!

## License

MIT License - feel free to use this boilerplate for your projects.

## Notes

- This boilerplate is designed for full-stack development with a focus on modern React patterns
- The Cursor AI rules are tailored for a developer named Rick with specific preferences
- Update the rules and configuration files as your needs evolve
