# Security Guidelines for Arrbit

- Never commit API keys, tokens, or passwords to the repository.
- Store secrets in environment variables or in files listed in `.gitignore` (e.g., `.env`).
- Do not log secrets or sensitive URLs. Always redact or avoid logging sensitive data.
- Review all configuration and payload files for accidental secrets before committing.
- If you add new config files with secrets, use a `.local` or `.env` extension and add them to `.gitignore`.
- For more details, see `.github/copilot-instructions.md` and `.github/golden_standard.md`.
