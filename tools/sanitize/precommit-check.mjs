import { execSync } from "child_process";
import fs from "fs";

const STAGED_SIZE_LIMIT_BYTES = 10 * 1024 * 1024;

const pathSizeOverrides = [
  { regex: /^data\/documents\/.+\.(pdf|docx|txt)$/i, maxBytes: 120 * 1024 * 1024 },
  { regex: /^data\/raw\/.+\.(pkl|json|csv)$/i, maxBytes: 120 * 1024 * 1024 },
];

const blockedPathPatterns = [
  /^chat_id\/.*$/i,
  /^data\/astra_db\/.*$/i,
  /^data\/embeddings\/.*$/i,
];

const secretPatterns = [
  { name: "OpenAI key", regex: /\bsk-[A-Za-z0-9]{20,}\b/g },
  { name: "AWS access key", regex: /\bAKIA[0-9A-Z]{16}\b/g },
  { name: "Private key block", regex: /-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----/g },
  { name: "Google API key", regex: /\bAIza[0-9A-Za-z\-_]{20,}\b/g },
  {
    name: "Likely secret assignment",
    regex:
      /^(OPENAI_API_KEY|ASTRA_DB_APPLICATION_TOKEN|NEXTAUTH_SECRET|APP_AUTH_PASSWORD)\s*=\s*["']?[A-Za-z0-9_\-\/+=]{8,}/gm,
  },
];

function getFilesToCheck() {
  const scanAll = process.argv.includes("--all");
  if (scanAll) {
    return execSync("git ls-files", { encoding: "utf8" })
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
  }

  return execSync("git diff --cached --name-only --diff-filter=ACM", { encoding: "utf8" })
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function shouldCheckLargeFiles(scanAll) {
  // In full-repo mode, enforce only secret/pattern checks and avoid failing on
  // existing tracked baseline assets. Large-file prevention still runs for staged files.
  return !scanAll;
}

function maxAllowedSizeForPath(file) {
  for (const override of pathSizeOverrides) {
    if (override.regex.test(file)) {
      return override.maxBytes;
    }
  }
  return STAGED_SIZE_LIMIT_BYTES;
}

function shouldCheckForSecrets(file) {
  const lower = file.toLowerCase();
  if (lower === ".env.example") {
    return false;
  }
  if (lower.endsWith(".md")) {
    return false;
  }
  return true;
}

function isBinary(buffer) {
  const maxScan = Math.min(buffer.length, 4096);
  for (let i = 0; i < maxScan; i += 1) {
    if (buffer[i] === 0) return true;
  }
  return false;
}

function main() {
  const scanAll = process.argv.includes("--all");
  const files = getFilesToCheck();
  const violations = [];
  const checkLargeFiles = shouldCheckLargeFiles(scanAll);

  for (const file of files) {
    if (!fs.existsSync(file)) {
      continue;
    }

    const stats = fs.statSync(file);
    const maxBytes = maxAllowedSizeForPath(file);
    if (checkLargeFiles && stats.size > maxBytes) {
      violations.push(
        `[large-file] ${file} is ${(stats.size / (1024 * 1024)).toFixed(2)} MB (limit ${(
          maxBytes /
          (1024 * 1024)
        ).toFixed(0)} MB).`,
      );
    }

    for (const pattern of blockedPathPatterns) {
      if (pattern.test(file)) {
        violations.push(`[blocked-path] ${file} matches blocked pattern ${pattern}.`);
      }
    }

    const buffer = fs.readFileSync(file);
    if (isBinary(buffer)) {
      continue;
    }
    if (!shouldCheckForSecrets(file)) {
      continue;
    }

    const content = buffer.toString("utf8");
    for (const pattern of secretPatterns) {
      if (pattern.regex.test(content)) {
        violations.push(`[secret] ${file} matches pattern "${pattern.name}".`);
      }
      pattern.regex.lastIndex = 0;
    }
  }

  if (violations.length > 0) {
    process.stderr.write("\nSanitization pre-commit checks failed:\n");
    for (const violation of violations) {
      process.stderr.write(`- ${violation}\n`);
    }
    process.stderr.write(
      "\nFix the issues or intentionally bypass with 'git commit --no-verify' (not recommended).\n",
    );
    process.exit(1);
  }

  process.stdout.write("Sanitization checks passed.\n");
}

main();
