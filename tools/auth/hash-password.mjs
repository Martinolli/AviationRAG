import crypto from "crypto";

const password = process.argv[2] || "";

if (!password) {
  process.stderr.write("Usage: node tools/auth/hash-password.mjs \"your-password\"\n");
  process.exit(1);
}

const digest = crypto.createHash("sha256").update(password, "utf8").digest("hex");
process.stdout.write(`sha256:${digest}\n`);
