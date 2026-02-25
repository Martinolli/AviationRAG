export function getMissingAstraEnvVars(env = process.env) {
  const missing = [];

  if (!env.ASTRA_DB_SECURE_BUNDLE_PATH) {
    missing.push("ASTRA_DB_SECURE_BUNDLE_PATH");
  }
  if (!env.ASTRA_DB_KEYSPACE) {
    missing.push("ASTRA_DB_KEYSPACE");
  }

  const hasAppToken = Boolean(env.ASTRA_DB_APPLICATION_TOKEN);
  if (!hasAppToken) {
    missing.push("ASTRA_DB_APPLICATION_TOKEN");
  }

  return missing;
}

export function getAstraCredentials(env = process.env) {
  const rawToken = env.ASTRA_DB_APPLICATION_TOKEN ? String(env.ASTRA_DB_APPLICATION_TOKEN).trim() : "";
  const appToken = rawToken.replace(/^['"]|['"]$/g, "");

  return {
    username: "token",
    password: appToken,
    authMode: "application_token",
  };
}
