export function getMissingAstraEnvVars(env = process.env) {
  const missing = [];

  if (!env.ASTRA_DB_SECURE_BUNDLE_PATH) {
    missing.push("ASTRA_DB_SECURE_BUNDLE_PATH");
  }
  if (!env.ASTRA_DB_KEYSPACE) {
    missing.push("ASTRA_DB_KEYSPACE");
  }

  const hasAppToken = Boolean(env.ASTRA_DB_APPLICATION_TOKEN);
  const hasLegacyCreds = Boolean(env.ASTRA_DB_CLIENT_ID) && Boolean(env.ASTRA_DB_CLIENT_SECRET);

  if (!hasAppToken && !hasLegacyCreds) {
    missing.push("ASTRA_DB_APPLICATION_TOKEN (or ASTRA_DB_CLIENT_ID + ASTRA_DB_CLIENT_SECRET)");
  }

  return missing;
}

export function getAstraCredentials(env = process.env) {
  const rawToken = env.ASTRA_DB_APPLICATION_TOKEN ? String(env.ASTRA_DB_APPLICATION_TOKEN).trim() : "";
  const appToken = rawToken.replace(/^['"]|['"]$/g, "");

  if (appToken) {
    return {
      username: "token",
      password: appToken,
      authMode: "application_token",
    };
  }

  const rawClientId = env.ASTRA_DB_CLIENT_ID ? String(env.ASTRA_DB_CLIENT_ID).trim() : "";
  const rawClientSecret = env.ASTRA_DB_CLIENT_SECRET ? String(env.ASTRA_DB_CLIENT_SECRET).trim() : "";

  return {
    username: rawClientId.replace(/^['"]|['"]$/g, ""),
    password: rawClientSecret.replace(/^['"]|['"]$/g, ""),
    authMode: "legacy_client_id_secret",
  };
}
