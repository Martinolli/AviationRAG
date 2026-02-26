import Head from "next/head";
import { signIn, useSession } from "next-auth/react";
import { useRouter } from "next/router";
import { FormEvent, useEffect, useState } from "react";
import styles from "../../styles/AuthSignIn.module.css";

export default function SignInPage() {
  const router = useRouter();
  const { status } = useSession();
  const [loading, setLoading] = useState(false);
  const [errorText, setErrorText] = useState("");

  useEffect(() => {
    if (status === "authenticated") {
      void router.replace("/");
    }
  }, [status, router]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setErrorText("");
    setLoading(true);

    const formData = new FormData(event.currentTarget);
    const email = String(formData.get("email") || "").trim();
    const password = String(formData.get("password") || "");

    if (!email || !password) {
      setLoading(false);
      setErrorText("Email and password are required.");
      return;
    }

    const result = await signIn("credentials", {
      email,
      password,
      redirect: false,
      callbackUrl: "/",
    });

    setLoading(false);

    if (!result || result.error) {
      setErrorText("Invalid email or password.");
      return;
    }

    void router.replace(result.url || "/");
  };

  return (
    <>
      <Head>
        <title>Sign In | AviationRAG</title>
      </Head>

      <main className={styles.page}>
        <section className={styles.card}>
          <h1>AviationRAG Login</h1>
          <p>Sign in to access the certification assistant.</p>

          <form onSubmit={onSubmit} className={styles.form}>
            <label>
              Email
              <input
                name="email"
                type="email"
                autoComplete="username"
                required
              />
            </label>

            <label>
              Password
              <input
                name="password"
                type="password"
                autoComplete="current-password"
                required
              />
            </label>

            {errorText ? <div className={styles.error}>{errorText}</div> : null}

            <button type="submit" disabled={loading}>
              {loading ? "Signing in..." : "Sign In"}
            </button>
          </form>
        </section>
      </main>
    </>
  );
}
