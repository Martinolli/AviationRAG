import type { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

const defaultEmail = "admin@aviationrag.local";
const defaultName = "AviationRAG User";

export const authOptions: NextAuthOptions = {
  session: { strategy: "jwt" },
  pages: { signIn: "/auth/signin" },
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        const expectedEmail = String(process.env.APP_AUTH_EMAIL || defaultEmail).trim().toLowerCase();
        const expectedPassword = String(process.env.APP_AUTH_PASSWORD || "").trim();

        const email = String(credentials?.email || "").trim().toLowerCase();
        const password = String(credentials?.password || "").trim();

        if (!expectedPassword) {
          return null;
        }

        if (email !== expectedEmail || password !== expectedPassword) {
          return null;
        }

        return {
          id: "local-user",
          name: process.env.APP_AUTH_NAME || defaultName,
          email: expectedEmail,
        };
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.name = user.name;
        token.email = user.email;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.name = String(token.name || session.user.name || defaultName);
        session.user.email = String(token.email || session.user.email || defaultEmail);
      }
      return session;
    },
  },
};

