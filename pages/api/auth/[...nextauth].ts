import NextAuth from "next-auth";
import { authOptions } from "../../../src/utils/server/auth_options";

export default NextAuth(authOptions);
