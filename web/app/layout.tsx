import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = { title: "CampusGather | Research analyses", description: "Interactive, reproducible analysis of aggregate campus interaction patterns." };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
