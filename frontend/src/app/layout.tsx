import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AEPO — Payment Control Room",
  description: "Autonomous Enterprise Payment Orchestrator Dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0f1117] text-slate-200 antialiased">{children}</body>
    </html>
  );
}
