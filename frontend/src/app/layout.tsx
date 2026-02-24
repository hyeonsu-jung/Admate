import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Admate RAG Chatbot",
  description: "AI 어시스턴트",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
