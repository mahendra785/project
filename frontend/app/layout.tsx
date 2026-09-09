import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';

import { Header } from '@/components/layout/header';
import { Sidebar } from '@/components/layout/sidebar';
import { Providers } from './providers';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const mono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono-stack',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'PUMS — Perception Uncertainty Management System',
  description:
    'A post-hoc hierarchical uncertainty calibration framework for dense medical vision models under distribution shift. Research prototype; not a medical device.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} ${mono.variable} font-sans antialiased`}>
        <Providers>
          <div className="flex h-screen overflow-hidden">
            <Sidebar className="hidden md:flex" />
            <div className="flex min-w-0 flex-1 flex-col">
              <Header />
              <main className="flex-1 overflow-y-auto">{children}</main>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
