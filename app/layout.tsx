import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";
import { NuqsAdapter } from "nuqs/adapters/next/app";

const geistSans = Geist({
	variable: "--font-geist-sans",
	subsets: ["latin"],
	fallback: [
		"ui-sans-serif",
		"system-ui",
		"sans-serif",
		"Apple Color Emoji",
		"Segoe UI Emoji",
		"Segoe UI Symbol",
		"Noto Color Emoji",
	],
});

export const metadata: Metadata = {
	title: "Cyber Buddy",
	description: "One stop place for cyber geeks",
};

export default function RootLayout({
	children,
}: Readonly<{
	children: React.ReactNode;
}>) {
	return (
		<html className={`${geistSans.variable} scroll-smooth`} lang="en" suppressHydrationWarning>
			<body className="antialiased" suppressHydrationWarning>
				<NuqsAdapter>{children}</NuqsAdapter>
			</body>
		</html>
	);
}
