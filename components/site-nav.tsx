import Link from "next/link";
import { Shield, Menu } from "lucide-react";
import { cn } from "@/lib/utils";
import { getCurrentSession } from "@/lib/session";
import {
	NavigationMenu,
	NavigationMenuContent,
	NavigationMenuItem,
	NavigationMenuList,
	NavigationMenuTrigger,
	NavigationMenuLink,
} from "@/components/ui/navigation-menu";
import { cva } from "class-variance-authority";
import { MobileNavWrapper } from "./mobile-nav-wrapper";

const navigationMenuTriggerStyle = cva(
	"group inline-flex h-9 w-full whitespace-nowrap items-start justify-center rounded-md bg-slate-800 px-4 py-2 text-sm font-medium disabled:pointer-events-none disabled:opacity-50 ring-ring/10 dark:ring-ring/20 dark:outline-ring/40 outline-ring/50 transition-[color,box-shadow] focus-visible:ring-4 focus-visible:outline-1 text-zinc-200 hover:bg-zinc-200/10 focus:bg-zinc-200/10 data-[active=true]:bg-zinc-200/10 data-[state=open]:bg-zinc-200/10 focus:text-zinc-200 data-[active=true]:text-zinc-200 hover:text-zinc-200",
);

export default async function Nav({ fixed = false }) {
	const session = await getCurrentSession();

	return (
		<div
			className={cn(
				"text-white top-0 w-full z-10 transition-all duration-200",
				"bg-slate-800",
				fixed ? "fixed" : "sticky",
			)}
		>
			<header className="py-4 container max-sm:px-4 flex items-center mx-auto">
				<Link className="flex items-center justify-center" href="/">
					<Shield className="size-5 lg:size-6 2xl:size-7" />
					<span className="ml-2 text-base lg:text-lg 2xl:text-xl font-bold">
						Cyber Buddy
					</span>
				</Link>

				{/* Mobile Menu */}
				<div className="ml-auto md:hidden">
					<MobileNavWrapper session={session} />
				</div>

				{/* Desktop Menu */}
				<div className="ml-auto hidden md:flex gap-4 sm:gap-6">
					<NavigationMenu>
						<NavigationMenuList>
							<NavigationMenuItem>
								<NavigationMenuTrigger className="bg-transparent text-zinc-50 hover:bg-zinc-200/10 focus:bg-zinc-200/10 data-[active=true]:bg-zinc-200/10 data-[state=open]:bg-zinc-200/10 focus:text-zinc-50 data-[active=true]:text-zinc-50 hover:text-zinc-50">
									Password Tools
								</NavigationMenuTrigger>
								<NavigationMenuContent className="bg-slate-800 text-white border-0">
									<Link href="/password/manager" legacyBehavior passHref>
										<NavigationMenuLink
											className={navigationMenuTriggerStyle()}
										>
											Password Manager
										</NavigationMenuLink>
									</Link>
									<Link
										className="w-full"
										href="/password/checker"
										legacyBehavior
										passHref
									>
										<NavigationMenuLink
											className={navigationMenuTriggerStyle()}
										>
											Strength Check
										</NavigationMenuLink>
									</Link>
								</NavigationMenuContent>
							</NavigationMenuItem>
							<NavigationMenuItem>
								<Link legacyBehavior passHref href="/news">
									<NavigationMenuLink className={navigationMenuTriggerStyle()}>
										News
									</NavigationMenuLink>
								</Link>
							</NavigationMenuItem>
							<NavigationMenuItem>
								<Link legacyBehavior passHref href="/url-checker">
									<NavigationMenuLink className={navigationMenuTriggerStyle()}>
										Link Checker
									</NavigationMenuLink>
								</Link>
							</NavigationMenuItem>
							<NavigationMenuItem>
								<Link legacyBehavior passHref href="/data-breach">
									<NavigationMenuLink className={navigationMenuTriggerStyle()}>
										Data Breach Check
									</NavigationMenuLink>
								</Link>
							</NavigationMenuItem>
							<NavigationMenuItem>
								<Link passHref href="https://cymet-ai.streamlit.app/">									
										Cyber AI
								</Link>
							</NavigationMenuItem>
							<NavigationMenuItem>
								{session.user ? (
									<NavigationMenuLink
										href="/signout"
										className={navigationMenuTriggerStyle()}
									>
										Sign out
									</NavigationMenuLink>
								) : (
									<Link href={"/login"} legacyBehavior passHref>
										<NavigationMenuLink
											className={navigationMenuTriggerStyle()}
										>
											Login
										</NavigationMenuLink>
									</Link>
								)}
							</NavigationMenuItem>
						</NavigationMenuList>
					</NavigationMenu>
				</div>
			</header>
		</div>
	);
}
