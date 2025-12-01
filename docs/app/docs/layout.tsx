import { DocsLayout } from "fumadocs-ui/layouts/docs"
import type { ReactNode } from "react"
import { BookOpen, Code, Cpu } from "lucide-react"
import { Logo } from "../../components/logo"

// Static page tree to avoid circular dependencies
const pageTree = {
  name: 'Docs',
  children: [
    {
      type: 'page',
      name: 'Introduction',
      url: '/docs',
    },
    {
      type: 'page',
      name: 'Benchmarks',
      url: '/docs/benchmarks',
    },
    {
      type: 'folder',
      name: 'SDK',
      children: [
        {
          type: 'page',
          name: 'Overview',
          url: '/docs/sdk',
        },
        {
          type: 'page',
          name: 'Go SDK',
          url: '/docs/sdk/go',
        },
        {
          type: 'page',
          name: 'Python SDK',
          url: '/docs/sdk/python',
        },
        {
          type: 'page',
          name: 'Rust SDK',
          url: '/docs/sdk/rust',
        },
        {
          type: 'page',
          name: 'C++ SDK',
          url: '/docs/sdk/cpp',
        },
        {
          type: 'page',
          name: 'C SDK',
          url: '/docs/sdk/c',
        },
        {
          type: 'page',
          name: 'MLX GPU',
          url: '/docs/sdk/mlx',
        },
      ],
    },
  ],
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <DocsLayout
      tree={pageTree}
      nav={{
        title: (
          <div className="flex items-center gap-2">
            <Logo size={24} variant="white" />
            <span className="font-bold">zen-omni</span>
          </div>
        ),
        transparentMode: "top",
      }}
      sidebar={{
        defaultOpenLevel: 0,
        banner: (
          <div className="rounded-lg bg-gradient-to-br from-lux-500 to-lux-700 p-4 text-white">
            <h3 className="text-sm font-semibold">v1.21.0 Released! 🎉</h3>
            <p className="mt-1 text-xs opacity-90">
              Multi-language SDK with quantum integration
            </p>
          </div>
        ),
        footer: (
          <div className="flex flex-col gap-2 p-4 text-xs text-muted-foreground">
            <a
              href="https://github.com/luxfi/consensus"
              className="hover:text-foreground"
            >
              GitHub
            </a>
            <a href="https://lux.fi" className="hover:text-foreground">
              Zen Network
            </a>
          </div>
        ),
      }}
      links={[
        {
          text: "Documentation",
          url: "/docs",
          icon: <BookOpen className="size-4" />,
        },
        {
          text: "SDK",
          url: "/docs/sdk",
          icon: <Code className="size-4" />,
        },
        {
          text: "Benchmarks",
          url: "/docs/benchmarks",
          icon: <Cpu className="size-4" />,
        },
      ]}
    >
      {children}
    </DocsLayout>
  )
}
