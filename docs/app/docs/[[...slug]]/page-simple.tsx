import { DocsPage, DocsBody } from "fumadocs-ui/page"
import { notFound } from "next/navigation"
import defaultMdxComponents from "fumadocs-ui/mdx"

// Import MDX files directly
import IndexMDX from "@/content/docs/index.mdx"
import BenchmarksMDX from "@/content/docs/benchmarks.mdx"
import SdkIndexMDX from "@/content/docs/sdk/index.mdx"
import SdkGoMDX from "@/content/docs/sdk/go.mdx"
import SdkCMDX from "@/content/docs/sdk/c.mdx"

export const revalidate = false
export const dynamicParams = false

// Map of slug paths to MDX components and metadata
const pages = {
  "": { Component: IndexMDX, title: "Introduction", description: "Multi-language consensus engine for blockchain systems" },
  "benchmarks": { Component: BenchmarksMDX, title: "Benchmarks", description: "Performance benchmarks and comparisons" },
  "sdk": { Component: SdkIndexMDX, title: "SDK Overview", description: "Software Development Kits for zen-omni" },
  "sdk/go": { Component: SdkGoMDX, title: "Go SDK", description: "Go SDK for zen-omni" },
  "sdk/c": { Component: SdkCMDX, title: "C SDK", description: "C SDK for zen-omni" },
}

export default async function Page({
  params,
}: {
  params: Promise<{ slug?: string[] }>
}) {
  const { slug = [] } = await params
  const path = slug.join("/")
  const page = pages[path as keyof typeof pages]

  if (!page) notFound()

  const MDX = page.Component

  return (
    <DocsPage
      tableOfContent={{
        style: "clerk",
      }}
    >
      <DocsBody>
        <h1>{page.title}</h1>
        <MDX components={{ ...defaultMdxComponents }} />
      </DocsBody>
    </DocsPage>
  )
}

export function generateStaticParams() {
  return [
    { slug: [] },
    { slug: ["benchmarks"] },
    { slug: ["sdk"] },
    { slug: ["sdk", "go"] },
    { slug: ["sdk", "c"] },
  ]
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug?: string[] }>
}) {
  const { slug = [] } = await params
  const path = slug.join("/")
  const page = pages[path as keyof typeof pages]

  if (!page) return {}

  return {
    title: `${page.title} | zen-omni`,
    description: page.description,
  }
}