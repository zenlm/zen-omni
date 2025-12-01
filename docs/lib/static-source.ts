// Static source implementation to avoid circular references
import type { InferPageType } from "fumadocs-core/source"

// Manually define page data structure
interface PageData {
  title: string
  description?: string
  body: any
  toc?: any[]
  full?: boolean
}

interface Page {
  url: string
  data: PageData
}

// Create a static source implementation
class StaticSource {
  private pages: Map<string, () => Promise<Page>> = new Map()

  constructor() {
    // Register all pages
    this.pages.set("", () => this.loadIndexPage())
    this.pages.set("benchmarks", () => this.loadBenchmarksPage())
    this.pages.set("sdk", () => this.loadSdkIndexPage())
    this.pages.set("sdk/go", () => this.loadSdkGoPage())
    this.pages.set("sdk/c", () => this.loadSdkCPage())
  }

  getPage(slug?: string[]): Page | null {
    const path = slug?.join("/") || ""
    const loader = this.pages.get(path)

    if (!loader) return null

    // For now, return a synchronous placeholder
    // In production, this would be pre-loaded
    return {
      url: `/docs/${path}`,
      data: {
        title: this.getTitleForPath(path),
        description: this.getDescriptionForPath(path),
        body: () => null, // Will be replaced with actual MDX
        toc: [],
        full: false,
      }
    }
  }

  private getTitleForPath(path: string): string {
    const titles: Record<string, string> = {
      "": "Introduction",
      "benchmarks": "Benchmarks",
      "sdk": "SDK Overview",
      "sdk/go": "Go SDK",
      "sdk/c": "C SDK",
    }
    return titles[path] || "Documentation"
  }

  private getDescriptionForPath(path: string): string {
    const descriptions: Record<string, string> = {
      "": "Multi-language consensus engine for blockchain systems",
      "benchmarks": "Performance benchmarks and comparisons",
      "sdk": "Software Development Kits for zen-omni",
      "sdk/go": "Go SDK for zen-omni",
      "sdk/c": "C SDK for zen-omni",
    }
    return descriptions[path] || ""
  }

  private async loadIndexPage(): Promise<Page> {
    const mod = await import("@/content/docs/index.mdx")
    return this.createPage("", mod)
  }

  private async loadBenchmarksPage(): Promise<Page> {
    const mod = await import("@/content/docs/benchmarks.mdx")
    return this.createPage("benchmarks", mod)
  }

  private async loadSdkIndexPage(): Promise<Page> {
    const mod = await import("@/content/docs/sdk/index.mdx")
    return this.createPage("sdk", mod)
  }

  private async loadSdkGoPage(): Promise<Page> {
    const mod = await import("@/content/docs/sdk/go.mdx")
    return this.createPage("sdk/go", mod)
  }

  private async loadSdkCPage(): Promise<Page> {
    const mod = await import("@/content/docs/sdk/c.mdx")
    return this.createPage("sdk/c", mod)
  }

  private createPage(path: string, mod: any): Page {
    return {
      url: `/docs/${path}`,
      data: {
        title: mod.title || this.getTitleForPath(path),
        description: mod.description || this.getDescriptionForPath(path),
        body: mod.default,
        toc: mod.toc || [],
        full: mod.full || false,
      }
    }
  }

  generateParams() {
    return [
      { slug: [] },
      { slug: ["benchmarks"] },
      { slug: ["sdk"] },
      { slug: ["sdk", "go"] },
      { slug: ["sdk", "c"] },
    ]
  }
}

export const staticSource = new StaticSource()
export type Page = InferPageType<any>