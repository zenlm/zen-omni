import { getSource } from "@/lib/source-loader"
import { DocsPage, DocsBody } from "fumadocs-ui/page"
import { notFound } from "next/navigation"
import defaultMdxComponents from "fumadocs-ui/mdx"

export const revalidate = false
// For static export, we cannot use dynamicParams
export const dynamicParams = false

export default async function Page({
  params,
}: {
  params: Promise<{ slug?: string[] }>
}) {
  const { slug } = await params
  const source = getSource()
  const page = source.getPage(slug)
  if (!page) notFound()

  const MDX = page.data.body

  return (
    <DocsPage
      toc={page.data.toc}
      full={page.data.full}
      tableOfContent={{
        style: "clerk",
      }}
    >
      <DocsBody>
        <h1>{page.data.title}</h1>
        <MDX components={{ ...defaultMdxComponents }} />
      </DocsBody>
    </DocsPage>
  )
}

// Commenting out generateStaticParams to avoid stack overflow
// The pages will be generated on-demand instead
// export function generateStaticParams() {
//   return [
//     { slug: [] },
//     { slug: ["benchmarks"] },
//     { slug: ["sdk"] },
//     { slug: ["sdk", "go"] },
//     { slug: ["sdk", "c"] },
//   ]
// }

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug?: string[] }>
}) {
  const { slug } = await params
  const source = getSource()
  const page = source.getPage(slug)
  if (!page) notFound()

  return {
    title: `${page.data.title} | zen-omni`,
    description: page.data.description,
  }
}
