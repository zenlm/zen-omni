import { source } from '@/lib/source'
import { DocsPage, DocsBody, DocsTitle, DocsDescription } from 'fumadocs-ui/page'
import { notFound } from 'next/navigation'
import type { Metadata } from 'next'
import { MDXProvider } from '@mdx-js/react'

interface PageProps {
  params: Promise<{ slug?: string[] }>
}

export async function generateStaticParams() {
  return source.getPages().map((page) => ({
    slug: page.slugs,
  }))
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const resolvedParams = await params
  const page = source.getPage(resolvedParams.slug)
  
  if (!page) notFound()
  
  return {
    title: page.data.title,
    description: page.data.description,
  }
}

export default async function Page({ params }: PageProps) {
  const resolvedParams = await params
  const page = source.getPage(resolvedParams.slug)
  
  if (!page) notFound()

  const MDX = page.data.default || (() => null)

  return (
    <DocsPage>
      <DocsTitle>{page.data.title}</DocsTitle>
      {page.data.description && (
        <DocsDescription>{page.data.description}</DocsDescription>
      )}
      <DocsBody>
        <MDX />
      </DocsBody>
    </DocsPage>
  )
}