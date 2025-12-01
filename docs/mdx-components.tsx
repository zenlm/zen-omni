import type { MDXComponents } from 'mdx/types'
import defaultComponents from 'fumadocs-ui/mdx'
import { Card, Cards } from 'fumadocs-ui/components/card'
import { Tabs, TabsList, TabsTrigger, TabsContent } from 'fumadocs-ui/components/tabs'

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    ...defaultComponents,
    Card,
    Cards,
    Tabs,
    TabsList,
    TabsTrigger,
    TabsContent,
    ...components,
  }
}