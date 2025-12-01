// Re-export from source-loader to maintain compatibility
import { getSource } from "./source-loader"
import type { InferPageType } from "fumadocs-core/source"

export const source = getSource()
export type Page = InferPageType<typeof source>
