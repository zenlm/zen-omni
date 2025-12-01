import Link from 'next/link';
import { redirect } from 'next/navigation';

export default function HomePage() {
  // For static export, we need to provide actual content
  // Uncomment redirect for server-side rendering
  // redirect('/docs');

  return (
    <>
      <meta httpEquiv="refresh" content="0;url=/docs" />
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4">zen-omni</h1>
          <p className="text-muted-foreground mb-6">
            Multi-language consensus engine for blockchain systems
          </p>
          <Link
            href="/docs"
            className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-3 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            View Documentation →
          </Link>
        </div>
      </div>
    </>
  );
}
