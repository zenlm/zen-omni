import { Link } from 'react-router-dom'
import { Book, Code2, Package, GitBranch } from 'lucide-react'

export default function DocumentationPage() {
  const sections = [
    {
      title: 'Getting Started',
      icon: Book,
      links: [
        { label: 'Installation', href: '#installation' },
        { label: 'Quick Start', href: '#quick-start' },
        { label: 'Basic Usage', href: '#basic-usage' },
      ]
    },
    {
      title: 'Core Concepts',
      icon: Package,
      links: [
        { label: 'Consensus Engines', href: '#engines' },
        { label: 'Block Interface', href: '#blocks' },
        { label: 'Vote Protocol', href: '#votes' },
      ]
    },
    {
      title: 'Language Implementations',
      icon: Code2,
      links: [
        { label: 'Go', href: '/docs/go' },
        { label: 'C', href: '/docs/c' },
        { label: 'Rust', href: '/docs/rust' },
        { label: 'C++', href: '/docs/cpp' },
        { label: 'Python', href: '/docs/python' },
      ]
    },
    {
      title: 'Advanced',
      icon: GitBranch,
      links: [
        { label: 'Network Integration', href: '#network' },
        { label: 'Performance Tuning', href: '#performance' },
        { label: 'Security', href: '#security' },
      ]
    },
  ]

  return (
    <div className="min-h-screen py-12">
      <div className="container">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="mb-12">
            <h1 className="text-4xl lg:text-5xl font-bold font-mono mb-4">
              Documentation
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-400">
              Everything you need to integrate high-performance consensus into your blockchain.
            </p>
          </div>

          {/* Navigation Grid */}
          <div className="grid md:grid-cols-2 gap-8 mb-16">
            {sections.map((section) => (
              <div key={section.title} className="card">
                <div className="flex items-center mb-4">
                  <section.icon className="w-5 h-5 mr-2 text-gray-400" />
                  <h2 className="text-xl font-bold">{section.title}</h2>
                </div>
                <ul className="space-y-2">
                  {section.links.map((link) => (
                    <li key={link.label}>
                      {link.href.startsWith('/') ? (
                        <Link
                          to={link.href}
                          className="text-gray-600 dark:text-gray-400 hover:text-black dark:hover:text-white transition-colors"
                        >
                          → {link.label}
                        </Link>
                      ) : (
                        <a
                          href={link.href}
                          className="text-gray-600 dark:text-gray-400 hover:text-black dark:hover:text-white transition-colors"
                        >
                          → {link.label}
                        </a>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          {/* Content Sections */}
          <div className="prose prose-gray dark:prose-invert max-w-none">
            <section id="installation" className="mb-16">
              <h2 className="text-3xl font-bold font-mono mb-6">Installation</h2>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div className="code-block">
                  <h3 className="text-sm font-mono mb-2 text-gray-500">Go</h3>
                  <pre className="text-xs">go get github.com/luxfi/consensus</pre>
                </div>
                
                <div className="code-block">
                  <h3 className="text-sm font-mono mb-2 text-gray-500">Rust</h3>
                  <pre className="text-xs">cargo add lux-consensus</pre>
                </div>
                
                <div className="code-block">
                  <h3 className="text-sm font-mono mb-2 text-gray-500">Python</h3>
                  <pre className="text-xs">pip install lux-consensus</pre>
                </div>
                
                <div className="code-block">
                  <h3 className="text-sm font-mono mb-2 text-gray-500">C/C++</h3>
                  <pre className="text-xs">make install</pre>
                </div>
              </div>
            </section>

            <section id="quick-start" className="mb-16">
              <h2 className="text-3xl font-bold font-mono mb-6">Quick Start</h2>
              
              <div className="code-block mb-6">
                <pre className="language-go">
{`package main

import (
    "fmt"
    "github.com/luxfi/consensus/engine/core"
)

func main() {
    // Configure consensus parameters
    params := core.ConsensusParams{
        K:               20,
        AlphaPreference: 15,
        AlphaConfidence: 15,
        Beta:            20,
    }
    
    // Create consensus engine
    consensus, err := core.NewCGOConsensus(params)
    if err != nil {
        panic(err)
    }
    
    // Add block and process votes
    consensus.Add(block)
    
    // Check consensus
    if consensus.IsAccepted(blockID) {
        fmt.Println("Consensus achieved!")
    }
}`}
                </pre>
              </div>
            </section>

            <section id="engines" className="mb-16">
              <h2 className="text-3xl font-bold font-mono mb-6">Consensus Engines</h2>
              
              <div className="grid gap-6">
                {[
                  { name: 'Snowball', desc: 'Classic Byzantine fault-tolerant consensus with simple voting mechanism' },
                  { name: 'Avalanche', desc: 'DAG-based consensus with conflict set resolution' },
                  { name: 'Snowflake', desc: 'Simplified binary consensus for quick decisions' },
                  { name: 'DAG', desc: 'Full directed acyclic graph consensus with topological ordering' },
                  { name: 'Chain', desc: 'Linear chain consensus for ordered block processing' },
                  { name: 'PostQuantum', desc: 'Quantum-resistant consensus with lattice-based cryptography' },
                ].map((engine) => (
                  <div key={engine.name} className="card">
                    <h3 className="font-mono font-bold mb-2">{engine.name}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{engine.desc}</p>
                  </div>
                ))}
              </div>
            </section>

            <section id="performance" className="mb-16">
              <h2 className="text-3xl font-bold font-mono mb-6">Performance</h2>
              
              <div className="card">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-800">
                      <th className="text-left py-2">Implementation</th>
                      <th className="text-right py-2">Votes/Second</th>
                      <th className="text-right py-2">Memory</th>
                      <th className="text-right py-2">Latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-100 dark:border-gray-900">
                      <td className="py-2 font-mono">C++</td>
                      <td className="text-right">15,000+</td>
                      <td className="text-right">&lt; 25 MB</td>
                      <td className="text-right">&lt; 1ms</td>
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-900">
                      <td className="py-2 font-mono">C</td>
                      <td className="text-right">14,000+</td>
                      <td className="text-right">&lt; 10 MB</td>
                      <td className="text-right">&lt; 1ms</td>
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-900">
                      <td className="py-2 font-mono">Rust</td>
                      <td className="text-right">13,500+</td>
                      <td className="text-right">&lt; 15 MB</td>
                      <td className="text-right">&lt; 1ms</td>
                    </tr>
                    <tr className="border-b border-gray-100 dark:border-gray-900">
                      <td className="py-2 font-mono">Go</td>
                      <td className="text-right">12,000+</td>
                      <td className="text-right">&lt; 20 MB</td>
                      <td className="text-right">&lt; 2ms</td>
                    </tr>
                    <tr>
                      <td className="py-2 font-mono">Python</td>
                      <td className="text-right">5,000+</td>
                      <td className="text-right">&lt; 50 MB</td>
                      <td className="text-right">&lt; 5ms</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  )
}