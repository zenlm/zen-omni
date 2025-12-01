import { useParams, Link } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'

const languageData = {
  go: {
    name: 'Go',
    icon: '🔷',
    description: 'Production blockchain integration with concurrent processing',
    installation: 'go get github.com/luxfi/consensus',
    example: `package main

import (
    "github.com/luxfi/consensus/engine/core"
    "github.com/luxfi/ids"
)

func main() {
    params := core.ConsensusParams{
        K: 20,
        AlphaPreference: 15,
    }
    
    consensus, _ := core.NewCGOConsensus(params)
    consensus.Add(block)
}`,
  },
  c: {
    name: 'C',
    icon: '🔧',
    description: 'High-performance native implementation with minimal overhead',
    installation: 'make install',
    example: `#include <consensus.h>

int main() {
    consensus_t* consensus = consensus_new(SNOWBALL);
    consensus_configure(consensus, &params);
    consensus_add_block(consensus, &block);
    consensus_free(consensus);
    return 0;
}`,
  },
  rust: {
    name: 'Rust',
    icon: '🦀',
    description: 'Memory-safe systems programming with zero-cost abstractions',
    installation: 'cargo add lux-consensus',
    example: `use lux_consensus::{Consensus, EngineType};

fn main() {
    let consensus = Consensus::new(
        EngineType::Snowball, 
        params
    )?;
    
    consensus.add_block(block).await?;
}`,
  },
  cpp: {
    name: 'C++',
    icon: '⚙️',
    description: 'Modern C++ with MLX GPU acceleration',
    installation: 'cmake . && make install',
    example: `#include <lux/consensus.hpp>

int main() {
    auto consensus = lux::consensus::Consensus::create(
        EngineType::Snowball, 
        params
    );
    
    consensus->add_block(block);
    return 0;
}`,
  },
  python: {
    name: 'Python',
    icon: '🐍',
    description: 'Research and prototyping with data science integration',
    installation: 'pip install lux-consensus',
    example: `from lux_consensus import Consensus, EngineType

consensus = Consensus(EngineType.SNOWBALL, params)
consensus.add_block(block)

if consensus.is_accepted(block_id):
    print("Consensus achieved!")`,
  },
}

export default function LanguagePage() {
  const { language } = useParams<{ language: string }>()
  const lang = languageData[language as keyof typeof languageData]

  if (!lang) {
    return (
      <div className="min-h-screen py-12">
        <div className="container">
          <h1 className="text-4xl font-bold font-mono mb-4">Language not found</h1>
          <Link to="/docs" className="btn">
            Back to Documentation
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen py-12">
      <div className="container">
        <div className="max-w-4xl mx-auto">
          {/* Back link */}
          <Link 
            to="/docs" 
            className="inline-flex items-center text-gray-600 dark:text-gray-400 hover:text-black dark:hover:text-white transition-colors mb-8"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Documentation
          </Link>

          {/* Header */}
          <div className="mb-12">
            <div className="flex items-center mb-4">
              <span className="text-5xl mr-4">{lang.icon}</span>
              <h1 className="text-4xl lg:text-5xl font-bold font-mono">
                {lang.name} Implementation
              </h1>
            </div>
            <p className="text-lg text-gray-600 dark:text-gray-400">
              {lang.description}
            </p>
          </div>

          {/* Installation */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold font-mono mb-4">Installation</h2>
            <div className="code-block">
              <pre>{lang.installation}</pre>
            </div>
          </section>

          {/* Example */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold font-mono mb-4">Quick Example</h2>
            <div className="code-block">
              <pre>{lang.example}</pre>
            </div>
          </section>

          {/* Features */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold font-mono mb-4">Key Features</h2>
            <ul className="space-y-2">
              <li className="flex items-start">
                <span className="mr-2">•</span>
                <span>Full consensus engine implementation</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2">•</span>
                <span>ZeroMQ network transport</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2">•</span>
                <span>Comprehensive test coverage</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2">•</span>
                <span>Production-ready performance</span>
              </li>
            </ul>
          </section>

          {/* Links */}
          <section className="mb-12">
            <h2 className="text-2xl font-bold font-mono mb-4">Resources</h2>
            <div className="flex flex-wrap gap-4">
              <a 
                href={`https://github.com/luxfi/consensus/tree/main/src/${language}`}
                target="_blank"
                rel="noopener noreferrer"
                className="btn"
              >
                View Source
              </a>
              <a 
                href={`https://github.com/luxfi/consensus/tree/main/docs/${language}`}
                target="_blank"
                rel="noopener noreferrer"
                className="btn"
              >
                Full Documentation
              </a>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}