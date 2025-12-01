import { Link } from 'react-router-dom'
import { ArrowRight, Zap, Shield, Cpu, Code2, BarChart3, Globe } from 'lucide-react'
import { motion } from 'framer-motion'
import LuxLogo from '../components/LuxLogo'
import { Button } from '../components/ui/button'

export default function HomePage() {
  const stats = [
    { label: 'Votes/Second', value: '14K+', icon: Zap },
    { label: 'Finality', value: '<10s', icon: Shield },
    { label: 'Throughput', value: '625M', unit: '@40Gbps', icon: BarChart3 },
    { label: 'Languages', value: '5', icon: Code2 },
    { label: 'Engines', value: '6', icon: Cpu },
    { label: 'Test Coverage', value: '100%', icon: Shield },
  ]

  const languages = [
    { 
      name: 'Go', 
      path: 'go',
      description: 'Production blockchain integration',
      icon: '🔷',
      performance: '12,000+ votes/sec'
    },
    { 
      name: 'C', 
      path: 'c',
      description: 'High-performance native implementation',
      icon: '🔧',
      performance: '14,000+ votes/sec'
    },
    { 
      name: 'Rust', 
      path: 'rust',
      description: 'Memory-safe systems programming',
      icon: '🦀',
      performance: '13,500+ votes/sec'
    },
    { 
      name: 'C++', 
      path: 'cpp',
      description: 'Modern C++ with GPU acceleration',
      icon: '⚙️',
      performance: '15,000+ votes/sec'
    },
    { 
      name: 'Python', 
      path: 'python',
      description: 'Research and prototyping',
      icon: '🐍',
      performance: '5,000+ votes/sec'
    },
  ]

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="py-24 lg:py-32">
        <div className="container">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="max-w-4xl"
          >
            <div className="flex items-center space-x-4 mb-6">
              <LuxLogo variant="logo-only" size="xl" outerClx="text-black" />
              <h1 className="text-5xl lg:text-7xl font-bold tracking-tighter">
                CONSENSUS
              </h1>
            </div>
            <p className="text-xl lg:text-2xl text-gray-600 mb-8">
              High-performance Byzantine fault-tolerant consensus implementations 
              across multiple languages.
            </p>
            
            <div className="flex flex-wrap gap-4">
              <Button size="lg" asChild>
                <Link to="/docs" className="inline-flex items-center">
                  DOCUMENTATION
                  <ArrowRight className="ml-2 w-4 h-4" />
                </Link>
              </Button>
              <Button variant="outline" size="lg" asChild>
                <a 
                  href="https://github.com/luxfi/consensus" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center"
                >
                  VIEW ON GITHUB
                  <ArrowRight className="ml-2 w-4 h-4" />
                </a>
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Grid */}
      <section className="py-16 bg-black text-white">
        <div className="container">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="text-center"
              >
                <stat.icon className="w-8 h-8 mx-auto mb-2" />
                <div className="text-2xl font-black font-mono uppercase">{stat.value}</div>
                {stat.unit && (
                  <div className="text-xs opacity-75 font-mono">{stat.unit}</div>
                )}
                <div className="text-sm opacity-75 mt-1 uppercase tracking-wider">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Languages Section */}
      <section className="py-24" id="languages">
        <div className="container">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <h2 className="text-3xl lg:text-4xl font-black uppercase mb-12">
              Choose Your Implementation
            </h2>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {languages.map((lang, index) => (
                <motion.div
                  key={lang.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <Link
                    to={`/docs/${lang.path}`}
                    className="block border-2 border-black bg-white p-6 hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] transition-all group"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="text-3xl">{lang.icon}</div>
                      <ArrowRight className="w-5 h-5 text-gray-400 group-hover:text-black group-hover:translate-x-1 transition-all" />
                    </div>
                    
                    <h3 className="text-xl font-bold mb-2">{lang.name}</h3>
                    <p className="text-gray-600 text-sm mb-3">
                      {lang.description}
                    </p>
                    <div className="text-xs font-mono text-gray-500">
                      {lang.performance}
                    </div>
                  </Link>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Code Example */}
      <section className="py-24 border-t border-gray-200 bg-gray-50">
        <div className="container">
          <h2 className="text-3xl lg:text-4xl font-bold font-mono mb-12">
            Quick Start
          </h2>
          
          <div className="code-block">
            <pre className="language-go">
{`// Initialize consensus with Snowball engine
params := ConsensusParams{
    K:               20,
    AlphaPreference: 15,
    AlphaConfidence: 15,
    Beta:            20,
}

consensus := NewConsensus(SNOWBALL, params)

// Add block to consensus
consensus.Add(block)

// Process votes until consensus
for i := 0; i < 20; i++ {
    vote := Vote{
        NodeID:   i,
        BlockID:  0x1234,
        VoteType: VOTE_PREFER,
    }
    consensus.ProcessVote(vote)
}

// Check if block achieved consensus
if consensus.IsAccepted(0x1234) {
    fmt.Println("✓ Consensus achieved!")
}`}
            </pre>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-24">
        <div className="container">
          <h2 className="text-3xl lg:text-4xl font-bold font-mono mb-12">
            Core Features
          </h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: Shield,
                title: 'Byzantine Fault Tolerance',
                description: 'Tolerates up to f < n/3 malicious nodes with proven safety guarantees'
              },
              {
                icon: Zap,
                title: 'High Performance',
                description: 'Optimized implementations achieving 14,000+ votes/second'
              },
              {
                icon: Globe,
                title: 'ZeroMQ Transport',
                description: 'Binary protocol over ZeroMQ for efficient network communication'
              },
              {
                icon: Shield,
                title: 'Post-Quantum Ready',
                description: 'PostQuantum engine with ML-KEM and ML-DSA cryptography'
              },
              {
                icon: Cpu,
                title: 'Multiple Engines',
                description: 'Snowball, Avalanche, Snowflake, DAG, Chain, and PostQuantum'
              },
              {
                icon: BarChart3,
                title: 'Comprehensive Testing',
                description: '100% test coverage with parity testing across implementations'
              },
            ].map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="flex flex-col items-center text-center"
              >
                <feature.icon className="w-12 h-12 mb-4 text-gray-400" />
                <h3 className="text-lg font-bold mb-2">{feature.title}</h3>
                <p className="text-sm text-gray-600">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}